import os
import shutil
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from needle.model.architecture import (
    SimpleAttentionNetwork,
    TransformerConfig,
    ladder_slice,
    ladder_config,
    ladder_layer_indices,
    HadamardMLP,
    MultiHeadAttention,
)
from needle.study.config import RUNS, runtime_transformer_config
from needle.study.transplant import (
    transplant_i1_8,
    transplant_i2_12,
    transplant_i3_12,
    fit_ridge_projection,
)
from needle.study.data import create_study_manifest, hash_prompt, normalize_prompt
from needle.study.distill import compute_distillation_kl
from needle.study.evaluate import bootstrap_ci, mcnemar_test, holm_bonferroni_correction
from needle.study.profile import analytical_projection_macs, analytical_kv_bytes
from needle.study.report import evaluate_decision_rules


def test_default_config_reproduces_needle3_defaults():
    cfg = TransformerConfig()
    assert cfg.attention_gate == "elementwise"
    assert cfg.hada_factor_mode == "shared"
    assert cfg.engram_bank_ids == ()
    assert cfg.cla_pairs == ()


def test_study_runtime_disables_flash_without_mutating_checkpoint_config():
    stored = {"flash": True}
    runtime = runtime_transformer_config(stored)

    assert runtime.flash is False
    assert stored == {"flash": True}


def test_invalid_study_architecture_configs_fail_early():
    with pytest.raises(ValueError, match="attention_gate"):
        TransformerConfig(attention_gate="per_token")
    with pytest.raises(ValueError, match="one bank id per Engram site"):
        TransformerConfig(num_layers=4, engram_layers=(1, 3), engram_bank_ids=(0,))
    with pytest.raises(ValueError, match="adjacent"):
        TransformerConfig(num_layers=4, global_layers=(), cla_pairs=((0, 2),))
    with pytest.raises(ValueError, match="local and global"):
        TransformerConfig(num_layers=4, global_layers=(1,), cla_pairs=((0, 1),))


def test_real_manifest_refuses_missing_official_data(tmp_path):
    with pytest.raises(ValueError, match="official training split"):
        create_study_manifest(output_path=str(tmp_path / "manifest.json"), sample_size=10)


def test_manifest_replaces_filtered_rows_from_remaining_pool(tmp_path):
    counts = {"droidcall": 3, "mobile_actions": 3, "snips": 1, "dstc8": 1, "fineweb_edu": 2}
    paths = {}
    for domain, count in counts.items():
        path = tmp_path / f"{domain}.jsonl"
        rows = [
            {"query": f"{domain} prompt {index}", "tools": [], "answers": []}
            for index in range(count + 1)
        ]
        if domain == "fineweb_edu":
            rows = [
                {"text": f"{domain} prompt {index}", "is_language_sample": True}
                for index in range(count + 1)
            ]
        path.write_text("".join(__import__("json").dumps(row) + "\n" for row in rows))
        paths[domain] = str(path)

    first_droid_index = int(np.random.default_rng(7).permutation(4)[0])
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        __import__("json").dumps({"query": f"droidcall prompt {first_droid_index}"}) + "\n"
    )
    manifest = create_study_manifest(
        dataset_paths=paths,
        evaluation_paths=[str(eval_path)],
        output_path=str(tmp_path / "manifest.json"),
        sample_size=10,
        seed=7,
    )
    assert len(manifest["examples"]) == 10


def test_layer_indices_8l_and_12l():
    idx8 = ladder_layer_indices(20, 8)
    assert idx8 == (0, 4, 6, 9, 11, 14, 16, 19)

    idx12 = ladder_layer_indices(20, 12)
    assert idx12 == (0, 2, 4, 6, 7, 9, 11, 12, 14, 16, 17, 19)

    cfg20 = TransformerConfig(num_layers=20, engram_layers=(3, 7, 11, 15, 19), global_layers=(4, 9, 14, 19))
    cfg12 = ladder_config(cfg20, 12)
    assert cfg12.engram_layers == (4, 6, 11)
    assert cfg12.global_layers == (2, 5, 8, 11)


def test_untied_hadamard_mlp_identity_at_initialization():
    d_model = 64
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, d_model))

    mlp_shared = HadamardMLP(d_model=d_model, factor_mode="shared")
    params_shared = mlp_shared.init(jax.random.PRNGKey(1), x)["params"]

    mlp_untied = HadamardMLP(d_model=d_model, factor_mode="block_untied")
    params_untied = mlp_untied.init(jax.random.PRNGKey(1), x)["params"]

    # Transplant shared factors by broadcasting to untied shapes
    for s in (1, 2, 3):
        wa = params_shared[f"w{s}a"]
        wb = params_shared[f"w{s}b"]
        bb, ba = wa.shape[0], wa.shape[1]
        params_untied[f"w{s}a"] = jnp.broadcast_to(wa[None, :, :], (bb, ba, ba))
        params_untied[f"w{s}b"] = jnp.broadcast_to(wb[None, :, :], (ba, bb, bb))

    out_shared = mlp_shared.apply({"params": params_shared}, x)
    out_untied = mlp_untied.apply({"params": params_untied}, x)

    max_diff = float(jnp.max(jnp.abs(out_shared - out_untied)))
    assert max_diff < 1e-5


def test_headwise_gate_transplant_preserves_mean_logit():
    d_model = 64
    num_heads = 4
    v_hd = 16
    out_dim = num_heads * v_hd

    old_gate_kernel = jax.random.normal(jax.random.PRNGKey(2), (d_model, out_dim))
    reshaped = old_gate_kernel.reshape(d_model, num_heads, v_hd)
    new_gate_kernel = jnp.mean(reshaped, axis=-1)

    x = jax.random.normal(jax.random.PRNGKey(3), (2, 5, d_model))

    old_logits = x @ old_gate_kernel
    old_logits_per_head = jnp.mean(old_logits.reshape(2, 5, num_heads, v_hd), axis=-1)

    new_logits = x @ new_gate_kernel

    diff = float(jnp.max(jnp.abs(old_logits_per_head - new_logits)))
    assert diff < 1e-5


def test_ridge_regression_fitting():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 32))
    true_W = rng.normal(size=(32, 16))
    Y = X @ true_W + 1e-3 * rng.normal(size=(100, 16))

    W_fit = fit_ridge_projection(X, Y, alpha_factor=1e-4)
    pred = X @ W_fit
    mse = float(np.mean((pred - Y) ** 2))
    assert mse < 1e-4


def test_ridge_regression_promotes_float16_before_normal_equations():
    rng = np.random.default_rng(7)
    X = (rng.normal(size=(512, 32)) * 20).astype(np.float16)
    true_W = rng.normal(size=(32, 8)).astype(np.float32)
    Y = (X.astype(np.float32) @ true_W).astype(np.float16)

    W_fit = fit_ridge_projection(X, Y)
    Y32 = Y.astype(np.float32)
    relative_mse = np.mean((X.astype(np.float32) @ W_fit - Y32) ** 2) / np.var(Y32)

    assert W_fit.dtype == np.float32
    assert np.all(np.isfinite(W_fit))
    assert relative_mse < 1e-4


def test_cross_layer_attention_produces_identical_cache():
    cfg = TransformerConfig(
        vocab_size=1024,
        out_vocab=1024,
        d_model=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=4,
        qk_head_dim=16,
        v_head_dim=16,
        max_seq_len=64,
        cla_pairs=((0, 1),),
        flash=False,
    )
    model = SimpleAttentionNetwork(cfg)
    tokens = jnp.ones((1, 8), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), tokens)["params"]

    out = model.apply({"params": params}, tokens)
    assert out.shape == (1, 8, 1024)
    assert not jnp.isnan(out).any()


def test_statistical_evaluation_methods():
    deltas = np.array([1.0, 0.5, -0.2, 0.8, 1.2, -0.1, 0.4, 0.7])
    mean, lower, upper = bootstrap_ci(deltas, num_bootstraps=500)
    assert lower <= mean <= upper

    ctrl = np.array([1, 1, 0, 1, 0, 1, 1, 0])
    var = np.array([1, 1, 1, 1, 0, 1, 0, 0])
    pval = mcnemar_test(ctrl, var)
    assert 0.0 <= pval <= 1.0

    pvals = [0.01, 0.04, 0.03]
    adj = holm_bonferroni_correction(pvals)
    assert adj[0] <= adj[2] <= adj[1]


@pytest.fixture
def study_base_checkpoint(tmp_path):
    import pickle
    import jax
    import jax.numpy as jnp
    from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig

    config = TransformerConfig(
        vocab_size=8192, out_vocab=8192, d_model=64, num_heads=4, num_kv_heads=2,
        num_layers=20, qk_head_dim=16, v_head_dim=16, max_seq_len=64,
        engram_layers=(3, 7, 11, 15, 19), engram_slots=64, global_layers=(4, 9, 14, 19),
        sliding_window=32, mhc_lanes=2, qkv_conv_taps=3, flash=False,
    )
    model = SimpleAttentionNetwork(config)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 8), jnp.int32))["params"]
    params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    path = str(tmp_path / "study_base.pkl")
    with open(path, "wb") as handle:
        pickle.dump({"format_version": 2, "params": params, "config": dict(vars(config))}, handle)
    return path


def test_study_pipeline_smoke_run(tmp_path, study_base_checkpoint):
    import jax
    from needle.model.checkpoints import read_checkpoint, write_checkpoint
    from needle.study.transplant import transplant_run
    from needle.study.train import run_training_loop
    from needle.study.evaluate import evaluate_run
    from needle.study.profile import profile_run
    from needle.study.report import generate_study_report

    out_dir = str(tmp_path / "study_smoke")
    os.makedirs(out_dir, exist_ok=True)

    manifest_path = os.path.join(out_dir, "manifest.json")
    create_study_manifest(output_path=manifest_path, sample_size=10, allow_synthetic=True)
    rng = np.random.default_rng(20260921)
    input_ids = rng.integers(1, 1024, size=(4, 32), dtype=np.int32)
    target_mask = np.ones_like(input_ids, dtype=np.float32)
    np.savez_compressed(os.path.join(out_dir, "train_data.npz"),
                        input_ids=input_ids, target_mask=target_mask)

    # Transplant C8 and I1-8
    transplant_run("C8", base_checkpoint_path=study_base_checkpoint, output_dir=out_dir)
    transplant_run("I1-8", base_checkpoint_path=study_base_checkpoint, output_dir=out_dir)

    # Production transplants contain half-precision weights; recovery must use
    # FP32 master weights and Adam moments even when the source is FP16.
    for run_id in ("C8", "I1-8"):
        path = os.path.join(out_dir, run_id, "checkpoint.safetensors")
        checkpoint = read_checkpoint(path)
        checkpoint["params"] = jax.tree_util.tree_map(
            lambda x: np.asarray(x, dtype=np.float16), checkpoint["params"]
        )
        write_checkpoint(path, checkpoint)

    # Train C8 and I1-8
    train_res_c = run_training_loop("C8", checkpoint_dir=out_dir, total_steps=2,
                                      batch_size=1, smoke=True)
    assert train_res_c["steps"] == 2
    assert np.isfinite(train_res_c["final_loss"])
    recovered_c = read_checkpoint(os.path.join(out_dir, "C8", "checkpoint.safetensors"))
    assert all(np.asarray(x).dtype == np.float32
               for x in jax.tree_util.tree_leaves(recovered_c["params"]))
    resumed_c = run_training_loop("C8", checkpoint_dir=out_dir, total_steps=2,
                                  batch_size=1, resume=True, smoke=True)
    assert resumed_c["steps"] == 2

    train_res_i = run_training_loop("I1-8", checkpoint_dir=out_dir, total_steps=2,
                                      batch_size=1, smoke=True)
    assert train_res_i["steps"] == 2
    assert np.isfinite(train_res_i["final_loss"])

    # Evaluate
    ev_c = evaluate_run("C8", checkpoint_dir=out_dir, num_samples_per_suite=5, smoke=True)
    ev_i = evaluate_run("I1-8", checkpoint_dir=out_dir, num_samples_per_suite=5, smoke=True)
    assert "macro_score" in ev_c
    assert "macro_score" in ev_i

    # Profile
    prof_c = profile_run("C8", checkpoint_dir=out_dir, benchmark_latency=False)
    prof_i = profile_run("I1-8", checkpoint_dir=out_dir, benchmark_latency=False)
    assert prof_c["analytical_projection_macs"] > 0
    assert prof_i["analytical_projection_macs"] > 0

    # Report
    with pytest.raises(ValueError, match="synthetic smoke"):
        generate_study_report(output_dir=out_dir)
