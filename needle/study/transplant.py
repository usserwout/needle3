from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from ..model.architecture import (
    TransformerConfig,
    ladder_slice,
    ladder_config,
    SimpleAttentionNetwork,
)
from ..model.checkpoints import read_checkpoint, write_checkpoint
from .config import RUNS, RunConfig, StudyConfig, BASE_COMMIT, DEFAULT_BASE_CHECKPOINT


def compute_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "missing"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def count_parameters(params: Any) -> Tuple[int, int]:
    """Calculate total and active parameter count."""
    flat = jax.tree_util.tree_leaves(params)
    total = sum(int(np.prod(p.shape)) for p in flat)
    return total, total


def transplant_i1_8(params: Dict[str, Any], config: TransformerConfig) -> Tuple[Dict[str, Any], TransformerConfig]:
    """Transplant 8-layer slice to Idea 1: headwise gating and block-untied Hadamard MLP."""
    new_params = copy.deepcopy(params)
    new_config = copy.deepcopy(config)
    new_config.attention_gate = "headwise"
    new_config.hada_factor_mode = "block_untied"

    layers = new_params["stack"]["layers"]
    block = layers.get("block", layers)
    self_attn = block["self_attn"]
    old_gate_kernel = np.asarray(self_attn["gate_proj"]["kernel"])  # (8, 768, 768)
    L, d_in, d_out = old_gate_kernel.shape
    num_heads = config.num_heads
    v_hd = d_out // num_heads

    # Reshape (8, 768, 12, 64) and mean pool over the 64 value channels
    reshaped_gate = old_gate_kernel.reshape(L, d_in, num_heads, v_hd)
    new_gate_kernel = jnp.asarray(np.mean(reshaped_gate, axis=-1))  # (8, 768, 12)
    self_attn["gate_proj"]["kernel"] = new_gate_kernel

    # Untie Hadamard Kronecker factors
    hadamard = block["hadamard_mlp"]
    for s in (1, 2, 3):
        old_wa = np.asarray(hadamard[f"w{s}a"])  # (8, 32, 32)
        old_wb = np.asarray(hadamard[f"w{s}b"])  # (8, 32, 32)
        bb, ba = old_wa.shape[1], old_wa.shape[2]
        # wa: (8, bb, ba, ba), wb: (8, ba, bb, bb)
        new_wa = jnp.asarray(np.broadcast_to(old_wa[:, None, :, :], (L, bb, ba, ba)))
        new_wb = jnp.asarray(np.broadcast_to(old_wb[:, None, :, :], (L, ba, bb, bb)))
        hadamard[f"w{s}a"] = new_wa
        hadamard[f"w{s}b"] = new_wb

    return new_params, new_config


def fit_ridge_projection(X: np.ndarray, Y: np.ndarray, alpha_factor: float = 1e-4) -> np.ndarray:
    """Solve min_W ||X W - Y||^2 + lambda ||W||^2 using ridge regression."""
    # Checkpoint Engram weights are float16. Accumulating the 40,000-row
    # normal equations in float16 overflows and NumPy does not support a
    # float16 linear solve. Calibration is an offline numerical operation, so
    # promote both operands before any multiplication.
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("ridge calibration inputs must be finite")
    d_in = X.shape[1]
    XtX = X.T @ X
    trace_val = float(np.trace(XtX))
    reg = alpha_factor * trace_val / max(d_in, 1)
    ridge_matrix = XtX + reg * np.eye(d_in, dtype=X.dtype)
    XtY = X.T @ Y
    W = np.linalg.solve(ridge_matrix, XtY)
    if not np.all(np.isfinite(W)):
        raise ValueError("ridge calibration produced non-finite weights")
    return W


def fetch_engram_rows(tables: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Fetch and concatenate all Engram table rows for [sample, table] indices."""
    tables = np.asarray(tables)
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.shape[1] != tables.shape[0]:
        raise ValueError(
            f"Engram indices must have shape [N, {tables.shape[0]}], got {indices.shape}"
        )
    if np.any(indices < 0) or np.any(indices >= tables.shape[1]):
        raise ValueError("Engram calibration indices contain an out-of-range slot")
    fetched = tables[np.arange(tables.shape[0])[None, :], indices]
    # Projection and ridge-fit products must accumulate above float16 range.
    return fetched.reshape(indices.shape[0], -1).astype(np.float32, copy=False)


def transplant_i2_12(
    params: Dict[str, Any],
    config: TransformerConfig,
    ngram_samples: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[Dict[str, Any], TransformerConfig, Dict[str, Any]]:
    """Transplant 12-layer slice to Idea 2: two shared Engram banks with refitted readers."""
    new_params = copy.deepcopy(params)
    new_config = copy.deepcopy(config)
    num_sites = len(config.engram_layers)
    assert num_sites == 3, f"Expected 3 Engram sites at 12L, got {num_sites}"

    old_tables = [np.asarray(params[f"engrams_{i}"]["embedding"]) for i in range(num_sites)]
    old_key_proj = [np.asarray(params[f"engrams_{i}"]["key_proj"]["kernel"]) for i in range(num_sites)]
    old_val_proj = [np.asarray(params[f"engrams_{i}"]["value_proj"]["kernel"]) for i in range(num_sites)]
    old_taps = [np.asarray(params[f"engrams_{i}"]["taps"]) for i in range(num_sites)]

    candidate_partitions = [
        ((0, 1), 2, (0, 0, 1)),
        ((0, 2), 1, (0, 1, 0)),
        ((1, 2), 0, (0, 1, 1)),
    ]

    if ngram_samples is None or not all(i in ngram_samples for i in range(num_sites)):
        raise ValueError(
            "I2-12 requires 50,000 real valid n-gram index rows per Engram site; "
            "refusing to choose a synthetic or uncalibrated partition"
        )
    for site, samples in ngram_samples.items():
        if np.asarray(samples).shape[0] < 50000:
            raise ValueError(f"Engram site {site} has fewer than 50,000 calibration rows")

    selected_partition_idx = 0
    calibration_metrics = {"partition_scores": []}
    best_score = float("inf")
    fitted_by_partition = {}
    for idx, (pair, single, bank_ids) in enumerate(candidate_partitions):
        p1, p2 = pair
        avg_bank = 0.5 * (old_tables[p1] + old_tables[p2])
        scores = []
        fitted = {}
        for site_idx in (p1, p2):
            fit_idx = np.asarray(ngram_samples[site_idx][:40000], dtype=np.int64)
            eval_idx = np.asarray(ngram_samples[site_idx][40000:50000], dtype=np.int64)

            orig_e_fit = fetch_engram_rows(old_tables[site_idx], fit_idx)
            orig_k_fit = orig_e_fit @ old_key_proj[site_idx]
            orig_v_fit = orig_e_fit @ old_val_proj[site_idx]
            cand_e_fit = fetch_engram_rows(avg_bank, fit_idx)
            cand_e_eval = fetch_engram_rows(avg_bank, eval_idx)
            orig_e_eval = fetch_engram_rows(old_tables[site_idx], eval_idx)
            orig_k_eval = orig_e_eval @ old_key_proj[site_idx]
            orig_v_eval = orig_e_eval @ old_val_proj[site_idx]

            new_w_k = fit_ridge_projection(cand_e_fit, orig_k_fit)
            new_w_v = fit_ridge_projection(cand_e_fit, orig_v_fit)
            fitted[site_idx] = (new_w_k, new_w_v)

            pred_k = cand_e_eval @ new_w_k
            pred_v = cand_e_eval @ new_w_v
            mse_k = float(np.mean((pred_k - orig_k_eval) ** 2) / (np.var(orig_k_eval) + 1e-8))
            mse_v = float(np.mean((pred_v - orig_v_eval) ** 2) / (np.var(orig_v_eval) + 1e-8))
            scores.append(0.5 * (mse_k + mse_v))

        total_score = float(np.mean(scores))
        fitted_by_partition[idx] = fitted
        calibration_metrics["partition_scores"].append({"partition": bank_ids, "score": total_score})
        if total_score < best_score:
            best_score = total_score
            selected_partition_idx = idx

    pair, single, chosen_bank_ids = candidate_partitions[selected_partition_idx]
    new_config.engram_bank_ids = chosen_bank_ids

    # Build new engram banks and readers
    p1, p2 = pair
    bank_0_embedding = jnp.asarray(0.5 * (old_tables[p1] + old_tables[p2]))
    bank_1_embedding = jnp.asarray(old_tables[single])

    for i in range(num_sites):
        new_params.pop(f"engrams_{i}", None)

    new_params["engram_banks_0"] = {"embedding": bank_0_embedding}
    new_params["engram_banks_1"] = {"embedding": bank_1_embedding}

    for i in range(num_sites):
        fitted = fitted_by_partition[selected_partition_idx].get(i)
        key_kernel = fitted[0] if fitted is not None else old_key_proj[i]
        value_kernel = fitted[1] if fitted is not None else old_val_proj[i]
        new_params[f"engram_readers_{i}"] = {
            "key_proj": {"kernel": jnp.asarray(key_kernel)},
            "value_proj": {"kernel": jnp.asarray(value_kernel)},
            "taps": jnp.asarray(old_taps[i]),
        }

    calibration_metrics["selected_partition"] = list(chosen_bank_ids)
    return new_params, new_config, calibration_metrics


def transplant_i3_12(
    params: Dict[str, Any],
    config: TransformerConfig,
) -> Tuple[Dict[str, Any], TransformerConfig, List[int]]:
    """Transplant 12-layer slice to Idea 3: Cross-Layer Attention with local layer pairs."""
    new_params = copy.deepcopy(params)
    new_config = copy.deepcopy(config)
    cla_pairs = ((0, 1), (3, 4), (6, 7), (9, 10))
    new_config.cla_pairs = cla_pairs

    inactive_consumers = [consumer for _, consumer in cla_pairs]
    return new_params, new_config, inactive_consumers


def transplant_run(
    run_id: str,
    base_checkpoint_path: str = DEFAULT_BASE_CHECKPOINT,
    output_dir: str = "study_runs",
    ngram_samples: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Transplant base checkpoint slice to target study run."""
    if run_id not in RUNS:
        raise ValueError(f"Unknown run ID: {run_id}. Valid runs: {list(RUNS.keys())}")

    run_cfg = RUNS[run_id]
    ckpt = read_checkpoint(base_checkpoint_path)
    base_params = ckpt["params"]
    raw_config = ckpt["config"]
    base_config = (
        raw_config
        if isinstance(raw_config, TransformerConfig)
        else TransformerConfig.from_saved(dict(raw_config))
    )

    sliced_params = ladder_slice(base_params, base_config, run_cfg.depth)
    sliced_config = ladder_config(base_config, run_cfg.depth)

    metadata: Dict[str, Any] = {
        "run_id": run_id,
        "depth": run_cfg.depth,
        "base_commit": BASE_COMMIT,
        "base_checkpoint_sha256": compute_sha256(base_checkpoint_path),
        "selected_original_layers": list(run_cfg.original_layers),
        "purpose": run_cfg.purpose,
        "attention_gate": run_cfg.attention_gate,
        "hada_factor_mode": run_cfg.hada_factor_mode,
        "engram_bank_ids": list(run_cfg.engram_bank_ids),
        "cla_pairs": [list(p) for p in run_cfg.cla_pairs],
        "teacher_run": run_cfg.teacher_run,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path):
        metadata["dataset_manifest_sha256"] = compute_sha256(manifest_path)

    if run_id in ("R8", "C8", "R12", "C12"):
        final_params = sliced_params
        final_config = sliced_config
    elif run_id == "I1-8":
        final_params, final_config = transplant_i1_8(sliced_params, sliced_config)
    elif run_id == "I2-12":
        final_params, final_config, calib = transplant_i2_12(sliced_params, sliced_config, ngram_samples)
        metadata["engram_calibration"] = calib
    elif run_id == "I3-12":
        final_params, final_config, inactive = transplant_i3_12(sliced_params, sliced_config)
        metadata["inactive_cla_consumers"] = inactive
    else:
        raise ValueError(f"Unhandled run ID: {run_id}")

    total_params, active_params = count_parameters(final_params)
    if run_id == "I3-12":
        # Consumer K/V tensors are physically present in scan but inactive
        qk_hd = final_config.qk_head_dim or (final_config.d_model // final_config.num_heads)
        v_hd = final_config.v_head_dim or (final_config.d_model // final_config.num_heads)
        consumer_kv_per_layer = (
            final_config.d_model * final_config.num_kv_heads * qk_hd +
            final_config.d_model * final_config.num_kv_heads * v_hd +
            final_config.qkv_conv_taps * final_config.num_kv_heads * (qk_hd + v_hd)
        )
        inactive_count = len(run_cfg.cla_pairs) * consumer_kv_per_layer
        active_params = total_params - inactive_count

    metadata["total_parameters"] = total_params
    metadata["active_parameters"] = active_params

    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    out_ckpt_path = os.path.join(run_dir, "checkpoint.safetensors")
    write_checkpoint(out_ckpt_path, {
        "format_version": 2,
        "params": final_params,
        "config": dict(vars(final_config)),
        "run": metadata,
    })

    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
