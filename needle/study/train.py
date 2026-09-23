from __future__ import annotations

import json
import os
import pickle
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
import optax

from ..model.architecture import SimpleAttentionNetwork
from ..model.checkpoints import read_checkpoint, write_checkpoint
from .config import RUNS, RunConfig, StudyConfig, runtime_transformer_config
from .distill import compute_distillation_kl, load_teacher_cache
from .token_batches import TokenCursor, next_target_batches


def classify_param_path(path_tuple: Tuple[str, ...], inactive_paths: List[str]) -> str:
    """Classify a parameter into its optimizer group."""
    path_str = "/".join(path_tuple)
    if any(inactive in path_str for inactive in inactive_paths):
        return "inactive"

    last = path_tuple[-1]
    name_str = ".".join(path_tuple).lower()

    if (
        last in ("bias", "b_pre", "b_post", "b_res", "b2", "d1", "d2", "d3", "d4", "attn_gate")
        or "norm" in name_str
        or "scale" in last
        or "tap" in name_str
    ):
        return "no_decay"

    if "engram_bank" in name_str or ("engram" in name_str and "embedding" in name_str):
        return "engram_tables"

    if "embedding" in name_str or "head" in name_str:
        return "embeddings_heads"

    return "backbone"


def build_study_optimizer(
    params: Dict[str, Any],
    total_steps: int,
    warmup_steps: int,
    inactive_paths: List[str],
    lr_backbone: float = 3e-5,
    lr_heads: float = 1e-5,
    lr_engram: float = 3e-6,
    weight_decay: float = 0.1,
) -> Tuple[optax.GradientTransformation, Any]:
    """Build multi-partition AdamW optimizer with learning rate schedules and exclusions."""
    flat_params = flatten_dict(params)
    partition_labels = {}
    for p_tuple in flat_params:
        partition_labels[p_tuple] = classify_param_path(p_tuple, inactive_paths)
    param_labels = unflatten_dict(partition_labels)

    def make_schedule(peak_lr: float):
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=0.0,
        )

    sched_backbone = make_schedule(lr_backbone)
    sched_heads = make_schedule(lr_heads)
    sched_engram = make_schedule(lr_engram)

    transforms = {
        "backbone": optax.adamw(learning_rate=sched_backbone, b1=0.9, b2=0.95, eps=1e-8, weight_decay=weight_decay),
        "embeddings_heads": optax.adamw(learning_rate=sched_heads, b1=0.9, b2=0.95, eps=1e-8, weight_decay=weight_decay),
        "engram_tables": optax.adamw(learning_rate=sched_engram, b1=0.9, b2=0.95, eps=1e-8, weight_decay=weight_decay),
        "no_decay": optax.adamw(learning_rate=sched_backbone, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.0),
        "inactive": optax.set_to_zero(),
    }

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(transforms, param_labels),
    )
    return optimizer, param_labels


def freeze_scanned_cla_consumer_updates(updates: Dict[str, Any], consumers: Tuple[int, ...]):
    """Zero updates for retained-but-inactive K/V slices on the scan layer axis."""
    if not consumers:
        return updates
    frozen_suffixes = {
        ("self_attn", "k_proj", "kernel"),
        ("self_attn", "v_proj", "kernel"),
        ("self_attn", "k_taps"),
        ("self_attn", "v_taps"),
    }
    flat = flatten_dict(updates)
    layer_ids = jnp.asarray(consumers, dtype=jnp.int32)
    for path, update in list(flat.items()):
        if any(tuple(path[-len(suffix):]) == suffix for suffix in frozen_suffixes):
            if getattr(update, "ndim", 0) == 0 or update.shape[0] <= max(consumers):
                raise ValueError(f"cannot freeze CLA consumer slices in {'/'.join(path)}")
            flat[path] = update.at[layer_ids].set(0)
    return unflatten_dict(flat)


def run_training_loop(
    run_id: str,
    checkpoint_dir: str = "study_runs",
    total_steps: int = 100,
    batch_size: int = 16,
    resume: bool = True,
    smoke: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    target_tokens_per_update: Optional[int] = None,
    stop_after_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute recovery training for a specific screening run."""
    run_dir = os.path.join(checkpoint_dir, run_id)
    ckpt_path = os.path.join(run_dir, "checkpoint.safetensors")
    ckpt_data = read_checkpoint(ckpt_path)
    # The published checkpoint is stored in FP16. Adam's 1e-8 epsilon rounds
    # to zero in FP16, and FP16 moment accumulators can overflow. Keep master
    # weights and optimizer state in FP32; model layers still use config.dtype.
    params = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value, dtype=jnp.float32)
        if np.issubdtype(np.asarray(value).dtype, np.floating)
        else jnp.asarray(value),
        ckpt_data["params"],
    )
    config_dict = ckpt_data["config"]
    config = runtime_transformer_config(config_dict)
    metadata = ckpt_data.get("run") or ckpt_data.get("metadata") or {}
    inactive_paths = metadata.get("inactive_parameter_paths", [])
    cla_consumers = tuple(metadata.get("inactive_cla_consumers", ()))

    model = SimpleAttentionNetwork(config)

    warmup_steps = max(1, int(total_steps * 0.05))
    optimizer, param_labels = build_study_optimizer(
        params,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        inactive_paths=inactive_paths,
    )
    opt_state = optimizer.init(params)
    state_path = os.path.join(run_dir, "training_state.pkl")
    start_step = 0
    token_cursor = TokenCursor()
    if resume and os.path.exists(state_path):
        with open(state_path, "rb") as handle:
            saved_state = pickle.load(handle)
        start_step = int(saved_state["step"])
        if saved_state.get("optimizer_precision") != "float32":
            raise ValueError("saved optimizer state predates FP32 training; retransplant and train without --resume")
        if target_tokens_per_update is not None:
            if (saved_state.get("total_steps") != total_steps or
                    saved_state.get("target_tokens_per_update") != target_tokens_per_update):
                raise ValueError("full-study schedule or target-token batch differs from saved state")
            token_cursor = TokenCursor(**saved_state["token_cursor"])
        elif saved_state.get("target_tokens_per_update") is not None:
            raise ValueError("cannot resume full-study state with the pilot batch mode")
        opt_state = saved_state["opt_state"]
        if start_step > total_steps:
            raise ValueError(
                f"saved step {start_step} exceeds requested total_steps={total_steps}"
            )
        if start_step == total_steps:
            return {
                "run_id": run_id,
                "steps": start_step,
                "final_loss": float(metadata.get("final_loss", 0.0)),
                "tokens_per_second": float(metadata.get("tokens_per_second", 0.0)),
            }

    # Load teacher distillation cache
    teacher_run = metadata.get("teacher_run", RUNS[run_id].teacher_run)
    teacher_cache_path = os.path.join(checkpoint_dir, "teachers", f"{teacher_run}_cache.npz")
    has_teacher = os.path.exists(teacher_cache_path)
    teacher_data = load_teacher_cache(teacher_cache_path) if has_teacher else None
    if not has_teacher and not smoke:
        raise FileNotFoundError(
            f"missing frozen teacher cache {teacher_cache_path}; refusing CE-only recovery training"
        )

    # Loss function combining supervised cross-entropy with teacher KL divergence
    def loss_fn(p, ids, mask, top_i, top_l, rem_m):
        logits = model.apply({"params": p}, ids, quant=False)
        shift_logits = logits[:, :-1]
        shift_targets = ids[:, 1:]
        shift_mask = mask[:, 1:]

        ce = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_targets)
        supervised_tokens = jnp.maximum(shift_mask.sum(), 1.0)
        ce_loss = (ce * shift_mask).sum() / supervised_tokens

        if top_i is not None:
            kl = compute_distillation_kl(shift_logits, top_i[:, :-1], top_l[:, :-1], rem_m[:, :-1], temperature=2.0)
            kl_loss = (kl * shift_mask).sum() / supervised_tokens
            total_loss = 0.5 * ce_loss + 0.5 * (4.0) * kl_loss
        else:
            total_loss = ce_loss

        return total_loss, ce_loss

    @jax.jit
    def train_step(p, opt_s, ids, mask, top_i, top_l, rem_m):
        (loss, ce), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, ids, mask, top_i, top_l, rem_m)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        updates = freeze_scanned_cla_consumer_updates(updates, cla_consumers)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_s, loss, ce

    @jax.jit
    def microbatch_grad(p, ids, mask, top_i, top_l, rem_m):
        (loss, ce), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            p, ids, mask, top_i, top_l, rem_m
        )
        return grads, loss, ce

    @jax.jit
    def apply_accumulated_grad(p, opt_s, grads):
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        updates = freeze_scanned_cla_consumer_updates(updates, cla_consumers)
        return optax.apply_updates(p, updates), new_opt_s

    data_path = os.path.join(checkpoint_dir, "train_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"missing {data_path}; run `python -m needle.study prepare` with official data first"
        )
    train_data = np.load(data_path)
    input_ids = np.asarray(train_data["input_ids"], dtype=np.int32)
    target_masks = np.asarray(train_data["target_mask"], dtype=np.float32)
    if len(input_ids) == 0 or input_ids.shape != target_masks.shape:
        raise ValueError("prepared input_ids and target_mask must be non-empty and have equal shapes")
    if target_tokens_per_update is not None and np.any(target_masks[:, 1:].sum(axis=1) == 0):
        raise ValueError("full study cannot train rows without supervised target tokens")
    if stop_after_steps is not None and not start_step < stop_after_steps <= total_steps:
        if start_step == stop_after_steps:
            return {"run_id": run_id, "steps": start_step,
                    "final_loss": float(metadata.get("final_loss", 0.0)),
                    "tokens_per_second": float(metadata.get("tokens_per_second", 0.0))}
        raise ValueError("stop_after_steps must follow the saved step and not exceed total_steps")
    if teacher_data is not None and len(teacher_data["top_indices"]) != len(input_ids):
        raise ValueError("teacher cache does not align with the prepared training examples")

    rng = np.random.default_rng(20260921)
    order = rng.permutation(len(input_ids))
    step_losses = []
    start_time = time.time()
    base_tokens_processed = int(metadata.get("tokens_processed", 0))
    supervised_tokens_processed = 0

    requested_end_step = min(
        total_steps,
        start_step + 2 if smoke else total_steps,
        stop_after_steps if stop_after_steps is not None else total_steps,
    )
    progress_every = 25 if target_tokens_per_update is None else max(1, min(5, total_steps // 100))
    for step in range(start_step, requested_end_step):
        if target_tokens_per_update is None:
            start = (step * batch_size) % len(order)
            batches = [(np.take(order, np.arange(start, start + batch_size), mode="wrap"),
                        target_masks[np.take(order, np.arange(start, start + batch_size), mode="wrap")])]
        else:
            batches, token_cursor = next_target_batches(
                target_masks, order, token_cursor, target_tokens_per_update, batch_size
            )
        gradient_sum = None
        weighted_loss = jnp.asarray(0.0, dtype=jnp.float32)
        weighted_ce = jnp.asarray(0.0, dtype=jnp.float32)
        for batch_indices, batch_mask in batches:
            batch_ids = input_ids[batch_indices]
            count = int(batch_mask[:, 1:].sum())
            supervised_tokens_processed += count
            if teacher_data is not None:
                top_i = jnp.asarray(teacher_data["top_indices"][batch_indices])
                top_l = jnp.asarray(teacher_data["top_logits"][batch_indices])
                rem_m = jnp.asarray(teacher_data["other_mass"][batch_indices])
            else:
                top_i, top_l, rem_m = None, None, None
            if target_tokens_per_update is None:
                params, opt_state, loss_val, ce_val = train_step(
                    params, opt_state, jnp.asarray(batch_ids), jnp.asarray(batch_mask),
                    top_i, top_l, rem_m
                )
            else:
                grads, partial_loss, partial_ce = microbatch_grad(
                    params, jnp.asarray(batch_ids), jnp.asarray(batch_mask), top_i, top_l, rem_m
                )
                weight = count / target_tokens_per_update
                if gradient_sum is None:
                    gradient_sum = jax.tree_util.tree_map(lambda grad: grad * weight, grads)
                else:
                    gradient_sum = jax.tree_util.tree_map(
                        lambda total, grad: total + grad * weight, gradient_sum, grads
                    )
                weighted_loss = weighted_loss + partial_loss * weight
                weighted_ce = weighted_ce + partial_ce * weight
        if target_tokens_per_update is not None:
            params, opt_state = apply_accumulated_grad(params, opt_state, gradient_sum)
            loss_val, ce_val = weighted_loss, weighted_ce
        loss_scalar, ce_scalar = float(loss_val), float(ce_val)
        if not np.isfinite(loss_scalar) or not np.isfinite(ce_scalar):
            raise FloatingPointError(
                f"{run_id} produced nonfinite loss at step {step + 1}: "
                f"loss={loss_scalar}, CE={ce_scalar}; checkpoint was not saved"
            )
        step_losses.append(loss_scalar)

        if (step + 1) % progress_every == 0 or step == requested_end_step - 1:
            msg = (f"Run {run_id} | Step {step+1}/{total_steps} | "
                   f"Targets: {base_tokens_processed + supervised_tokens_processed:,} | "
                   f"Loss: {loss_scalar:.4f} | CE: {ce_scalar:.4f}")
            if progress:
                progress(msg)

        if (step + 1) % 250 == 0:
            metadata["training_steps_completed"] = step + 1
            metadata["tokens_processed"] = base_tokens_processed + supervised_tokens_processed
            write_checkpoint(ckpt_path, {
                "format_version": 2,
                "params": params,
                "config": config_dict,
                "run": metadata,
            })
            with open(state_path, "wb") as handle:
                pickle.dump({"step": step + 1, "opt_state": opt_state,
                             "optimizer_precision": "float32",
                             "total_steps": total_steps,
                             "target_tokens_per_update": target_tokens_per_update,
                             "token_cursor": vars(token_cursor)}, handle)

    elapsed = time.time() - start_time
    completed_steps = requested_end_step
    tokens_processed = base_tokens_processed + supervised_tokens_processed
    tokens_per_second = supervised_tokens_processed / max(elapsed, 1e-4)

    # Save recovered checkpoint
    metadata["training_steps_completed"] = completed_steps
    metadata["tokens_processed"] = tokens_processed
    metadata["final_loss"] = float(step_losses[-1]) if step_losses else 0.0
    metadata["tokens_per_second"] = tokens_per_second

    write_checkpoint(ckpt_path, {
        "format_version": 2,
        "params": params,
        "config": config_dict,
        "run": metadata,
    })
    with open(state_path, "wb") as handle:
        pickle.dump({"step": completed_steps, "opt_state": opt_state,
                     "optimizer_precision": "float32",
                     "total_steps": total_steps,
                     "target_tokens_per_update": target_tokens_per_update,
                     "token_cursor": vars(token_cursor)}, handle)

    summary = {
        "run_id": run_id,
        "steps": completed_steps,
        "final_loss": metadata["final_loss"],
        "tokens_per_second": tokens_per_second,
    }
    return summary
