from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from ..model.architecture import SimpleAttentionNetwork, TransformerConfig
from ..model.checkpoints import read_checkpoint
from .config import runtime_transformer_config


def analytical_projection_macs(config: TransformerConfig, assume_cla_skip: bool = True) -> int:
    """Calculate projection MACs per token.

    ``assume_cla_skip`` describes the projected deployment kernel. The current
    homogeneous JAX scan retains and executes consumer K/V projections before
    selecting shared tensors, so executable-JAX accounting must pass False.
    """
    d = config.d_model
    h = config.num_heads
    kv_h = config.num_kv_heads
    qk_hd = config.qk_head_dim or (d // h)
    v_hd = config.v_head_dim or (d // h)
    L = config.num_layers
    num_cla_consumers = len(getattr(config, "cla_pairs", ())) if assume_cla_skip else 0

    # Q projection across all layers
    q_macs = L * (d * h * qk_hd)
    # K and V projections: skipped by CLA consumer layers
    active_kv_layers = L - num_cla_consumers
    k_macs = active_kv_layers * (d * kv_h * qk_hd)
    v_macs = active_kv_layers * (d * kv_h * v_hd)

    # Gate projection: headwise vs elementwise
    if getattr(config, "attention_gate", "elementwise") == "headwise":
        gate_macs = L * (d * h)
    else:
        gate_macs = L * (d * h * v_hd)

    out_macs = L * ((h * v_hd) * d)

    # Hadamard structured transformations (3 stages of 2 factors on 32x32 grid)
    # 3 * 2 * 32 * 32 * 32 = 196,608 MACs per layer
    hada_macs = L * (3 * 2 * 32 * 32 * 32)

    total_macs = q_macs + k_macs + v_macs + gate_macs + out_macs + hada_macs
    return int(total_macs)


def analytical_kv_bytes(config: TransformerConfig, context_length: int, bytes_per_element: int = 1) -> int:
    """Calculate analytical attention KV cache bytes across context lengths."""
    kv_h = config.num_kv_heads
    qk_hd = config.qk_head_dim or (config.d_model // config.num_heads)
    v_hd = config.v_head_dim or (config.d_model // config.num_heads)
    kd = kv_h * qk_hd
    vd = kv_h * v_hd
    entry_bytes_per_token = (kd + vd) * bytes_per_element
    if bytes_per_element == 1:
        # Native int8 KV stores one FP32 scale per group of 32 values.
        entry_bytes_per_token += (kd // 32 + vd // 32) * 4

    window = getattr(config, "sliding_window", 1024) or 1024
    global_layers = set(getattr(config, "global_layers", ()))
    cla_consumers = {c for _, c in getattr(config, "cla_pairs", ())}

    total_bytes = 0
    for l in range(config.num_layers):
        if l in cla_consumers:
            # Consumer reuses producer's cache
            continue
        if l in global_layers:
            # Full context retention
            tokens_stored = context_length
        else:
            # Sliding window retention
            tokens_stored = min(context_length, window)
        total_bytes += tokens_stored * entry_bytes_per_token

    return int(total_bytes)


def measure_prefill_latency(
    model: SimpleAttentionNetwork,
    params: Any,
    seq_len: int,
    vocab_size: int,
    num_warmup: int = 20,
    num_iter: int = 100,
) -> Tuple[float, float]:
    """Measure synchronized batch-1 prefill latency (median and p90 in ms)."""
    x = jnp.ones((1, seq_len), dtype=jnp.int32)

    @jax.jit
    def run_prefill(inp):
        return model.apply({"params": params}, inp, quant=False)

    # Warmup
    for _ in range(num_warmup):
        out = run_prefill(x)
        out.block_until_ready()

    timings = []
    for _ in range(num_iter):
        t0 = time.perf_counter()
        out = run_prefill(x)
        out.block_until_ready()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    median_ms = float(np.median(timings))
    p90_ms = float(np.percentile(timings, 90))
    return median_ms, p90_ms


def profile_run(
    run_id: str,
    checkpoint_dir: str = "study_runs",
    benchmark_latency: bool = True,
) -> Dict[str, Any]:
    """Run efficiency profiling for a screening run."""
    run_dir = os.path.join(checkpoint_dir, run_id)
    ckpt_path = os.path.join(run_dir, "checkpoint.safetensors")
    ckpt_data = read_checkpoint(ckpt_path)
    params = ckpt_data["params"]
    config = runtime_transformer_config(ckpt_data["config"])
    metadata = ckpt_data.get("run") or ckpt_data.get("metadata") or {}

    total_params = metadata.get("total_parameters", 0)
    active_params = metadata.get("active_parameters", total_params)

    # Estimated weight sizes
    # FP16/BF16: 2.0 bytes/param
    # CQ4: ~0.55 bytes/param
    # CQ2: ~0.27 bytes/param
    serialized_sizes_mb = {
        "fp16_mb": (total_params * 2.0) / (1024 * 1024),
        "cq4_mb": (total_params * 0.55) / (1024 * 1024),
        "cq2_mb": (total_params * 0.27) / (1024 * 1024),
    }

    projected_macs_per_token = analytical_projection_macs(config, assume_cla_skip=True)
    executable_macs_per_token = analytical_projection_macs(config, assume_cla_skip=False)

    kv_bytes_by_ctx = {
        ctx: analytical_kv_bytes(config, ctx, bytes_per_element=1)
        for ctx in (128, 512, 1024, 4096)
    }

    latencies = {}
    if benchmark_latency:
        model = SimpleAttentionNetwork(config)
        for s_len in (128, 512, 1024):
            med_ms, p90_ms = measure_prefill_latency(model, params, s_len, config.vocab_size)
            latencies[f"prefill_{s_len}_ms"] = {"median": med_ms, "p90": p90_ms}

    report = {
        "run_id": run_id,
        "total_parameters": total_params,
        "active_parameters": active_params,
        "serialized_sizes_mb": serialized_sizes_mb,
        "analytical_projection_macs": projected_macs_per_token,
        "executable_jax_projection_macs": executable_macs_per_token,
        "analytical_kv_bytes": kv_bytes_by_ctx,
        "latency_measurements": latencies,
    }

    prof_out_path = os.path.join(run_dir, "profile_results.json")
    with open(prof_out_path, "w") as f:
        json.dump(report, f, indent=2)

    return report
