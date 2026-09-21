from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from ..model.architecture import SimpleAttentionNetwork, TransformerConfig
from ..model.checkpoints import read_checkpoint


def compute_teacher_cache(
    model: SimpleAttentionNetwork,
    params: Any,
    input_ids: np.ndarray,
    target_mask: np.ndarray,
    temperature: float = 2.0,
    top_k: int = 32,
    batch_size: int = 8,
) -> Dict[str, np.ndarray]:
    """Precompute and store top-K teacher logits, token IDs, and residual mass."""
    num_samples, seq_len = input_ids.shape
    all_top_indices = []
    all_top_logits = []
    all_other_mass = []

    @jax.jit
    def teacher_step(ids_chunk):
        logits = model.apply({"params": params}, ids_chunk, quant=False)
        scaled_logits = logits / temperature
        top_l, top_i = jax.lax.top_k(scaled_logits, top_k)
        
        # Exact softmax probability distribution over vocabulary
        probs = jax.nn.softmax(scaled_logits, axis=-1)
        top_p = jnp.take_along_axis(probs, top_i, axis=-1)
        remainder_mass = jnp.maximum(1.0 - jnp.sum(top_p, axis=-1, keepdims=True), 1e-8)
        return top_i.astype(jnp.int32), top_l.astype(jnp.float32), remainder_mass.astype(jnp.float32)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        chunk = jnp.asarray(input_ids[start:end])
        top_i, top_l, rem = teacher_step(chunk)
        all_top_indices.append(np.asarray(top_i))
        all_top_logits.append(np.asarray(top_l))
        all_other_mass.append(np.asarray(rem))

    top_indices = np.concatenate(all_top_indices, axis=0)
    top_logits = np.concatenate(all_top_logits, axis=0)
    other_mass = np.concatenate(all_other_mass, axis=0)

    return {
        "top_indices": top_indices,
        "top_logits": top_logits,
        "other_mass": other_mass,
    }


def compute_distillation_kl(
    student_logits: jnp.ndarray,
    top_indices: jnp.ndarray,
    top_teacher_logits: jnp.ndarray,
    other_teacher_mass: jnp.ndarray,
    temperature: float = 2.0,
) -> jnp.ndarray:
    """Compute KL(teacher || student) over top-32 tokens plus residual other bucket."""
    scaled_student_logits = student_logits / temperature
    student_log_probs = jax.nn.log_softmax(scaled_student_logits, axis=-1)
    student_top_logp = jnp.take_along_axis(student_log_probs, top_indices, axis=-1)

    # Student probability mass in the other bucket
    student_top_probs = jnp.exp(student_top_logp)
    student_other_prob = jnp.maximum(1.0 - jnp.sum(student_top_probs, axis=-1, keepdims=True), 1e-8)
    student_other_logp = jnp.log(student_other_prob)

    # Teacher distribution over top-32 + other
    teacher_top_probs = jax.nn.softmax(top_teacher_logits, axis=-1) * (1.0 - other_teacher_mass)
    teacher_top_logp = jnp.log(jnp.maximum(teacher_top_probs, 1e-8))
    teacher_other_logp = jnp.log(jnp.maximum(other_teacher_mass, 1e-8))

    # KL = sum P_t * (log P_t - log P_s)
    kl_top = jnp.sum(teacher_top_probs * (teacher_top_logp - student_top_logp), axis=-1)
    kl_other = (other_teacher_mass * (teacher_other_logp - student_other_logp)).squeeze(-1)
    return kl_top + kl_other


def save_teacher_cache(cache: Dict[str, np.ndarray], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        top_indices=cache["top_indices"],
        top_logits=cache["top_logits"].astype(np.float16),
        other_mass=cache["other_mass"].astype(np.float16),
    )


def load_teacher_cache(cache_path: str) -> Dict[str, np.ndarray]:
    data = np.load(cache_path)
    return {
        "top_indices": data["top_indices"],
        "top_logits": data["top_logits"].astype(np.float32),
        "other_mass": data["other_mass"].astype(np.float32),
    }
