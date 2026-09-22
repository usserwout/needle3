from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import jax.numpy as jnp

from ..model.architecture import TransformerConfig, engram_geometry, engram_indices

from ..model.tokenizer import (
    get_tokenizer,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    IM_START,
    IM_END,
    THINK_START,
    THINK_END,
    TOOLS_START,
    TOOLS_END,
    TOOL_CALL_START,
    TOOL_CALL_END,
)
from .config import STUDY_SEED, StudyConfig


def normalize_prompt(text: str) -> str:
    """Normalize text for exact duplicate detection and prompt hygiene."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return text.lower()


def hash_prompt(text: str) -> str:
    norm = normalize_prompt(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def render_study_example(example: Dict[str, Any]) -> Tuple[str, str, bool]:
    """Render a structured tool-call or text example into prompt and target.
    
    Returns (prompt, target, is_general_language).
    """
    if example.get("is_language_sample", False):
        text = example.get("text", "")
        return "", text, True

    tools = example.get("tools", [])
    tools_json = (
        tools
        if isinstance(tools, str)
        else json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    )
    answers = example.get("answers", example.get("function_calls", []))
    answers_json = (
        answers
        if isinstance(answers, str)
        else json.dumps(answers, separators=(",", ":"), ensure_ascii=False)
    )
    reasoning = (example.get("reasoning") or "").strip()
    system = (example.get("system") or "").strip()
    prefix = IM_START + "system\n" + system + IM_END + "\n" if system else ""
    prompt = (
        prefix
        + IM_START
        + "user\n"
        + TOOLS_START
        + tools_json
        + TOOLS_END
        + "\n"
        + example.get("query", "")
        + IM_END
        + "\n"
        + IM_START
        + "assistant\n"
    )
    think = THINK_START + "\n" + reasoning + "\n" + THINK_END + "\n" if reasoning else ""
    target = think + TOOL_CALL_START + answers_json + TOOL_CALL_END + IM_END
    return prompt, target, False


def encode_study_example(
    tokenizer: Any,
    example: Dict[str, Any],
    max_len: int = 512,
) -> Tuple[List[int], List[float]]:
    """Encode an example into token IDs and response-only supervised loss mask."""
    prompt, target, is_general = render_study_example(example)

    if is_general:
        target_ids = tokenizer.encode(target)
        ids = [BOS_ID] + target_ids + [EOS_ID]
        # Supervise all next-token positions for general language text
        mask = [1.0] * len(ids)
    else:
        prompt_ids = tokenizer.encode(prompt)
        target_ids = tokenizer.encode(target)
        ids = [BOS_ID] + prompt_ids + target_ids + [EOS_ID]
        # Mask prompt positions; compute loss strictly on response tokens
        mask = [0.0] * (1 + len(prompt_ids)) + [1.0] * (len(target_ids) + 1)

    ids = ids[:max_len]
    mask = mask[:max_len]
    pad = max_len - len(ids)
    if pad > 0:
        ids = ids + [PAD_ID] * pad
        mask = mask + [0.0] * pad
    return ids, mask


def pack_examples(
    tokenized_items: List[Tuple[List[int], List[float]]],
    seq_len: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically pack items into fixed-length sequence arrays."""
    seqs = np.array([item[0] for item in tokenized_items], dtype=np.int32)
    masks = np.array([item[1] for item in tokenized_items], dtype=np.float32)
    return seqs, masks


def collect_engram_calibration_indices(
    input_ids: np.ndarray,
    config: TransformerConfig,
    count: int = 50000,
) -> np.ndarray:
    """Collect deterministic valid n-gram slot indices from prepared training text."""
    orders, heads, _ = engram_geometry(config)
    all_indices = np.asarray(engram_indices(
        jnp.asarray(input_ids), orders, heads, config.engram_slots,
        getattr(config, "engram_seed_heads", 0),
    ))
    positions = np.arange(input_ids.shape[1])[None, :]
    valid = (input_ids != PAD_ID) & (positions >= max(orders) - 1)
    rows = all_indices[valid]
    if len(rows) < count:
        raise ValueError(
            f"prepared training data yields {len(rows)} valid n-gram rows; {count} are required"
        )
    return np.asarray(rows[:count], dtype=np.int32)


def build_synthetic_task_examples(count: int, domain: str, seed: int = STUDY_SEED) -> List[Dict[str, Any]]:
    """Deterministic synthetic example generator for offline tests and smoke runs."""
    rng = np.random.default_rng(seed)
    examples = []
    actions = ["open", "close", "set", "query", "book", "find", "cancel"]
    slots = ["lights", "thermostat", "music", "alarm", "flight", "reminder", "camera"]

    for i in range(count):
        act = actions[int(rng.integers(0, len(actions)))]
        slot = slots[int(rng.integers(0, len(slots)))]
        query = f"{act} the {slot} for room {i}"
        tools = [{
            "name": f"{act}_{slot}",
            "description": f"Automate {act} for {slot}",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
        }]
        answers = [{"name": f"{act}_{slot}", "arguments": {"target": f"room {i}"}}]
        examples.append({
            "domain": domain,
            "query": query,
            "tools": tools,
            "answers": answers,
            "reasoning": f"'{query}' maps to {act}_{slot}",
            "is_language_sample": False,
        })
    return examples


def build_synthetic_language_examples(count: int, seed: int = STUDY_SEED) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    phrases = [
        "In artificial intelligence and numerical computation, memory hierarchy governs inference speed.",
        "Cross-layer attention reduces working key-value cache memory across transformer blocks.",
        "Structured matrices provide higher parameter capacity without increasing arithmetic operations.",
        "Decoupling lexical memory tables enables depth-specific projections on shared vocabulary stores.",
    ]
    examples = []
    for i in range(count):
        text = phrases[int(rng.integers(0, len(phrases)))] + f" Sample index {i}."
        examples.append({
            "domain": "fineweb_edu",
            "text": text,
            "is_language_sample": True,
        })
    return examples


def create_study_manifest(
    dataset_paths: Optional[Dict[str, str]] = None,
    evaluation_paths: Optional[Sequence[str]] = None,
    output_path: str = "study_runs/manifest.json",
    sample_size: int = 1000,
    seed: int = STUDY_SEED,
    allow_synthetic: bool = False,
) -> Dict[str, Any]:
    """Create balanced immutable dataset manifest.
    
    Ratios:
    - 30% DroidCall
    - 30% Mobile Actions
    - 15% SNIPS
    - 15% DSTC8
    - 10% FineWeb-Edu
    """
    rng = np.random.default_rng(seed)
    counts = {
        "droidcall": int(sample_size * 0.30),
        "mobile_actions": int(sample_size * 0.30),
        "snips": int(sample_size * 0.15),
        "dstc8": int(sample_size * 0.15),
        "fineweb_edu": sample_size - (2 * int(sample_size * 0.30) + 2 * int(sample_size * 0.15)),
    }

    manifest_examples: List[Dict[str, Any]] = []
    eval_hash_set: Set[str] = set()
    for eval_path in evaluation_paths or ():
        if not os.path.exists(eval_path):
            raise FileNotFoundError(f"evaluation dataset does not exist: {eval_path}")
        with open(eval_path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                eval_hash_set.add(hash_prompt(row.get("query", row.get("text", ""))))
    seen_hashes: Set[str] = set()

    for domain, count in counts.items():
        domain_file = (dataset_paths or {}).get(domain)
        loaded = []
        if domain_file and os.path.exists(domain_file):
            with open(domain_file, "r") as f:
                for line in f:
                    if line.strip():
                        loaded.append(json.loads(line))
        if len(loaded) < count and not allow_synthetic:
            raise ValueError(
                f"{domain} contains {len(loaded)} rows, but {count} are required; "
                "provide the official training split or explicitly enable synthetic smoke data"
            )
        if len(loaded) < count:
            if domain == "fineweb_edu":
                synth = build_synthetic_language_examples(count - len(loaded), seed=seed + 1)
            else:
                synth = build_synthetic_task_examples(count - len(loaded), domain=domain, seed=seed + 2)
            loaded.extend(synth)

        # Shuffle the complete pool, then filter evaluation duplicates and
        # duplicates selected from another domain before taking ``count``.
        # Sampling exactly ``count`` rows first can leave an apparent shortage
        # even when the training split contains enough valid rows.
        perm = rng.permutation(len(loaded))
        added = 0
        for idx in perm:
            ex = loaded[idx]
            p_hash = hash_prompt(ex.get("query", ex.get("text", "")))
            if p_hash in eval_hash_set or p_hash in seen_hashes:
                continue
            manifest_examples.append(ex)
            seen_hashes.add(p_hash)
            added += 1
            if added == count:
                break
        if added < count and not allow_synthetic:
            raise ValueError(
                f"{domain} contains only {added} unique non-evaluation rows, "
                f"but {count} are required"
            )

    rng.shuffle(manifest_examples)
    if not allow_synthetic and len(manifest_examples) != sample_size:
        raise ValueError(
            f"deduplication/evaluation isolation left {len(manifest_examples)} of "
            f"{sample_size} requested training examples"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    manifest_data = {
        "seed": seed,
        "sample_size": len(manifest_examples),
        "ratios": counts,
        "synthetic_smoke_data": bool(allow_synthetic),
        "examples": manifest_examples,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest_examples, sort_keys=True).encode("utf-8")).hexdigest(),
    }

    with open(output_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data
