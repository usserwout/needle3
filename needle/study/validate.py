"""Small, paired held-out teacher-forced check for recovery runs.

This is a useful training diagnostic, not the six-suite generation benchmark.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..model.architecture import SimpleAttentionNetwork
from ..model.checkpoints import read_checkpoint
from ..model.tokenizer import get_tokenizer
from .config import runtime_transformer_config
from .data import encode_study_example, hash_prompt


SUITES = ("droidcall", "mobile_actions", "snips", "dstc8")


def validate_run(
    run_id: str,
    checkpoint_dir: str = "study_runs",
    data_dir: str = "data/normalized",
    max_examples: int = 50,
    batch_size: int = 4,
) -> dict:
    """Measure response-only NLL and token accuracy on fixed held-out rows."""
    if max_examples < 1 or batch_size < 1:
        raise ValueError("max_examples and batch_size must be positive")
    run_dir = Path(checkpoint_dir) / run_id
    checkpoint = read_checkpoint(run_dir / "checkpoint.safetensors")
    config = runtime_transformer_config(checkpoint["config"])
    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer(config.vocab_size)
    params = jax.tree_util.tree_map(jnp.asarray, checkpoint["params"])
    manifest_path = Path(checkpoint_dir) / "manifest.json"
    train_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_hashes = {hash_prompt(row.get("query", row.get("text", "")))
                    for row in train_manifest["examples"]}

    @jax.jit
    def score(runtime_params, ids, masks):
        logits = model.apply({"params": runtime_params}, ids, quant=False)[:, :-1]
        targets = ids[:, 1:]
        mask = masks[:, 1:]
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        correct = (jnp.argmax(logits, axis=-1) == targets).astype(jnp.float32)
        return (jnp.sum(losses * mask, axis=-1),
                jnp.sum(correct * mask, axis=-1),
                jnp.sum(mask, axis=-1))

    suites = {}
    for suite in SUITES:
        path = Path(data_dir) / f"{suite}_eval.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing official held-out split: {path}")
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        rows.sort(key=lambda row: hashlib.sha256(
            str(row.get("query", "")).encode("utf-8")
        ).hexdigest())
        selected = rows[:max_examples]
        print(f"{run_id} | {suite}: validating {len(selected)} held-out examples", flush=True)
        overlaps = [row for row in selected if hash_prompt(row.get("query", "")) in train_hashes]
        if overlaps:
            raise ValueError(f"{suite} validation overlaps recovery training by {len(overlaps)} prompts")
        encoded = [encode_study_example(tokenizer, row,
                                        max_len=min(512, config.max_seq_len))
                   for row in selected]
        examples = []
        for start in range(0, len(selected), batch_size):
            chunk = encoded[start:start + batch_size]
            ids = jnp.asarray([item[0] for item in chunk], dtype=jnp.int32)
            masks = jnp.asarray([item[1] for item in chunk], dtype=jnp.float32)
            loss_sums, correct_sums, counts = map(np.asarray, score(params, ids, masks))
            for row, loss_sum, correct_sum, count in zip(
                selected[start:start + batch_size], loss_sums, correct_sums, counts
            ):
                count = float(count)
                if count == 0:
                    continue
                loss = float(loss_sum / count)
                accuracy = float(correct_sum / count)
                if not np.isfinite(loss):
                    raise FloatingPointError(f"{run_id} has nonfinite validation loss on {suite}")
                examples.append({
                    "prompt_sha256": hashlib.sha256(str(row.get("query", "")).encode()).hexdigest(),
                    "nll": loss,
                    "token_accuracy": accuracy,
                    "target_tokens": int(count),
                })
            completed = min(start + batch_size, len(selected))
            if completed % 10 < batch_size or completed == len(selected):
                print(f"{run_id} | {suite}: {completed}/{len(selected)}", flush=True)
        if not examples:
            raise ValueError(f"no supervised target tokens in {path}")
        suites[suite] = {
            "nll": float(np.mean([item["nll"] for item in examples])),
            "token_accuracy": float(np.mean([item["token_accuracy"] for item in examples])),
            "examples": examples,
            "skipped_truncated": len(selected) - len(examples),
        }
        print(f"{run_id} | {suite}: NLL {suites[suite]['nll']:.4f}, "
              f"token accuracy {suites[suite]['token_accuracy']:.2%}", flush=True)

    result = {
        "run_id": run_id,
        "metric_kind": "held_out_teacher_forced_pilot",
        "max_examples_per_suite": max_examples,
        "macro_nll": float(np.mean([suite["nll"] for suite in suites.values()])),
        "macro_token_accuracy": float(np.mean([suite["token_accuracy"] for suite in suites.values()])),
        "suites": suites,
    }
    with (run_dir / "validation_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
