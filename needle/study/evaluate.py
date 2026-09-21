from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats

from ..model.architecture import SimpleAttentionNetwork, TransformerConfig
from ..model.checkpoints import read_checkpoint
from ..model.tokenizer import get_tokenizer, BOS_ID, EOS_ID, PAD_ID
from .config import RUNS, RunConfig, STUDY_SEED


def bootstrap_ci(
    deltas: np.ndarray,
    num_bootstraps: int = 10000,
    alpha: float = 0.05,
    seed: int = STUDY_SEED,
) -> Tuple[float, float, float]:
    """Compute mean and stratified bootstrap confidence interval."""
    if len(deltas) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(deltas)
    indices = rng.integers(0, n, size=(num_bootstraps, n))
    bootstrap_means = np.mean(deltas[indices], axis=1)
    lower = float(np.percentile(bootstrap_means, 100 * (alpha / 2)))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    mean = float(np.mean(deltas))
    return mean, lower, upper


def mcnemar_test(control_correct: np.ndarray, variant_correct: np.ndarray) -> float:
    """Exact McNemar test for paired binary classification outcomes."""
    b = int(np.sum((control_correct == 1) & (variant_correct == 0)))
    c = int(np.sum((control_correct == 0) & (variant_correct == 1)))
    n = b + c
    if n == 0:
        return 1.0
    # Two-sided binomial test with p=0.5
    pval = scipy.stats.binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
    return float(pval)


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Adjust p-values using Holm-Bonferroni step-down procedure."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p = p_values[idx]
        adj = min(1.0, (m - rank) * p)
        cum_max = max(cum_max, adj)
        adjusted[idx] = cum_max
    return adjusted


def create_diagnostic_dataset(seed: int = STUDY_SEED) -> List[Dict[str, Any]]:
    """Create 1,000 diagnostic examples (500 absent fields, 500 corrupted/gibberish)."""
    rng = np.random.default_rng(seed)
    diagnostic = []

    # 500 absent fields
    for i in range(500):
        diagnostic.append({
            "type": "absent_field",
            "query": f"Send the monthly summary report for project {i}",
            "tools": [{
                "name": "send_report",
                "description": "Send a report to an email recipient",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project": {"type": "string"},
                        "recipient_email": {"type": "string"},
                    },
                    "required": ["project"],
                },
            }],
            # Expected answer has no recipient_email
            "expected": {"project": f"project {i}"},
        })

    # 500 corrupted/gibberish inputs
    for i in range(500):
        gibberish = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz   "), size=25))
        diagnostic.append({
            "type": "corrupted_input",
            "query": f"xqz {gibberish} 9872",
            "tools": [{
                "name": "manage_device",
                "description": "Control device switches",
                "parameters": {"type": "object", "properties": {}},
            }],
            "expected": None,  # Abstention expected
        })

    return diagnostic


def evaluate_run(
    run_id: str,
    checkpoint_dir: str = "study_runs",
    num_samples_per_suite: int = 50,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Evaluate quality suites on target checkpoint.

    Synthetic outcomes are permitted only for explicit pipeline smoke tests. Real
    study runs must use the official benchmark adapters, which are not yet part
    of this repository.
    """
    if not smoke:
        raise NotImplementedError(
            "official Mobile Actions, DroidCall, BFCL, DSTC8, and SNIPS adapters "
            "are not implemented; refusing to fabricate screening scores"
        )
    run_dir = os.path.join(checkpoint_dir, run_id)
    ckpt_path = os.path.join(run_dir, "checkpoint.safetensors")
    ckpt_data = read_checkpoint(ckpt_path)
    params = ckpt_data["params"]
    config = TransformerConfig(**ckpt_data["config"])
    tokenizer = get_tokenizer(config.vocab_size)
    model = SimpleAttentionNetwork(config)

    suites = [
        "mobile_actions",
        "droidcall",
        "bfcl_v4",
        "dstc8",
        "snips_slots",
        "snips_intent",
    ]

    suite_scores = {}
    rng = np.random.default_rng(STUDY_SEED)

    # Explicitly synthetic simulation for plumbing tests only.
    for suite in suites:
        base_rate = 0.78 + (sum(suite.encode("utf-8")) % 10) * 0.01
        correctness = (rng.uniform(size=num_samples_per_suite) < base_rate).astype(np.float32)
        suite_scores[suite] = {
            "score": float(np.mean(correctness) * 100),
            "parse_validity": 100.0,
            "outcomes": correctness.tolist(),
        }

    # Evaluate diagnostic set
    diagnostic_set = create_diagnostic_dataset(STUDY_SEED)
    absent_correct = [1.0 if rng.random() > 0.1 else 0.0 for _ in range(500)]
    corrupt_abstain = [1.0 if rng.random() > 0.05 else 0.0 for _ in range(500)]

    macro_score = float(np.mean([suite_scores[s]["score"] for s in suites]))
    results = {
        "run_id": run_id,
        "synthetic_smoke_results": True,
        "macro_score": macro_score,
        "suite_scores": suite_scores,
        "diagnostic_metrics": {
            "absent_field_accuracy": float(np.mean(absent_correct) * 100),
            "corrupted_abstention_accuracy": float(np.mean(corrupt_abstain) * 100),
        },
    }

    eval_out_path = os.path.join(run_dir, "eval_results.json")
    with open(eval_out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
