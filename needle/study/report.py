from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import RUNS, RunConfig
from .evaluate import bootstrap_ci, holm_bonferroni_correction, mcnemar_test


def evaluate_decision_rules(
    idea_id: str,
    variant_run: str,
    control_run: str,
    variant_eval: Dict[str, Any],
    control_eval: Dict[str, Any],
    variant_profile: Dict[str, Any],
    control_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate formal screening decision rules against same-depth control."""
    var_macro = variant_eval.get("macro_score", 0.0)
    ctrl_macro = control_eval.get("macro_score", 0.0)
    macro_delta = var_macro - ctrl_macro

    # Paired suite delta analysis
    suites = list(variant_eval.get("suite_scores", {}).keys())
    suite_deltas = {}
    max_suite_drop = 0.0

    all_var_outcomes = []
    all_ctrl_outcomes = []

    for s in suites:
        v_s = variant_eval["suite_scores"][s]["score"]
        c_s = control_eval["suite_scores"][s]["score"]
        d = v_s - c_s
        suite_deltas[s] = d
        if d < 0:
            max_suite_drop = max(max_suite_drop, abs(d))
        
        v_out = variant_eval["suite_scores"][s].get("outcomes", [])
        c_out = control_eval["suite_scores"][s].get("outcomes", [])
        all_var_outcomes.extend(v_out)
        all_ctrl_outcomes.extend(c_out)

    v_arr = np.array(all_var_outcomes)
    c_arr = np.array(all_ctrl_outcomes)
    paired_deltas = (v_arr - c_arr) * 100.0 if len(v_arr) == len(c_arr) and len(v_arr) > 0 else np.array([macro_delta])

    mean_d, ci_lower, ci_upper = bootstrap_ci(paired_deltas)
    mcnemar_p = mcnemar_test(c_arr, v_arr) if len(v_arr) == len(c_arr) and len(v_arr) > 0 else 1.0

    # Efficiency target verification
    v_act = variant_profile.get("active_parameters", 0)
    c_act = control_profile.get("active_parameters", 0)
    param_savings = c_act - v_act

    v_lat = variant_profile.get("latency_measurements", {}).get("prefill_512_ms", {}).get("median")
    c_lat = control_profile.get("latency_measurements", {}).get("prefill_512_ms", {}).get("median")
    has_latency = v_lat is not None and c_lat is not None and c_lat > 0
    lat_overhead_pct = (((v_lat - c_lat) / c_lat) * 100.0) if has_latency else None

    efficiency_passed = False
    if idea_id == "Idea 1":
        # Save at least 3.12M active parameters, latency within 2%
        efficiency_passed = (param_savings >= 3_120_000) and has_latency and (lat_overhead_pct <= 2.0)
    elif idea_id == "Idea 2":
        # Save exactly one bank (14,155,776 params), latency within 2%
        efficiency_passed = (param_savings == 14_155_776) and has_latency and (lat_overhead_pct <= 2.0)
    elif idea_id == "Idea 3":
        # Cache reduction demonstrated, latency within 3%
        v_kv = variant_profile.get("analytical_kv_bytes", {}).get(1024, 0)
        c_kv = control_profile.get("analytical_kv_bytes", {}).get(1024, 0)
        cache_reduction = ((c_kv - v_kv) / max(c_kv, 1)) * 100.0
        efficiency_passed = (cache_reduction >= 30.0) and has_latency and (lat_overhead_pct <= 3.0)

    quality_passed = (ci_lower >= -1.0) and (max_suite_drop <= 2.0)

    if quality_passed and efficiency_passed:
        decision = "promote to 20L confirmation"
    elif not quality_passed and ci_upper < -1.0:
        decision = "reject"
    else:
        decision = "inconclusive"

    return {
        "idea": idea_id,
        "variant": variant_run,
        "control": control_run,
        "macro_delta": macro_delta,
        "ci_95": [ci_lower, ci_upper],
        "mcnemar_p_value": mcnemar_p,
        "param_savings": param_savings,
        "latency_overhead_pct": lat_overhead_pct,
        "quality_passed": quality_passed,
        "efficiency_passed": efficiency_passed,
        "decision": decision,
    }


def generate_study_report(checkpoint_dir: str = "study_runs", output_dir: Optional[str] = None) -> str:
    """Aggregate screening results and generate markdown comparison table and summary."""
    if output_dir is not None:
        checkpoint_dir = output_dir

    comparisons = [
        ("Idea 1", "I1-8", "C8"),
        ("Idea 2", "I2-12", "C12"),
        ("Idea 3", "I3-12", "C12"),
    ]

    decisions = []
    p_values = []

    for idea_id, var_run, ctrl_run in comparisons:
        var_eval_p = os.path.join(checkpoint_dir, var_run, "eval_results.json")
        ctrl_eval_p = os.path.join(checkpoint_dir, ctrl_run, "eval_results.json")
        var_prof_p = os.path.join(checkpoint_dir, var_run, "profile_results.json")
        ctrl_prof_p = os.path.join(checkpoint_dir, ctrl_run, "profile_results.json")

        required = (var_eval_p, ctrl_eval_p, var_prof_p, ctrl_prof_p)
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"cannot report {idea_id}; missing artifacts: {missing}")
        var_eval = json.load(open(var_eval_p))
        ctrl_eval = json.load(open(ctrl_eval_p))
        var_prof = json.load(open(var_prof_p))
        ctrl_prof = json.load(open(ctrl_prof_p))
        if var_eval.get("synthetic_smoke_results") or ctrl_eval.get("synthetic_smoke_results"):
            raise ValueError("synthetic smoke evaluations cannot be used for screening decisions")

        dec = evaluate_decision_rules(idea_id, var_run, ctrl_run, var_eval, ctrl_eval, var_prof, ctrl_prof)
        decisions.append(dec)
        p_values.append(dec["mcnemar_p_value"])

    adj_p = holm_bonferroni_correction(p_values)
    for i, dec in enumerate(decisions):
        dec["holm_adjusted_p"] = adj_p[i]

    report_data = {
        "screening_decisions": decisions,
        "summary": "Screening study comparison completed against same-depth recovery controls.",
    }

    report_path = os.path.join(checkpoint_dir, "screening_report.json")
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # Print terminal comparison table
    print("\n" + "=" * 80)
    print("NEEDLE 4 ARCHITECTURE SCREENING STUDY: DECISION SUMMARY")
    print("=" * 80)
    print(f"{'Idea':<10} | {'Variant':<8} | {'Delta (%)':<10} | {'95% CI':<16} | {'Param Save':<12} | {'Decision':<25}")
    print("-" * 80)
    for dec in decisions:
        ci_str = f"[{dec['ci_95'][0]:+.2f}, {dec['ci_95'][1]:+.2f}]"
        print(f"{dec['idea']:<10} | {dec['variant']:<8} | {dec['macro_delta']:+8.2f}% | {ci_str:<16} | {dec['param_savings']:<12,d} | {dec['decision']:<25}")
    print("=" * 80 + "\n")

    return report_path
