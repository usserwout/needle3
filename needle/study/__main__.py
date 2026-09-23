from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from ..model.architecture import SimpleAttentionNetwork
from ..model.checkpoints import read_checkpoint
from ..model.tokenizer import get_tokenizer
from .config import RUNS, StudyConfig, DEFAULT_BASE_CHECKPOINT, runtime_transformer_config
from .data import create_study_manifest, encode_study_example, collect_engram_calibration_indices
from .distill import compute_teacher_cache, save_teacher_cache
from .evaluate import evaluate_run
from .profile import profile_run
from .report import generate_study_report
from .train import run_training_loop
from .transplant import transplant_run
from .validate import validate_run


def load_study_config(path: str) -> StudyConfig:
    if not path or not os.path.exists(path):
        return StudyConfig()
    with open(path) as handle:
        if path.endswith(".json"):
            raw = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("YAML study configs require the train extra (pyyaml)") from exc
            raw = yaml.safe_load(handle) or {}
    return StudyConfig.from_dict(raw)


def prepare_cmd(args):
    print("Preparing study manifest and teacher caches...")
    study_cfg = load_study_config(args.config)
    out_dir = args.out_dir or study_cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    dataset_paths = dict(study_cfg.dataset_paths)
    for item in args.dataset:
        if "=" not in item:
            raise ValueError("--dataset must use domain=/path/to/train.jsonl")
        domain, path = item.split("=", 1)
        dataset_paths[domain] = path
    manifest = create_study_manifest(
        dataset_paths=dataset_paths,
        evaluation_paths=args.eval_data or study_cfg.evaluation_paths,
        output_path=manifest_path,
        sample_size=args.sample_size or study_cfg.sample_size,
        allow_synthetic=args.smoke,
    )
    print(f"  Manifest written with {len(manifest['examples'])} examples to {manifest_path}")

    # If base checkpoint is available, generate teacher caches for R8 and R12
    base_ckpt = args.checkpoint or study_cfg.base_checkpoint
    if os.path.exists(base_ckpt):
        print(f"  Precomputing frozen teacher caches from {base_ckpt}...")
        teacher_dir = os.path.join(out_dir, "teachers")
        os.makedirs(teacher_dir, exist_ok=True)

        for teacher_run in ("R8", "R12"):
            t_meta = transplant_run(teacher_run, base_checkpoint_path=base_ckpt, output_dir=out_dir)
            t_ckpt = read_checkpoint(os.path.join(out_dir, teacher_run, "checkpoint.safetensors"))
            t_cfg = runtime_transformer_config(t_ckpt["config"])
            t_model = SimpleAttentionNetwork(t_cfg)
            tokenizer = get_tokenizer(t_cfg.vocab_size)
            encoded = [encode_study_example(tokenizer, row, max_len=512)
                       for row in manifest["examples"]]
            input_ids = np.asarray([row[0] for row in encoded], dtype=np.int32)
            target_mask = np.asarray([row[1] for row in encoded], dtype=np.float32)
            data_path = os.path.join(out_dir, "train_data.npz")
            np.savez_compressed(data_path, input_ids=input_ids, target_mask=target_mask)
            engram_path = os.path.join(out_dir, "engram_calibration_indices.npz")
            if not args.smoke and not os.path.exists(engram_path):
                rows = collect_engram_calibration_indices(input_ids, t_cfg, count=50000)
                np.savez_compressed(engram_path, indices=rows)
            cache = compute_teacher_cache(t_model, t_ckpt["params"], input_ids, target_mask)
            cache_p = os.path.join(teacher_dir, f"{teacher_run}_cache.npz")
            save_teacher_cache(cache, cache_p)
            print(f"  Saved {teacher_run} teacher cache to {cache_p}")
    print("Preparation complete.")


def transplant_cmd(args):
    run_id = args.run
    base_ckpt = getattr(args, "checkpoint", DEFAULT_BASE_CHECKPOINT)
    out_dir = getattr(args, "out_dir", "study_runs")
    print(f"Transplanting run {run_id} from {base_ckpt}...")
    ngram_samples = None
    if run_id == "I2-12":
        sample_path = os.path.join(out_dir, "engram_calibration_indices.npz")
        if not os.path.exists(sample_path):
            raise FileNotFoundError(
                f"missing {sample_path}; prepare the official study data before I2-12"
            )
        indices = np.load(sample_path)["indices"]
        ngram_samples = {site: indices for site in range(3)}
    meta = transplant_run(run_id, base_checkpoint_path=base_ckpt, output_dir=out_dir,
                          ngram_samples=ngram_samples)
    print(f"  Successfully transplanted {run_id} ({meta['depth']} layers, {meta['active_parameters']:,d} active params).")


def train_cmd(args):
    run_id = args.run
    out_dir = getattr(args, "out_dir", "study_runs")
    smoke = getattr(args, "smoke", False)
    steps = 2 if smoke else getattr(args, "steps", 100)
    print(f"Starting recovery training for {run_id} ({steps} steps)...")
    res = run_training_loop(run_id, checkpoint_dir=out_dir, total_steps=steps,
                            resume=args.resume, smoke=smoke, progress=print,
                            target_tokens_per_update=args.target_tokens_per_update,
                            stop_after_steps=args.stop_after)
    print(f"  Training finished: {res}")


def evaluate_cmd(args):
    run_id = args.run
    out_dir = getattr(args, "out_dir", "study_runs")
    print(f"Evaluating quality benchmarks for {run_id}...")
    res = evaluate_run(run_id, checkpoint_dir=out_dir, smoke=args.smoke)
    print(f"  Evaluation complete. Macro score: {res['macro_score']:.2f}%")


def profile_cmd(args):
    run_id = args.run
    out_dir = getattr(args, "out_dir", "study_runs")
    print(f"Profiling efficiency for {run_id}...")
    res = profile_run(run_id, checkpoint_dir=out_dir)
    print(f"  Profiling complete. Analytical MACs: {res['analytical_projection_macs']:,d}")


def validate_cmd(args):
    result = validate_run(args.run, checkpoint_dir=args.out_dir,
                          data_dir=args.data_dir, max_examples=args.max_examples)
    print(f"{args.run}: held-out NLL={result['macro_nll']:.4f}, "
          f"token accuracy={result['macro_token_accuracy']:.2%} "
          "(teacher-forced pilot; not a six-suite benchmark)")


def report_cmd(args):
    out_dir = getattr(args, "out_dir", "study_runs")
    print("Synthesizing study report...")
    res = generate_study_report(checkpoint_dir=out_dir)


def main():
    parser = argparse.ArgumentParser(prog="needle.study", description="Needle 4 Architecture Screening Framework")
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare
    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--config", default="study.yaml")
    p_prep.add_argument("--checkpoint")
    p_prep.add_argument("--out-dir")
    p_prep.add_argument("--sample-size", type=int)
    p_prep.add_argument("--dataset", action="append", default=[],
                        help="Official normalized training split as domain=/path/file.jsonl")
    p_prep.add_argument("--eval-data", action="append", default=[],
                        help="Evaluation JSONL used only for train/eval prompt deduplication")
    p_prep.add_argument("--smoke", action="store_true",
                        help="Allow deterministic synthetic data for pipeline tests only")

    # transplant
    p_tr = sub.add_parser("transplant")
    p_tr.add_argument("--run", required=True, choices=list(RUNS.keys()))
    p_tr.add_argument("--checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    p_tr.add_argument("--out-dir", default="study_runs")

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--run", required=True, choices=list(RUNS.keys()))
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument("--smoke", action="store_true")
    p_train.add_argument("--steps", type=int, default=100)
    p_train.add_argument("--target-tokens-per-update", type=int,
                         help="Accumulate exactly this many supervised tokens before each update")
    p_train.add_argument("--stop-after", type=int,
                         help="Stop at this update while preserving the original optimizer schedule")
    p_train.add_argument("--out-dir", default="study_runs")

    # evaluate
    p_ev = sub.add_parser("evaluate")
    p_ev.add_argument("--run", required=True, choices=list(RUNS.keys()))
    p_ev.add_argument("--out-dir", default="study_runs")
    p_ev.add_argument("--smoke", action="store_true",
                      help="Use synthetic outcomes to verify plumbing; never use for decisions")

    # profile
    p_prof = sub.add_parser("profile")
    p_prof.add_argument("--run", required=True, choices=list(RUNS.keys()))
    p_prof.add_argument("--out-dir", default="study_runs")

    p_val = sub.add_parser("validate")
    p_val.add_argument("--run", required=True, choices=list(RUNS.keys()))
    p_val.add_argument("--out-dir", default="study_runs")
    p_val.add_argument("--data-dir", default="data/normalized")
    p_val.add_argument("--max-examples", type=int, default=50)

    # report
    p_rep = sub.add_parser("report")
    p_rep.add_argument("--out-dir", default="study_runs")

    args = parser.parse_args()
    cmds = {
        "prepare": prepare_cmd,
        "transplant": transplant_cmd,
        "train": train_cmd,
        "evaluate": evaluate_cmd,
        "profile": profile_cmd,
        "validate": validate_cmd,
        "report": report_cmd,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
