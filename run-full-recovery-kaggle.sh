#!/usr/bin/env bash
set -Eeuo pipefail

# Fresh, paired recovery training. The 100-step pilot is left untouched.
# Run after setup-kaggle.sh and the pilot's run-study-kaggle.sh.

cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME CUDA_VISIBLE_DEVICES

readonly python=.venv-kaggle/bin/python
readonly pilot=study_runs
readonly output=study_full_runs
readonly target_tokens=8192

[[ -x "$python" ]] || { echo "Run setup-kaggle.sh first" >&2; exit 1; }
[[ $(nvidia-smi -L | wc -l) -ge 2 ]] || {
    echo "Two NVIDIA GPUs are required for this two-queue runner" >&2; exit 1;
}
for run in C8 I1-8 C12 I2-12 I3-12; do
    [[ -s "$pilot/$run/checkpoint.safetensors" ]] || {
        echo "Missing 100-step pilot: $pilot/$run/checkpoint.safetensors" >&2; exit 1;
    }
done
for name in manifest.json train_data.npz engram_calibration_indices.npz teachers; do
    [[ -e "$pilot/$name" ]] || { echo "Missing $pilot/$name" >&2; exit 1; }
done

mkdir -p "$output"
for name in manifest.json train_data.npz engram_calibration_indices.npz teachers; do
    if [[ ! -e "$output/$name" ]]; then
        ln -s "../$pilot/$name" "$output/$name"
    fi
done

# The same-depth budgets are fixed once and stored. Pilot throughput is the
# measured single-model target-token rate available before a fresh full run.
# This avoids changing the optimizer horizon when training is resumed.
budget_path="$output/budget.json"
if [[ ! -s "$budget_path" ]]; then
    "$python" - "$pilot" "$budget_path" <<'PY'
import json
import math
import sys
from pathlib import Path
from needle.model.checkpoints import read_checkpoint

pilot, destination = Path(sys.argv[1]), Path(sys.argv[2])
rates = {}
for run in ("C8", "I1-8", "C12", "I2-12", "I3-12"):
    checkpoint = read_checkpoint(pilot / run / "checkpoint.safetensors")
    rate = float(checkpoint["run"].get("tokens_per_second", 0.0))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"{run} has no usable pilot throughput")
    rates[run] = rate
steps = {
    "8": math.floor(min(rates["C8"], rates["I1-8"]) * 2.5 * 3600 / 8192),
    "12": math.floor(min(rates["C12"], rates["I2-12"], rates["I3-12"]) * 3 * 3600 / 8192),
}
if min(steps.values()) < 10:
    raise ValueError(f"pilot throughput gives implausible budgets: {steps}")
destination.write_text(json.dumps({
    "source": "100-step pilot target-token throughput",
    "pilot_rates": rates,
    "target_tokens_per_update": 8192,
    "steps": steps,
    "target_token_budgets": {depth: count * 8192 for depth, count in steps.items()},
    "mini_steps": {depth: max(1, count // 10) for depth, count in steps.items()},
    "seed": 20260921,
}, indent=2))
print(destination.read_text(), flush=True)
PY
fi

read -r steps8 steps12 mini8 mini12 < <("$python" - "$budget_path" <<'PY'
import json, sys
budget = json.load(open(sys.argv[1]))
print(budget["steps"]["8"], budget["steps"]["12"],
      budget["mini_steps"]["8"], budget["mini_steps"]["12"])
PY
)

run_queue() {
    local gpu="$1" phase="$2"
    shift 2
    local run depth steps stop
    for run in "$@"; do
        if [[ "$run" == *8 ]]; then depth=8; steps=$steps8; stop=$mini8
        else depth=12; steps=$steps12; stop=$mini12; fi
        mkdir -p "$output/$run"
        if [[ ! -s "$output/$run/checkpoint.safetensors" ]]; then
            echo "GPU $gpu: transplanting fresh $run"
            CUDA_VISIBLE_DEVICES="$gpu" "$python" -u -m needle.study transplant \
                --run "$run" --out-dir "$output"
        fi
        echo "GPU $gpu: $phase $run ($steps updates, $((steps * target_tokens)) target tokens)"
        if [[ "$phase" == mini ]]; then
            CUDA_VISIBLE_DEVICES="$gpu" "$python" -u -m needle.study train \
                --run "$run" --out-dir "$output" --steps "$steps" \
                --target-tokens-per-update "$target_tokens" --stop-after "$stop" \
                --resume 2>&1 | tee -a "$output/$run/full_train.log"
        else
            CUDA_VISIBLE_DEVICES="$gpu" "$python" -u -m needle.study train \
                --run "$run" --out-dir "$output" --steps "$steps" \
                --target-tokens-per-update "$target_tokens" --resume \
                2>&1 | tee -a "$output/$run/full_train.log"
        fi
    done
}

run_phase() {
    local phase="$1" status=0
    run_queue 0 "$phase" C12 I2-12 & local gpu0=$!
    run_queue 1 "$phase" C8 I1-8 I3-12 & local gpu1=$!
    wait "$gpu0" || status=1
    wait "$gpu1" || status=1
    (( status == 0 )) || { echo "$phase training failed; inspect full_train.log" >&2; exit 1; }
}

run_phase mini
echo "Ten-percent recovery boundary reached on both GPUs."
run_phase finish
echo "Full target-token recovery training completed on both GPUs."
