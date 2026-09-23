#!/usr/bin/env bash
set -Eeuo pipefail

# Run the five independent recovery studies on Kaggle's two T4 GPUs.
# Run once after setup-kaggle.sh from the repository root. Safe to rerun:
# train --resume returns immediately for runs that completed all 100 updates.

if [[ ! -x .venv-kaggle/bin/python ]]; then
    echo "Run bash setup-kaggle.sh first." >&2
    exit 1
fi
for required in checkpoints/needle3.safetensors study_runs/train_data.npz \
                study_runs/teachers/R8_cache.npz study_runs/teachers/R12_cache.npz \
                study_runs/engram_calibration_indices.npz \
                data/normalized/droidcall_eval.jsonl; do
    if [[ ! -s "$required" ]]; then
        echo "Missing $required; run bash setup-kaggle.sh first." >&2
        exit 1
    fi
done

gpu_count=$(nvidia-smi -L | wc -l | tr -d ' ')
if (( gpu_count < 2 )); then
    echo "This runner needs two visible NVIDIA GPUs; Kaggle currently exposes $gpu_count." >&2
    exit 1
fi

# Kaggle's CUDA toolkit libraries conflict with the pinned jax[cuda12]
# wheels. JAX loads its own CUDA runtime and uses only this driver directory.
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

run_queue() {
    local gpu="$1"
    shift
    local run
    for run in "$@"; do
        mkdir -p "study_runs/$run"
        if [[ ! -s "study_runs/$run/checkpoint.safetensors" ]]; then
            echo "GPU $gpu: transplanting $run"
            CUDA_VISIBLE_DEVICES="$gpu" .venv-kaggle/bin/python -m needle.study transplant --run "$run"
        fi
        echo "GPU $gpu: training or resuming $run"
        CUDA_VISIBLE_DEVICES="$gpu" .venv-kaggle/bin/python -u -m needle.study train \
            --run "$run" --steps 100 --resume 2>&1 | tee "study_runs/$run/train.log"
    done
}

# A 12-layer run takes longer than an 8-layer run. These queues balance the
# measured work reasonably well and never place two models on the same GPU.
run_queue 0 C12 I2-12 &
gpu0_pid=$!
run_queue 1 C8 I1-8 I3-12 &
gpu1_pid=$!

status=0
wait "$gpu0_pid" || status=1
wait "$gpu1_pid" || status=1
if (( status != 0 )); then
    echo "At least one GPU queue failed; inspect study_runs/*/train.log." >&2
    exit 1
fi
echo "All five recovery runs completed."

validate_queue() {
    local gpu="$1"
    shift
    local run
    for run in "$@"; do
        echo "GPU $gpu: validating $run"
        CUDA_VISIBLE_DEVICES="$gpu" .venv-kaggle/bin/python -u -m needle.study validate \
            --run "$run" --max-examples 50 2>&1 | tee "study_runs/$run/validate.log"
    done
}

validate_queue 0 R12 C12 I2-12 &
gpu0_pid=$!
validate_queue 1 R8 C8 I1-8 I3-12 &
gpu1_pid=$!
status=0
wait "$gpu0_pid" || status=1
wait "$gpu1_pid" || status=1
if (( status != 0 )); then
    echo "At least one validation queue failed; inspect study_runs/*/validate.log." >&2
    exit 1
fi
echo "Held-out validation completed on both GPUs."
