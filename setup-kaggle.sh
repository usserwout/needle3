#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible single-GPU Kaggle setup for the Needle architecture study.
# Run from the repository root:  bash setup-kaggle.sh

readonly NUMPY_VERSION="2.0.2"
readonly SCIPY_VERSION="1.14.1"
readonly JAX_VERSION="0.7.2"
readonly FLAX_VERSION="0.10.2"
readonly OPTAX_VERSION="0.2.4"
readonly DATASETS_VERSION="3.2.0"
readonly HF_HUB_VERSION="0.27.1"
readonly VENV_DIR=".venv-kaggle"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

if [[ ! -f pyproject.toml || ! -d needle ]]; then
    die "run this script from the needle3 repository root"
fi

command -v python >/dev/null 2>&1 || die "Kaggle Python was not found"
command -v nvidia-smi >/dev/null 2>&1 || die \
    "no NVIDIA GPU is attached; in Kaggle open Notebook options, select a GPU accelerator, and restart the session"
nvidia-smi -L >/dev/null 2>&1 || die \
    "the NVIDIA driver cannot see a GPU; enable a Kaggle GPU accelerator or wait for GPU quota"

# Keep the screening study reproducible when Kaggle offers a dual-T4 runtime.
export CUDA_VISIBLE_DEVICES=0

# Kaggle prepends its CUDA toolkit directories to LD_LIBRARY_PATH. That makes
# JAX load a mixture of Kaggle and pip CUDA libraries, which can crash inside
# cuInit/cuSPARSE. Keep only the NVIDIA driver directory; jax[cuda12] then loads
# its matched CUDA runtime libraries from the isolated environment.
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64

# Kaggle injects a global sitecustomize via these variables. In an isolated
# environment it imports optional global packages such as wrapt and emits a
# misleading startup failure. The study environment must not inherit it.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"ERROR: expected Kaggle Python 3.12, found {sys.version.split()[0]}"
    )
PY

if [[ ! -x "${VENV_DIR}/bin/python" || ! -x "${VENV_DIR}/bin/pip" ]]; then
    printf 'Creating isolated environment at %s...\n' "${VENV_DIR}"
    # Kaggle's base Python may not ship a working ensurepip, so stdlib venv can
    # leave a half-created environment. virtualenv carries its own pip seed and
    # --clear safely repairs that script-owned directory on a repeated run.
    python -m pip install --quiet --disable-pip-version-check "virtualenv==20.29.1"
    python -m virtualenv --clear --python "$(command -v python)" "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

printf 'GPU detected. Installing the pinned Needle study environment...\n'
python -m pip install --quiet --upgrade "pip<26"
python -m pip install --quiet --no-cache-dir --force-reinstall \
    "numpy==${NUMPY_VERSION}" \
    "scipy==${SCIPY_VERSION}" \
    "jax[cuda12]==${JAX_VERSION}" \
    "flax==${FLAX_VERSION}" \
    "optax==${OPTAX_VERSION}" \
    "datasets==${DATASETS_VERSION}" \
    "huggingface_hub==${HF_HUB_VERSION}" \
    "safetensors==0.4.5" \
    "sentencepiece==0.2.0" \
    "PyYAML==6.0.2" \
    "wrapt==1.17.2" \
    "pytest==8.3.4"

# The train extra is intentionally not used: it contains unpinned JAX and NumPy
# requirements and can undo the compatible environment above.
python -m pip install --quiet --no-deps -e .

printf 'Verifying NumPy and JAX in a clean Python process...\n'
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false python - <<'PY'
import numpy as np
import scipy
import jax

print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("JAX:", jax.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())

assert np.__version__ == "2.0.2"
assert scipy.__version__ == "1.14.1"
assert jax.__version__ == "0.7.2"
assert jax.default_backend() == "gpu"
assert jax.devices(), "JAX returned no devices"
PY

printf 'Downloading the Needle 3 training checkpoint...\n'
mkdir -p checkpoints data/raw/droidcall data/raw/mobile_actions data/raw/fineweb_edu
python - <<'PY'
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download
from needle.agent.fetch import fetch_checkpoint

fetch_checkpoint("needle3.safetensors", "checkpoints", generation=3)

downloads = {
    "data/raw/droidcall/DroidCall_train.jsonl": (
        "mllmTeam/DroidCall", "DroidCall_train.jsonl"
    ),
    "data/raw/droidcall/DroidCall_test.jsonl": (
        "mllmTeam/DroidCall", "DroidCall_test.jsonl"
    ),
    "data/raw/mobile_actions/dataset.jsonl": (
        "google/mobile-actions", "dataset.jsonl"
    ),
}
for destination, (repo_id, filename) in downloads.items():
    source = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print("Downloaded", target)
PY

if [[ ! -d data/raw/dstc8/.git ]]; then
    git clone --quiet --depth 1 \
        https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git \
        data/raw/dstc8
fi

if [[ ! -d data/raw/snips/.git ]]; then
    git clone --quiet --depth 1 \
        https://github.com/snipsco/nlu-benchmark.git \
        data/raw/snips
fi

if [[ ! -s data/raw/fineweb_edu/train.jsonl ]]; then
    printf 'Downloading a deterministic 2,000-row FineWeb-Edu sample...\n'
    python - <<'PY'
import json
from pathlib import Path

from datasets import load_dataset

output = Path("data/raw/fineweb_edu/train.jsonl")
stream = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    "sample-10BT",
    split="train",
    streaming=True,
)
with output.open("w", encoding="utf-8") as handle:
    for index, row in enumerate(stream):
        if index == 2000:
            break
        handle.write(json.dumps({
            "text": row["text"],
            "is_language_sample": True,
        }, ensure_ascii=False) + "\n")
print("Downloaded", output, flush=True)

# Some Kaggle images crash while PyArrow tears down background state after a
# streaming dataset closes. The file is complete at this point, so skip Python
# extension finalizers and return success to the shell.
import os
os._exit(0)
PY
fi

python -m needle.study.kaggle_data

printf 'Building the immutable manifest and frozen teacher caches...\n'
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m needle.study prepare --config study.yaml

printf '\nSetup completed successfully.\n'
printf 'Use .venv-kaggle/bin/python for every study command. No notebook restart is required.\n'
printf 'Verify later with: .venv-kaggle/bin/python -c "import jax; print(jax.devices())"\n'
printf 'Datasets are normalized under data/normalized and teacher caches are in study_runs/teachers.\n'
