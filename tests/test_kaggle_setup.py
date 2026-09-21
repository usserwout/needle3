from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "setup-kaggle.sh"


def test_kaggle_setup_is_valid_bash_and_guards_gpu_before_installing():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    source = SCRIPT.read_text()
    gpu_check = source.index("nvidia-smi")
    venv_creation = source.index('python -m venv "${VENV_DIR}"')
    package_install = source.index("pip install")
    assert gpu_check < package_install
    assert venv_creation < package_install
    assert 'readonly JAX_VERSION="0.4.38"' in source
    assert '"jax[cuda12]==${JAX_VERSION}"' in source
    assert 'readonly NUMPY_VERSION="2.0.2"' in source
    assert '"numpy==${NUMPY_VERSION}"' in source
    assert "jax.default_backend()" in source
    assert "export CUDA_VISIBLE_DEVICES=0" in source
    assert 'source "${VENV_DIR}/bin/activate"' in source
    assert ".venv-kaggle/bin/python" in source
    assert "fetch_checkpoint" in source
    assert 'DroidCall_train.jsonl' in source
    assert 'google/mobile-actions' in source


def test_kaggle_setup_does_not_reinstall_unpinned_train_extra():
    source = SCRIPT.read_text()
    assert '.[train]' not in source
    assert "--no-deps -e ." in source
