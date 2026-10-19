"""Create an isolated Python 3.11 venv for FAD and verify imports/versions.

Colab のシステム Python (3.13.x / numpy>=2) には触らない。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements-fad.txt"
DEFAULT_VENV = Path(os.environ.get("FAD_VENV", "/content/fad_venv"))


def _run(cmd: list[str], *, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _ensure_uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    home_uv = Path.home() / ".local" / "bin" / "uv"
    if home_uv.is_file():
        return str(home_uv)
    _run(["bash", "-lc", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
    uv = shutil.which("uv") or str(home_uv)
    if not Path(uv).is_file():
        raise RuntimeError("uv のインストールに失敗しました")
    return uv


def _cuda_extra_index() -> str | None:
    """nvidia-smi の CUDA Version から PyTorch wheel index を選ぶ。"""
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if not m:
        return None
    major, minor = int(m.group(1)), int(m.group(2))
    # driver が対応する最大 CUDA。新しい方を優先。
    if major >= 13:
        return "https://download.pytorch.org/whl/cu130"
    if major == 12 and minor >= 8:
        return "https://download.pytorch.org/whl/cu128"
    if major == 12:
        return "https://download.pytorch.org/whl/cu126"
    return "https://download.pytorch.org/whl/cu126"


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _verify(py: Path) -> None:
    code = r"""
import sys
from importlib.metadata import version, PackageNotFoundError

assert sys.version_info[:2] == (3, 11), sys.version

import numpy
import scipy
import soundfile
import torch
import fadtk
from fadtk.model_loader import CLAPLaionModel

nv = tuple(int(x) for x in numpy.__version__.split(".")[:2])
assert nv < (2, 0), numpy.__version__
sv = tuple(int(x) for x in scipy.__version__.split(".")[:1])
assert sv < (2,), scipy.__version__

try:
    fad_ver = version("fadtk")
except PackageNotFoundError:
    fad_ver = "?"

print("python", sys.version.split()[0])
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("fadtk", fad_ver)
print("cuda", torch.cuda.is_available(), end="")
if torch.cuda.is_available():
    print("", torch.cuda.get_device_name(0))
else:
    print(" (CPU — Colab なら GPU ランタイムか torch CUDA wheel を確認)")
# モデル定義だけ触る（重み DL は初回 compute 時）
assert CLAPLaionModel is not None
print("verify ok")
"""
    _run([str(py), "-c", code])


def main() -> None:
    if not REQ.is_file():
        raise SystemExit(f"missing {REQ}")

    uv = _ensure_uv()
    env = os.environ.copy()
    local_bin = str(Path.home() / ".local" / "bin")
    env["PATH"] = local_bin + os.pathsep + env.get("PATH", "")

    venv = DEFAULT_VENV
    py = _venv_python(venv)

    _run([uv, "python", "install", "3.11"], env=env)
    _run([uv, "venv", str(venv), "--python", "3.11", "--clear"], env=env)

    index = _cuda_extra_index()
    install = [uv, "pip", "install", "--python", str(py), "-r", str(REQ)]
    if index:
        print(f"[fad-env] torch extra-index: {index}", flush=True)
        install.extend(["--extra-index-url", index])
    else:
        print("[fad-env] nvidia-smi なし → PyPI 既定の torch", flush=True)
    _run(install, env=env)

    # Linux + GPU なのに CPU torch になったら CUDA index で torch だけ入れ直す
    check = subprocess.run(
        [str(py), "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"],
        check=False,
    )
    if check.returncode != 0 and index:
        print("[fad-env] cuda unavailable; reinstall torch from CUDA index", flush=True)
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(py),
                "--reinstall-package",
                "torch",
                "--reinstall-package",
                "torchvision",
                "--index-url",
                index,
                "--extra-index-url",
                "https://pypi.org/simple",
                "torch>=2.3.0,<3",
                "torchvision>=0.22.0",
            ],
            env=env,
        )

    _verify(py)
    print(f"FAD_PY={py}", flush=True)


if __name__ == "__main__":
    main()
