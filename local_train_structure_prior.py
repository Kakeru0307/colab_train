"""Structure prior を家のローカル（CPU/GPU）で学習する入口。Colab と併用可。

前提:
  - このディレクトリが cwd（colab_train）
  - data/memo2496_pairs/structure_pairs.jsonl 等があること
  - GPU を使うなら CUDA 版 torch（現状リポ venv が CPU のみのことがある）

使い方:
  cd colab_train
  ..\\venv\\Scripts\\python.exe local_train_structure_prior.py
  ..\\venv\\Scripts\\python.exe local_train_structure_prior.py --epochs 80 --batch-size 64

スコア・ckpt は checkpoints/structure_prior/structure_prior_last.pt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _check_torch() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "torch がありません。リポ venv で torch を入れてください。"
        ) from exc
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA OK: {name}  VRAM≈{mem:.1f} GiB")
    else:
        print(
            "警告: CUDA が見えません → CPU 学習になります。\n"
            "  RTX を使うには CUDA 版 torch が必要です（例）:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            "  （ドライバ／CUDA 版は環境に合わせて選ぶ）",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local structure prior training (VA→BPM/key/mode/prog)"
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints" / "structure_prior",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=ROOT / "data" / "prior_pairs" / "manifests" / "accept.jsonl",
    )
    parser.add_argument(
        "--no-memo",
        action="store_true",
        help="memo2496_pairs を extras に載せない",
    )
    parser.add_argument(
        "--no-balanced",
        action="store_true",
        help="va_bpm_curated balanced_116 を載せない",
    )
    parser.add_argument(
        "--no-emopia",
        action="store_true",
        help="emopia_pairs を載せない",
    )
    args = parser.parse_args()

    _check_torch()

    extras: list[Path] = []
    bal = ROOT / "data" / "va_bpm_curated" / "structure_pairs_balanced_116.jsonl"
    memo = ROOT / "data" / "memo2496_pairs" / "structure_pairs.jsonl"
    emo = ROOT / "data" / "emopia_pairs" / "structure_pairs.jsonl"
    if not args.no_balanced and bal.is_file():
        extras.append(bal)
    elif not args.no_balanced:
        print(f"[警告] missing {bal}", file=sys.stderr)
    if not args.no_memo and memo.is_file():
        extras.append(memo)
    elif not args.no_memo:
        print(f"[警告] missing {memo}", file=sys.stderr)
    if not args.no_emopia and emo.is_file():
        extras.append(emo)

    jsonl = args.jsonl
    if not jsonl.is_file():
        raise SystemExit(f"accept jsonl がありません: {jsonl}")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_structure_prior.py"),
        "--jsonl",
        str(jsonl),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--hidden",
        str(args.hidden),
        "--seed",
        str(args.seed),
    ]
    if extras:
        cmd.append("--extra-jsonl")
        cmd.extend(str(p) for p in extras)

    print("cmd:", " ".join(cmd))
    print("extras:", [str(p) for p in extras])
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
