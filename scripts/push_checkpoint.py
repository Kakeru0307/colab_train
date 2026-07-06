"""Colab 学習後にチェックポイントを GitHub へ push する。"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "checkpoints" / "unet_last.pt"


def run(cmd: list[str], *, cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def resolve_ckpt_path(ckpt: Path) -> Path:
    """相対パス指定でも ROOT 基準で解決する。"""
    if ckpt.is_absolute():
        return ckpt.resolve()
    return (ROOT / ckpt).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", default="Update checkpoint from Colab")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = parser.parse_args()

    ckpt_path = resolve_ckpt_path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"チェックポイントがありません: {ckpt_path}")
    if ROOT not in ckpt_path.parents and ckpt_path != ROOT:
        raise ValueError(f"チェックポイントはリポジトリ内である必要があります: {ckpt_path}")

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "環境変数 GITHUB_TOKEN を設定してください（GitHub Personal Access Token）"
        )

    remote_url = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=ROOT, text=True
    ).strip()
    if remote_url.startswith("https://") and "@" not in remote_url:
        authed = remote_url.replace("https://", f"https://{token}@")
        run(["git", "remote", "set-url", "origin", authed], cwd=ROOT)

    run(["git", "add", str(ckpt_path.relative_to(ROOT))], cwd=ROOT)
    run(["git", "commit", "-m", args.message], cwd=ROOT)
    run(["git", "push", "origin", "HEAD"], cwd=ROOT)
    print("push 完了")


if __name__ == "__main__":
    main()
