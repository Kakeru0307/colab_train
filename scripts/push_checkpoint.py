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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", default="Update checkpoint from Colab")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = parser.parse_args()

    if not args.ckpt.is_file():
        raise FileNotFoundError(f"チェックポイントがありません: {args.ckpt}")

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

    run(["git", "add", str(args.ckpt.relative_to(ROOT))], cwd=ROOT)
    run(["git", "commit", "-m", args.message], cwd=ROOT)
    run(["git", "push", "origin", "HEAD"], cwd=ROOT)
    print("push 完了")


if __name__ == "__main__":
    main()
