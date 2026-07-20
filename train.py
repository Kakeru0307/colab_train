"""smp U-Net の学習スクリプト。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim

from dataset import PatchPairDataset, SinglePatchDataset, get_dataloader
from model import build_unet
from program_utils import GUITAR_PROGRAM

SCRIPT_DIR = Path(__file__).resolve().parent
GUITAR_CATEGORY = GUITAR_PROGRAM // 8


def weighted_mse_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: float = 10.0,
) -> torch.Tensor:
    """音符セル（非ゼロ）に重みを付けて、無音ばかり予測するのを防ぐ。"""
    weights = torch.ones_like(targets)
    weights[targets > 0] = pos_weight
    return (weights * (outputs - targets) ** 2).mean()


def resolve_pair_dirs(
    data_dir: Path | None,
    pairs_dir: Path | None,
) -> tuple[Path, Path | None]:
    if pairs_dir is not None:
        input_dir = pairs_dir / "input"
        target_dir = pairs_dir / "target"
        if not input_dir.is_dir():
            raise FileNotFoundError(f"input ディレクトリがありません: {input_dir}")
        if not target_dir.is_dir():
            raise FileNotFoundError(f"target ディレクトリがありません: {target_dir}")
        return input_dir, target_dir

    if data_dir is None:
        raise ValueError("--data-dir または --pairs-dir を指定してください")

    return data_dir, None


def train(
    data_dir: Path | None,
    checkpoint_dir: Path,
    *,
    pairs_dir: Path | None = None,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    overfit_single: bool = False,
    encoder_weights: str | None = None,
    pos_weight: float = 10.0,
    resume: Path | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    input_dir, target_dir = resolve_pair_dirs(data_dir, pairs_dir)

    if overfit_single:
        patch_files = sorted(input_dir.rglob("*_tonal.npy"))
        if not patch_files:
            raise FileNotFoundError(f"パッチが見つかりません: {input_dir}")
        dataset = SinglePatchDataset(patch_files[0], length=32)
    else:
        dataset = PatchPairDataset(input_dir, target_dir)
    print(f"学習パッチ数: {len(dataset)}")

    dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=True)
    model = build_unet(encoder_weights=encoder_weights).to(device)
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        prev_epochs = checkpoint.get("epochs", "?")
        print(f"再開: {resume} から重みを読み込み（前 epochs={prev_epochs}）")
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = weighted_mse_loss(outputs, targets, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch}/{epochs}  loss={avg_loss:.6f}")

    ckpt_path = checkpoint_dir / "unet_last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": epochs,
            "lr": lr,
            "model_type": "unet",
            "in_channels": 12,
            "out_channels": 11,
        },
        ckpt_path,
    )
    print(f"チェックポイント保存: {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="単一フォルダ学習（input=target）",
    )
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        default=None,
        help="prepare_dataset.py で作った pairs ルート（input/ と target/）",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=SCRIPT_DIR / "checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--overfit-single",
        action="store_true",
        help="1 パッチで過学習テスト",
    )
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="既存チェックポイントから重みを読み込んで学習を継続する",
    )
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default=None,
        help="例: imagenet（ネットワーク接続が必要）",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None and args.pairs_dir is None:
        data_dir = (
            SCRIPT_DIR / "stash" / "data_legacy" / "patches" / "test1"
        )

    train(
        data_dir,
        args.checkpoint_dir,
        pairs_dir=args.pairs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        overfit_single=args.overfit_single,
        encoder_weights=args.encoder_weights,
        pos_weight=args.pos_weight,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
