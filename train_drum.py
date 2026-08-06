"""ドラム専用 U-Net 学習（in=24 tonal+BPM+beat → out=1 drum、BCEWithLogitsLoss）。

入力: tonal11 + BPM1 + beat_type one-hot12 = 24ch。
学習実行は Colab 側で行う。ローカルでのデフォルト実行は想定しない。
旧 in=12 ckpt はチャネル不一致で非互換。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim

from dataset_drum import get_drum_dataloader
from model import build_unet

SCRIPT_DIR = Path(__file__).resolve().parent


def train_drum(
    pairs_dir: Path,
    checkpoint_dir: Path,
    *,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-3,
    pos_weight: float = 20.0,
    resume: Path | None = None,
    encoder_weights: str | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    input_dir = pairs_dir / "input"
    target_dir = pairs_dir / "target"
    if not input_dir.is_dir() or not target_dir.is_dir():
        raise FileNotFoundError(
            f"pairs がありません: {pairs_dir} (input/ と target/ が必要)"
        )

    loader = get_drum_dataloader(
        input_dir,
        target_dir,
        batch_size=batch_size,
        shuffle=True,
    )
    sample_x, sample_y = next(iter(loader))
    in_ch = int(sample_x.shape[1])
    out_ch = int(sample_y.shape[1])
    print(f"device={device} in_channels={in_ch} out_channels={out_ch} batches={len(loader)}")

    model = build_unet(
        in_channels=in_ch,
        out_channels=out_ch,
        encoder_weights=encoder_weights,
    ).to(device)

    if resume is not None and Path(resume).is_file():
        blob = torch.load(resume, map_location=device, weights_only=False)
        state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
        model.load_state_dict(state)
        print(f"resumed from {resume}")

    pw = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n += 1
        mean_loss = total / max(1, n)
        print(f"epoch {epoch:03d}  loss={mean_loss:.4f}")

        ckpt = {
            "model_state_dict": model.state_dict(),
            "in_channels": in_ch,
            "out_channels": out_ch,
            "model_type": "unet_drum",
            "epochs": epoch,
            "lr": lr,
            "pos_weight": pos_weight,
            "loss": mean_loss,
        }
        torch.save(ckpt, checkpoint_dir / "unet_last.pt")
        if epoch % 5 == 0 or epoch == epochs:
            torch.save(ckpt, checkpoint_dir / f"unet_ep{epoch:03d}.pt")

    print(f"wrote {checkpoint_dir / 'unet_last.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train drum U-Net (in=12 → out=1)")
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "pairs" / "drum",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "drum",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=20.0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--encoder-weights", type=str, default=None)
    args = parser.parse_args()

    train_drum(
        args.pairs_dir,
        args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pos_weight=args.pos_weight,
        resume=args.resume,
        encoder_weights=args.encoder_weights,
    )


if __name__ == "__main__":
    main()
