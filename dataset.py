"""パッチ画像の Dataset / DataLoader。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from midi_to_patch import normalize_pianoroll


class PatchPairDataset(Dataset):
    """入力・正解の tonal パッチ (.npy) ペアを読み込む。

    input_dir 直下またはサブフォルダ内の ``*_tonal.npy`` を再帰的に収集する。
    """

    def __init__(self, input_dir: str | Path, target_dir: str | Path | None = None):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir) if target_dir else self.input_dir
        self.files = sorted(self.input_dir.rglob("*_tonal.npy"))
        if not self.files:
            raise FileNotFoundError(f"パッチが見つかりません: {self.input_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_path = self.files[idx]
        rel_path = input_path.relative_to(self.input_dir)
        target_path = self.target_dir / rel_path

        input_arr = normalize_pianoroll(np.load(input_path))
        if self.target_dir == self.input_dir:
            target_arr = input_arr.copy()
        else:
            target_arr = normalize_pianoroll(np.load(target_path))

        return (
            torch.from_numpy(input_arr),
            torch.from_numpy(target_arr),
        )


class SinglePatchDataset(Dataset):
    """1 枚のパッチを繰り返し返す（過学習テスト用）。"""

    def __init__(self, patch_path: str | Path, length: int = 32):
        self.patch = torch.from_numpy(
            normalize_pianoroll(np.load(patch_path))
        )
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.patch, self.patch


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
