"""ドラム学習用 Dataset（input tonal+BPM → target drum 1ch 二値）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from density_cond import make_bpm_cond_map
from midi_to_patch import normalize_pianoroll


class DrumPairDataset(Dataset):
    """input/*_tonal.npy + *_cond.npy → target/*_drum.npy (1,H,W) 0/1。"""

    def __init__(self, input_dir: str | Path, target_dir: str | Path):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.files = sorted(self.input_dir.rglob("*_tonal.npy"))
        if not self.files:
            raise FileNotFoundError(f"パッチが見つかりません: {self.input_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _load_cond(self, input_path: Path, height: int, width: int) -> np.ndarray:
        cond_path = input_path.with_name(
            input_path.name.replace("_tonal.npy", "_cond.npy")
        )
        if cond_path.is_file():
            raw = np.load(cond_path)
            arr = np.asarray(raw)
            if arr.ndim >= 2 and arr.shape[-2:] == (height, width):
                if arr.ndim == 2:
                    return arr.reshape(1, height, width).astype(np.float32, copy=False)
                return arr.astype(np.float32, copy=False)[:1]
            unit = float(arr.reshape(-1)[0])
            return np.full((1, height, width), np.float32(unit), dtype=np.float32)
        return make_bpm_cond_map(105.0, height=height, width=width)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_path = self.files[idx]
        rel = input_path.relative_to(self.input_dir)
        target_path = self.target_dir / rel.with_name(
            rel.name.replace("_tonal.npy", "_drum.npy")
        )
        if not target_path.is_file():
            raise FileNotFoundError(f"drum target がありません: {target_path}")

        input_arr = normalize_pianoroll(np.load(input_path))
        drum = np.asarray(np.load(target_path), dtype=np.float32)
        if drum.ndim == 2:
            drum = drum[np.newaxis, ...]
        drum = (drum > 0).astype(np.float32)

        cond = self._load_cond(input_path, input_arr.shape[1], input_arr.shape[2])
        model_in = np.concatenate(
            [input_arr.astype(np.float32), cond],
            axis=0,
        )
        return (
            torch.from_numpy(model_in),
            torch.from_numpy(drum),
        )


def get_drum_dataloader(
    input_dir: Path,
    target_dir: Path,
    *,
    batch_size: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    ds = DrumPairDataset(input_dir, target_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
