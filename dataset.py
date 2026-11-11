"""パッチ Dataset（通常: tonal11+BPM1、lead: さらにpower attack mask1）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from util.density_cond import make_bpm_cond_map
from util.midi_to_patch import normalize_pianoroll


class PatchPairDataset(Dataset):
    """入力・正解の tonal パッチ (.npy) ペアを読み込む。

    同階層の ``*_cond.npy`` に正規化 BPM（スカラーまたは (1,H,W)）を置く。
    無い場合は中立 0.5（旧ペア互換・非推奨）。

    cache=True（デフォルト）のとき、初回に全件をメモリに展開する。
    データセットが GPU メモリに乗らない場合でも CPU RAM に乗れば有効。
    """

    def __init__(
        self,
        input_dir: str | Path,
        target_dir: str | Path | None = None,
        *,
        require_power_cond: bool = False,
        section_cond: bool = False,
        require_section_cond: bool = False,
        cache: bool = True,
    ):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir) if target_dir else self.input_dir
        self.require_power_cond = require_power_cond
        self.section_cond = bool(section_cond or require_section_cond)
        self.require_section_cond = require_section_cond
        self.files = sorted(self.input_dir.rglob("*_tonal.npy"))
        if not self.files:
            raise FileNotFoundError(f"パッチが見つかりません: {self.input_dir}")

        self._cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if cache:
            print(f"[dataset] キャッシュ構築中 ({len(self.files)} 件)…")
            self._cache = [self._load(i) for i in range(len(self.files))]
            print("[dataset] キャッシュ完了")

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
        return np.full((1, height, width), np.float32(0.5), dtype=np.float32)

    def _load_power_cond(
        self,
        input_path: Path,
        height: int,
        width: int,
    ) -> np.ndarray | None:
        power_path = input_path.with_name(
            input_path.name.replace("_tonal.npy", "_power.npy")
        )
        if not power_path.is_file():
            if self.require_power_cond:
                raise FileNotFoundError(
                    f"power maskがありません。lead pairsを再生成してください: "
                    f"{power_path}"
                )
            return None
        arr = np.asarray(np.load(power_path), dtype=np.float32)
        if arr.shape == (height, width):
            return arr.reshape(1, height, width)
        if arr.shape == (1, height, width):
            return arr
        raise ValueError(
            f"power mask shape mismatch: {power_path} {arr.shape} "
            f"!= {(height, width)}"
        )

    def _load_section_cond(
        self,
        input_path: Path,
        height: int,
        width: int,
    ) -> np.ndarray | None:
        if not self.section_cond:
            return None
        from song_form import SECTION_ROLES, make_section_onehot_map

        section_path = input_path.with_name(
            input_path.name.replace("_tonal.npy", "_section.npy")
        )
        n = len(SECTION_ROLES)
        if not section_path.is_file():
            if self.require_section_cond:
                raise FileNotFoundError(
                    f"section ラベルがありません: {section_path}"
                )
            return np.zeros((n, height, width), dtype=np.float32)
        sid = int(np.asarray(np.load(section_path)).reshape(-1)[0])
        return make_section_onehot_map(sid, height=height, width=width)

    def _load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_path = self.files[idx]
        rel_path = input_path.relative_to(self.input_dir)
        target_path = self.target_dir / rel_path

        input_arr = normalize_pianoroll(np.load(input_path))
        if self.target_dir == self.input_dir:
            target_arr = input_arr.copy()
        else:
            target_arr = normalize_pianoroll(np.load(target_path))

        cond = self._load_cond(input_path, input_arr.shape[1], input_arr.shape[2])
        channels = [input_arr.astype(np.float32), cond]
        power = self._load_power_cond(
            input_path, input_arr.shape[1], input_arr.shape[2]
        )
        if power is not None:
            channels.append(power)
        section = self._load_section_cond(
            input_path, input_arr.shape[1], input_arr.shape[2]
        )
        if section is not None:
            channels.append(section)
        model_in = np.concatenate(channels, axis=0)

        return (
            torch.from_numpy(model_in),
            torch.from_numpy(target_arr.astype(np.float32)),
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            return self._cache[idx]
        return self._load(idx)


class SinglePatchDataset(Dataset):
    """1 枚のパッチを繰り返し返す（過学習テスト用）。"""

    def __init__(self, patch_path: str | Path, length: int = 32, bpm: float = 120.0):
        tonal = normalize_pianoroll(np.load(patch_path))
        cond = make_bpm_cond_map(bpm, height=tonal.shape[1], width=tonal.shape[2])
        self.patch_in = torch.from_numpy(
            np.concatenate([tonal.astype(np.float32), cond], axis=0)
        )
        self.patch_tg = torch.from_numpy(tonal.astype(np.float32))
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.patch_in, self.patch_tg


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    import torch
    pin = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
