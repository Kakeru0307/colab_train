"""target/input muspy → lead pairs ディレクトリへ保存（power mask はゼロ可）。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import muspy
import numpy as np

from util.density_cond import bpm_to_unit
from util.midi_to_patch import PATCH_TICKS, MidiPatch, midi_to_patches


def _music_to_patches(music: muspy.Music, tmp_midi: Path) -> list[MidiPatch]:
    muspy.write_midi(tmp_midi, music)
    return midi_to_patches(tmp_midi)


def _has_onsets(patch: MidiPatch, min_onsets: int) -> bool:
    return int((patch.tonal_chw == 1).sum()) >= min_onsets


def save_lead_pair_song(
    pairs_dir: Path,
    song_id: str,
    *,
    input_music: muspy.Music,
    target_music: muspy.Music,
    bpm: float,
    min_onsets: int = 1,
    zero_power_mask: bool = True,
) -> int:
    """1曲分を保存。保存パッチ数を返す。"""
    input_root = pairs_dir / "input" / song_id
    target_root = pairs_dir / "target" / song_id
    input_root.mkdir(parents=True, exist_ok=True)
    target_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_in = Path(tmp) / "in.mid"
        tmp_tg = Path(tmp) / "tg.mid"
        inp_patches = _music_to_patches(input_music, tmp_in)
        tgt_patches = _music_to_patches(target_music, tmp_tg)

    saved = 0
    for inp, tgt in zip(inp_patches, tgt_patches):
        if not _has_onsets(tgt, min_onsets):
            continue
        stem = f"bar{tgt.bar_index:04d}"
        np.save(input_root / f"{stem}_tonal.npy", inp.tonal_chw.astype(np.uint8, copy=False))
        np.save(target_root / f"{stem}_tonal.npy", tgt.tonal_chw.astype(np.uint8, copy=False))
        np.save(input_root / f"{stem}_cond.npy", np.float32(bpm_to_unit(bpm)))
        if zero_power_mask:
            np.save(
                input_root / f"{stem}_power.npy",
                np.zeros((PATCH_TICKS, 128), dtype=np.uint8),
            )
        saved += 1
    return saved
