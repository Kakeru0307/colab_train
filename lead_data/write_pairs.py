"""target/input muspy → lead pairs ディレクトリへ保存（power mask はゼロ可）。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import muspy
import numpy as np

from lead_data.quality import LeadQualityParams, is_clear_lead_patch
from util.density_cond import bpm_to_unit
from util.midi_to_patch import PATCH_TICKS, MidiPatch, midi_to_patches


def _music_to_patches(music: muspy.Music, tmp_midi: Path) -> list[MidiPatch]:
    muspy.write_midi(tmp_midi, music)
    return midi_to_patches(tmp_midi)


def save_lead_pair_song(
    pairs_dir: Path,
    song_id: str,
    *,
    input_music: muspy.Music,
    target_music: muspy.Music,
    bpm: float,
    min_onsets: int = 4,
    zero_power_mask: bool = True,
    section_id: int | None = None,
    quality: LeadQualityParams | None = None,
) -> int:
    """1曲分を保存。発音がはっきりしたパッチだけ残す。保存数を返す。

    section_id を渡すと input 側に ``*_section.npy``（スカラー int）を書く。
    """
    base = quality or LeadQualityParams()
    params = LeadQualityParams(
        min_onsets=int(min_onsets),
        min_onset_ratio=base.min_onset_ratio,
        max_fill=base.max_fill,
        min_fill=base.min_fill,
        min_active_ticks=base.min_active_ticks,
    )

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
        if not is_clear_lead_patch(tgt, params):
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
        if section_id is not None:
            np.save(input_root / f"{stem}_section.npy", np.int32(section_id))
        saved += 1
    return saved
