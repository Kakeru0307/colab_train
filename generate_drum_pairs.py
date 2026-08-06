"""ドラム学習用の input/target パッチペアを生成する。

input  : ギターコード骨格 (downbeat_chord tonal 11ch) + BPM cond
target : ドラムパターン (*_drum.npy, shape (1,128,128), 0/1 二値)

bass と同一の乱数派生で progression/key/bpm/energy を揃える想定だが、
単体実行でも独立に学習ペアを生成できる。
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Literal

import muspy
import numpy as np

from density_cond import bpm_to_unit
from makeData.constants import BPM_RANGE, DEFAULT_BARS, KEYS
from makeData.drum import generate_drum_pattern
from makeData.patterns import choose_progression
from midi_to_patch import MidiPatch, midi_to_patches
from progression_input import build_backing_skeleton_music

SCRIPT_DIR = Path(__file__).resolve().parent
EnergyLevel = Literal["low", "mid", "high"]


def _music_to_patches(music: muspy.Music, tmp_midi: Path) -> list[MidiPatch]:
    muspy.write_midi(tmp_midi, music)
    return midi_to_patches(tmp_midi)


def _binarize_drum(drum_chw: np.ndarray) -> np.ndarray:
    """onset=1 / sustain=2 → 0/1 二値 (1,128,128)。"""
    return (drum_chw > 0).astype(np.uint8)


def _has_drum_hits(patch: MidiPatch, min_hits: int) -> bool:
    return int((patch.drum > 0).sum()) >= min_hits


def _sample_energy(rng: random.Random, bpm: float) -> EnergyLevel:
    if bpm < 85:
        return rng.choices(("low", "mid", "high"), weights=(0.45, 0.40, 0.15), k=1)[0]
    if bpm < 120:
        return rng.choices(("low", "mid", "high"), weights=(0.20, 0.55, 0.25), k=1)[0]
    return rng.choices(("low", "mid", "high"), weights=(0.10, 0.35, 0.55), k=1)[0]


def generate_drum_pairs(
    pairs_dir: Path,
    *,
    count: int,
    seed: int = 42,
    bars: int = DEFAULT_BARS,
    min_hits: int = 4,
) -> dict:
    input_root = pairs_dir / "input"
    target_root = pairs_dir / "target"
    rng = random.Random(seed)

    stats: dict = {
        "mode": "drum",
        "pairs_dir": str(pairs_dir),
        "songs": [],
        "total_patches": 0,
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_in = Path(tmp) / "in.mid"
        tmp_tg = Path(tmp) / "tg.mid"
        for i in range(count):
            song_rng = random.Random(rng.randint(0, 2**31 - 1))
            spec = choose_progression(song_rng)
            key = song_rng.choice(KEYS)
            bpm = float(song_rng.randint(*BPM_RANGE))
            bars_per_chord = song_rng.choice((1, 1, 1, 2))
            energy = _sample_energy(song_rng, bpm)

            target_music = generate_drum_pattern(
                bpm=bpm,
                bars=bars,
                energy=energy,
                rng=song_rng,
            )
            input_music = build_backing_skeleton_music(
                progression=spec,
                key=key,
                bars=bars,
                bpm=bpm,
                bars_per_chord=bars_per_chord,
            )

            tgt_patches = _music_to_patches(target_music, tmp_tg)
            inp_patches = _music_to_patches(input_music, tmp_in)
            if not tgt_patches or not inp_patches:
                continue

            song_id = f"drum{i:05d}"
            in_dir = input_root / song_id
            tg_dir = target_root / song_id
            in_dir.mkdir(parents=True, exist_ok=True)
            tg_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for inp, tgt in zip(inp_patches, tgt_patches):
                if not _has_drum_hits(tgt, min_hits):
                    continue
                stem = f"bar{tgt.bar_index:04d}"
                np.save(
                    in_dir / f"{stem}_tonal.npy",
                    inp.tonal_chw.astype(np.uint8, copy=False),
                )
                # target は *_drum.npy（1ch 二値）。ファイル名は stem_drum.npy
                np.save(
                    tg_dir / f"{stem}_drum.npy",
                    _binarize_drum(tgt.drum_chw),
                )
                np.save(in_dir / f"{stem}_cond.npy", np.float32(bpm_to_unit(bpm)))
                saved += 1

            stats["songs"].append(
                {
                    "song_id": song_id,
                    "progression": spec.name,
                    "key": key,
                    "bpm": bpm,
                    "energy": energy,
                    "patches": saved,
                }
            )
            stats["total_patches"] += saved
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{count} 本 ... 累計 {stats['total_patches']} パッチ")

    pairs_dir.mkdir(parents=True, exist_ok=True)
    with open(pairs_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"合計: {stats['total_patches']} パッチ / {len(stats['songs'])} 本")
    print(f"保存先: {pairs_dir}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="ドラム学習用 input/target ペアを生成")
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "pairs" / "drum",
    )
    parser.add_argument("--count", type=int, default=2000, help="生成する曲数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bars", type=int, default=DEFAULT_BARS)
    parser.add_argument("--min-hits", type=int, default=4)
    args = parser.parse_args()

    generate_drum_pairs(
        args.pairs_dir,
        count=args.count,
        seed=args.seed,
        bars=args.bars,
        min_hits=args.min_hits,
    )


if __name__ == "__main__":
    main()
