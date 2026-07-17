"""raw MIDI から学習用 input/target パッチペアを一括生成する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from midi_to_patch import MidiPatch, midi_to_patches
from skeleton import PAIR_MODES, make_input_tonal


def song_id_from_path(midi_path: Path, raw_root: Path) -> str:
    rel = midi_path.relative_to(raw_root)
    return str(rel.with_suffix("")).replace("\\", "/")


def patch_has_notes(patch: MidiPatch, *, min_onsets: int = 1) -> bool:
    onsets = int((patch.tonal_chw == 1).sum())
    return onsets >= min_onsets


SCRIPT_DIR = Path(__file__).resolve().parent


def save_pair_patches(
    patches: list[MidiPatch],
    input_dir: Path,
    target_dir: Path,
    *,
    mode: str,
    min_onsets: int,
) -> int:
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for patch in patches:
        if not patch_has_notes(patch, min_onsets=min_onsets):
            continue

        target_tonal = patch.tonal_chw
        input_tonal = make_input_tonal(target_tonal, mode)
        stem = f"bar{patch.bar_index:04d}"

        np.save(input_dir / f"{stem}_tonal.npy", input_tonal.astype(np.uint8, copy=False))
        np.save(target_dir / f"{stem}_tonal.npy", target_tonal.astype(np.uint8, copy=False))
        # drum は現行学習で未使用のため保存しない（Colab ディスク節約）
        saved += 1

    return saved


def prepare_pairs(
    raw_dir: Path,
    pairs_dir: Path,
    *,
    mode: str = "onset_to_full",
    min_onsets: int = 1,
    categories: list[str] | None = None,
) -> dict:
    input_root = pairs_dir / "input"
    target_root = pairs_dir / "target"
    midi_files = sorted(raw_dir.rglob("*.mid"))

    if categories:
        midi_files = [
            path
            for path in midi_files
            if path.relative_to(raw_dir).parts[0] in categories
        ]

    if not midi_files:
        raise FileNotFoundError(f"MIDI が見つかりません: {raw_dir}")

    stats = {
        "mode": mode,
        "raw_dir": str(raw_dir),
        "pairs_dir": str(pairs_dir),
        "midi_files": len(midi_files),
        "songs": [],
        "total_patches": 0,
    }

    for midi_path in midi_files:
        song_id = song_id_from_path(midi_path, raw_dir)
        patches = midi_to_patches(midi_path)
        saved = save_pair_patches(
            patches,
            input_root / song_id,
            target_root / song_id,
            mode=mode,
            min_onsets=min_onsets,
        )
        stats["songs"].append(
            {
                "song_id": song_id,
                "midi": str(midi_path),
                "patches": saved,
            }
        )
        stats["total_patches"] += saved
        print(f"{song_id}: {saved} patches")

    manifest_path = pairs_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n合計: {stats['total_patches']} パッチ ({stats['midi_files']} MIDI)")
    print(f"保存先: {pairs_dir}")
    print(f"manifest: {manifest_path}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="raw MIDI から U-Net 学習用 input/target パッチペアを生成",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "raw" / "guitar-techs",
        help="生 MIDI のルート（サブフォルダごとに曲 ID 化）",
    )
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "pairs" / "guitar-techs",
        help="output/input と output/target を作る先",
    )
    parser.add_argument(
        "--mode",
        choices=PAIR_MODES,
        default="onset_to_full",
        help="identity / onset_to_full / downbeat_chord / root_per_bar / melody_line",
    )
    parser.add_argument(
        "--min-onsets",
        type=int,
        default=1,
        help="パッチを残す最小オンセット数",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="例: P3_music P1_techniques（指定時は該当フォルダのみ）",
    )
    args = parser.parse_args()

    prepare_pairs(
        args.raw_dir,
        args.pairs_dir,
        mode=args.mode,
        min_onsets=args.min_onsets,
        categories=args.categories,
    )


if __name__ == "__main__":
    main()
