"""合成 MIDI + Guitar-TECHS を Colab 上で一括パッチ化する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from makeData.constants import DEFAULT_SYNTHETIC_COUNT
from makeData.generate import count_existing_midis, generate_dataset
from prepare_dataset import prepare_pairs
from remap_guitar_programs import remap_directory
from skeleton import PAIR_MODES

SCRIPT_DIR = Path(__file__).resolve().parent

RAW_SYNTHETIC = SCRIPT_DIR / "data" / "raw" / "synthetic"
RAW_GUITAR_TECHS = SCRIPT_DIR / "data" / "raw" / "guitar-techs"
PAIRS_SYNTHETIC = SCRIPT_DIR / "data" / "pairs" / "synthetic"
PAIRS_GUITAR_TECHS = SCRIPT_DIR / "data" / "pairs" / "guitar-techs"


def count_midi_files(raw_dir: Path) -> int:
    if not raw_dir.is_dir():
        return 0
    return len(list(raw_dir.rglob("*.mid")))


def ensure_synthetic_raw(
    *,
    count: int,
    seed: int,
    force_regenerate: bool = False,
) -> dict:
    existing = count_existing_midis(RAW_SYNTHETIC)
    if force_regenerate:
        print(f"合成 MIDI: {count} 本を先頭から再生成します")
        return generate_dataset(RAW_SYNTHETIC, count=count, seed=seed, start_index=0)

    if existing < count:
        print(f"合成 MIDI: 既存 {existing} 本 → 目標 {count} 本（+{count - existing} 本を追記）")
        return generate_dataset(
            RAW_SYNTHETIC,
            count=count,
            seed=seed,
            start_index=existing,
        )

    print(f"合成 MIDI: 既存 {existing} 本を使用（再生成スキップ）")
    manifest_path = RAW_SYNTHETIC / "manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    return {"count": existing, "seed": seed, "output_dir": str(RAW_SYNTHETIC)}


def prepare_all(
    *,
    synthetic_count: int = DEFAULT_SYNTHETIC_COUNT,
    synthetic_seed: int = 42,
    force_regenerate: bool = False,
    mode: str = "downbeat_chord",
    min_onsets: int = 1,
    skip_guitar_remap: bool = False,
) -> dict:
    print("=== Step 1: 合成 MIDI ===")
    synthetic_raw_manifest = ensure_synthetic_raw(
        count=synthetic_count,
        seed=synthetic_seed,
        force_regenerate=force_regenerate,
    )

    print("\n=== Step 2: Guitar-TECHS program 確認 ===")
    if not skip_guitar_remap and RAW_GUITAR_TECHS.is_dir():
        remap_directory(RAW_GUITAR_TECHS)
    else:
        print("Guitar-TECHS remap をスキップ")

    print("\n=== Step 3: 合成データをパッチ化 ===")
    synthetic_pairs = prepare_pairs(
        RAW_SYNTHETIC,
        PAIRS_SYNTHETIC,
        mode=mode,
        min_onsets=min_onsets,
    )

    print("\n=== Step 4: Guitar-TECHS をパッチ化 ===")
    if not RAW_GUITAR_TECHS.is_dir():
        raise FileNotFoundError(f"Guitar-TECHS がありません: {RAW_GUITAR_TECHS}")
    guitar_pairs = prepare_pairs(
        RAW_GUITAR_TECHS,
        PAIRS_GUITAR_TECHS,
        mode=mode,
        min_onsets=min_onsets,
    )

    summary = {
        "mode": mode,
        "synthetic": {
            "raw_dir": str(RAW_SYNTHETIC),
            "pairs_dir": str(PAIRS_SYNTHETIC),
            "midi_files": synthetic_raw_manifest.get("count", count_midi_files(RAW_SYNTHETIC)),
            "total_patches": synthetic_pairs["total_patches"],
        },
        "guitar_techs": {
            "raw_dir": str(RAW_GUITAR_TECHS),
            "pairs_dir": str(PAIRS_GUITAR_TECHS),
            "midi_files": guitar_pairs["midi_files"],
            "total_patches": guitar_pairs["total_patches"],
        },
        "total_patches": synthetic_pairs["total_patches"] + guitar_pairs["total_patches"],
    }

    summary_path = SCRIPT_DIR / "data" / "pairs" / "manifest_all.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 完了 ===")
    print(f"合成パッチ: {summary['synthetic']['total_patches']}")
    print(f"TECHS パッチ: {summary['guitar_techs']['total_patches']}")
    print(f"合計パッチ: {summary['total_patches']}")
    print(f"manifest: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合成 + Guitar-TECHS を一括パッチ化（Colab 用）",
    )
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=DEFAULT_SYNTHETIC_COUNT,
        help=f"目標 MIDI 本数（既定 {DEFAULT_SYNTHETIC_COUNT}、8小節≈1パッチ）",
    )
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="合成 MIDI を必ず再生成する",
    )
    parser.add_argument(
        "--mode",
        choices=PAIR_MODES,
        default="downbeat_chord",
        help="学習タスク: downbeat_chord=小節頭コード骨格→フル（推奨）",
    )
    parser.add_argument("--min-onsets", type=int, default=1)
    parser.add_argument(
        "--skip-guitar-remap",
        action="store_true",
        help="Guitar-TECHS の program 統一をスキップ",
    )
    args = parser.parse_args()

    prepare_all(
        synthetic_count=args.synthetic_count,
        synthetic_seed=args.synthetic_seed,
        force_regenerate=args.force_regenerate,
        mode=args.mode,
        min_onsets=args.min_onsets,
        skip_guitar_remap=args.skip_guitar_remap,
    )


if __name__ == "__main__":
    main()
