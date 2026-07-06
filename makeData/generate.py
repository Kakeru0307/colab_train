"""合成テスト MIDI の一括生成 CLI。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import muspy

from .constants import DEFAULT_BARS
from .patterns import generate_random_phrase

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent


def build_filename(index: int, meta: dict) -> str:
    pattern = meta["pattern"]
    key = meta["key"]
    bpm = meta["bpm"]
    if pattern in ("chord_strum", "arpeggio"):
        suffix = meta["quality"]
    else:
        suffix = meta.get("mode", "scale")
    return f"synth_{pattern}_{key}_{suffix}_bpm{bpm:03d}_{index:04d}.mid"


def generate_dataset(
    output_dir: Path,
    *,
    count: int,
    seed: int = 42,
    bars: int = DEFAULT_BARS,
    manifest_name: str = "manifest.json",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    entries: list[dict] = []

    for index in range(count):
        music, meta = generate_random_phrase(rng=rng, bars=bars)
        filename = build_filename(index, meta)
        midi_path = output_dir / filename
        muspy.write_midi(midi_path, music)
        entries.append(
            {
                "file": filename,
                "path": str(midi_path),
                **meta,
            }
        )

    manifest = {
        "count": count,
        "seed": seed,
        "bars": bars,
        "output_dir": str(output_dir),
        "entries": entries,
    }
    manifest_path = output_dir / manifest_name
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合成ギター MIDI（コード・音階・ストローク）を生成",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "data" / "raw" / "synthetic",
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bars", type=int, default=DEFAULT_BARS)
    args = parser.parse_args()

    manifest = generate_dataset(
        args.output_dir,
        count=args.count,
        seed=args.seed,
        bars=args.bars,
    )
    print(f"生成: {manifest['count']} MIDI")
    print(f"保存先: {args.output_dir}")
    print(f"manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
