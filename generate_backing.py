"""進行指定からバッキングギター MIDI を生成する。

流れ:
  進行名 + キー + BPM
    → 骨格 MIDI（小節頭コードトーン）
    → U-Net（input は骨格をそのまま）
    → バッキング MIDI（1 トラック）
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import muspy

from inference import (
    MIDI_DIR,
    export_guitar_music,
    load_model,
    predict_patches,
)
from midi_to_patch import midi_to_patches
from patch_to_midi import patches_to_music, save_music
from progression_input import (
    build_backing_skeleton_music,
    format_progression_catalog,
    get_progression,
)
from program_utils import (
    GUITAR_OVERDRIVE_PROGRAM,
    remap_tonal_program,
)
from makeData.progressions import resolve_progression_chords

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "backing" / "unet_last.pt"


def _apply_tempo(music: muspy.Music, bpm: float) -> muspy.Music:
    music.tempos = [muspy.Tempo(time=0, qpm=float(bpm))]
    return music


def _name_backing_track(music: muspy.Music, program: int) -> muspy.Music:
    for track in music.tracks:
        if not track.is_drum:
            track.name = "Backing"
            track.program = program
    return music


def generate_backing(
    *,
    progression: str,
    key: str,
    bars: int = 8,
    bpm: float = 120.0,
    bars_per_chord: int = 1,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    output: Path | None = None,
    save_skeleton: bool = True,
    guitar_program: int = GUITAR_OVERDRIVE_PROGRAM,
    identity: bool = False,
) -> Path:
    """進行からバッキング MIDI を生成して保存する。"""
    spec = get_progression(progression)
    chords = resolve_progression_chords(spec, key)
    chord_label = "-".join(
        f"{root}{'' if quality == 'maj' else quality}" for root, quality in chords
    )

    stem = f"backing_{progression}_{key}_bpm{int(bpm):03d}_{bars}bars"
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output or (MIDI_DIR / f"{stem}.mid")
    skeleton_path = MIDI_DIR / f"{stem}_skeleton.mid"

    skeleton = build_backing_skeleton_music(
        progression=spec,
        key=key,
        bars=bars,
        bpm=bpm,
        bars_per_chord=bars_per_chord,
    )
    if save_skeleton:
        save_music(skeleton, skeleton_path)

    # 一時ファイル経由でパッチ化（既存パイプラインを再利用）
    with tempfile.TemporaryDirectory() as tmp:
        tmp_midi = Path(tmp) / "skeleton.mid"
        muspy.write_midi(tmp_midi, skeleton)
        patches = midi_to_patches(tmp_midi)

        if not patches:
            raise RuntimeError(
                f"パッチが 0 件です。bars={bars} を 8 以上にしてください。"
            )

        if identity:
            output_patches = patches
        else:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model(checkpoint, device)
            # 入力 MIDI 自体が骨格なので full（再抽出で薄めない）
            # 学習時 downbeat_chord と同等のスパースさ
            output_patches = predict_patches(
                model,
                patches,
                device,
                input_mode="full",
                guitar_only=True,
            )

    music = patches_to_music(output_patches)
    music = export_guitar_music(
        remap_tonal_program(music, guitar_program),
        program=guitar_program,
        include_drums=False,
    )
    music = _name_backing_track(music, guitar_program)
    music = _apply_tempo(music, bpm)
    save_music(music, output_path)

    note_count = sum(len(t.notes) for t in music.tracks)
    print(f"進行: {progression} ({spec.family}, mode={spec.mode})")
    print(f"キー: {key} / 例: {chord_label}")
    print(f"BPM: {bpm}, bars: {bars}, bars_per_chord: {bars_per_chord}")
    if save_skeleton:
        print(f"骨格: {skeleton_path}")
    print(f"バッキング: {output_path}")
    print(f"ノート数: {note_count}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="コード進行からバッキングギター MIDI を生成する",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="利用可能な進行名を一覧表示して終了",
    )
    parser.add_argument(
        "--progression",
        type=str,
        default="marusa",
        help="進行名（--list で一覧）",
    )
    parser.add_argument("--key", type=str, default="C", help="キー（例: C, Am は A + minor 進行）")
    parser.add_argument("--bars", type=int, default=8, help="小節数（8 以上）")
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument(
        "--bars-per-chord",
        type=int,
        default=1,
        help="1 コードを何小節伸ばすか",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-skeleton",
        action="store_true",
        help="骨格 MIDI を midi/ に保存しない",
    )
    parser.add_argument(
        "--guitar-program",
        type=int,
        default=GUITAR_OVERDRIVE_PROGRAM,
    )
    parser.add_argument(
        "--identity",
        action="store_true",
        help="モデルを使わず骨格をそのまま出力（動作確認用）",
    )
    args = parser.parse_args()

    if args.list:
        print(format_progression_catalog())
        return

    # 短調進行はキーを短調トニックで渡す（例: minor_komuro + A）
    generate_backing(
        progression=args.progression,
        key=args.key,
        bars=args.bars,
        bpm=args.bpm,
        bars_per_chord=args.bars_per_chord,
        checkpoint=args.checkpoint,
        output=args.output,
        save_skeleton=not args.no_skeleton,
        guitar_program=args.guitar_program,
        identity=args.identity,
    )


if __name__ == "__main__":
    main()
