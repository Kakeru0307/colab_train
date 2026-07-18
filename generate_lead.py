"""進行指定からリードギター MIDI（単音）を生成する。

backing と対称:
  進行名 + キー + BPM
    → 骨格 MIDI（小節頭コードトーン、backing と共通）
    → U-Net（checkpoints/lead）
    → 単音リード MIDI（1 トラック）
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
    GUITAR_DISTORTION_PROGRAM,
    remap_tonal_program,
)
from makeData.progressions import resolve_progression_chords

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "lead" / "unet_last.pt"


def _apply_tempo(music: muspy.Music, bpm: float) -> muspy.Music:
    music.tempos = [muspy.Tempo(time=0, qpm=float(bpm))]
    return music


def _make_monophonic(music: muspy.Music, program: int) -> muspy.Music:
    """各発音時刻で最高音だけを残し、単音のリードにする。"""
    for track in music.tracks:
        if track.is_drum:
            continue
        best: dict[int, muspy.Note] = {}
        for note in track.notes:
            cur = best.get(note.time)
            if cur is None or note.pitch > cur.pitch:
                best[note.time] = note
        track.notes = sorted(best.values(), key=lambda n: (n.time, n.pitch))
        track.name = "Lead"
        track.program = program
    return music


def generate_lead(
    *,
    progression: str,
    key: str,
    bars: int = 8,
    bpm: float = 120.0,
    bars_per_chord: int = 1,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    output: Path | None = None,
    save_skeleton: bool = True,
    guitar_program: int = GUITAR_DISTORTION_PROGRAM,
    identity: bool = False,
) -> Path:
    """進行からリード MIDI を生成して保存する。"""
    spec = get_progression(progression)
    chords = resolve_progression_chords(spec, key)
    chord_label = "-".join(
        f"{root}{'' if quality == 'maj' else quality}" for root, quality in chords
    )

    stem = f"lead_{progression}_{key}_bpm{int(bpm):03d}_{bars}bars"
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

    with tempfile.TemporaryDirectory() as tmp:
        tmp_midi = Path(tmp) / "skeleton.mid"
        muspy.write_midi(tmp_midi, skeleton)
        patches = midi_to_patches(tmp_midi)
        if not patches:
            raise RuntimeError(f"パッチが 0 件です。bars={bars} を 8 以上にしてください。")

        if identity:
            output_patches = patches
        else:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model(checkpoint, device)
            output_patches = predict_patches(
                model,
                patches,
                device,
                input_mode="downbeat_chord",
                guitar_only=True,
            )

    music = patches_to_music(output_patches)
    music = export_guitar_music(
        remap_tonal_program(music, guitar_program),
        program=guitar_program,
        include_drums=False,
    )
    music = _make_monophonic(music, guitar_program)
    music = _apply_tempo(music, bpm)
    save_music(music, output_path)

    note_count = sum(len(t.notes) for t in music.tracks)
    print(f"進行: {progression} ({spec.family}, mode={spec.mode})")
    print(f"キー: {key} / 例: {chord_label}")
    print(f"BPM: {bpm}, bars: {bars}, bars_per_chord: {bars_per_chord}")
    if save_skeleton:
        print(f"骨格: {skeleton_path}")
    print(f"リード: {output_path}")
    print(f"ノート数(単音化後): {note_count}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="コード進行からリードギター MIDI（単音）を生成する")
    parser.add_argument("--list", action="store_true", help="利用可能な進行名を一覧表示して終了")
    parser.add_argument("--progression", type=str, default="marusa", help="進行名（--list で一覧）")
    parser.add_argument("--key", type=str, default="C")
    parser.add_argument("--bars", type=int, default=8, help="小節数（8 以上）")
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--bars-per-chord", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-skeleton", action="store_true", help="骨格 MIDI を保存しない")
    parser.add_argument("--guitar-program", type=int, default=GUITAR_DISTORTION_PROGRAM)
    parser.add_argument("--identity", action="store_true", help="モデルを使わず骨格をそのまま出力（動作確認用）")
    args = parser.parse_args()

    if args.list:
        print(format_progression_catalog())
        return

    generate_lead(
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
