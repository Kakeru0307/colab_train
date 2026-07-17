"""MIDI -> パッチ -> U-Net -> MIDI の推論パイプライン。"""

from __future__ import annotations

import argparse
from pathlib import Path

import muspy
import numpy as np
import torch

from midi_to_patch import (
    MidiPatch,
    constrain_guitar_from_input_onsets,
    denormalize_pianoroll_inference,
    filter_pitch_range_chw,
    midi_to_patches,
    normalize_pianoroll,
)
from model import build_unet
from patch_to_midi import patches_to_music, save_music
from skeleton import PAIR_MODES, make_input_tonal
from program_utils import (
    GUITAR_OVERDRIVE_PROGRAM,
    GUITAR_PITCH_MAX,
    GUITAR_PITCH_MIN,
    GUITAR_PROGRAM,
    guitar_track_name,
    remap_tonal_program,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MIDI_DIR = SCRIPT_DIR / "midi"
GUITAR_CATEGORY = GUITAR_PROGRAM // 8


def generated_midi_path(input_midi: Path) -> Path:
    """入力曲名に対応した生成 MIDI を midi/ 配下へ保存する。"""
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    return MIDI_DIR / f"{input_midi.stem}_generated.mid"


def deduplicate_notes(notes: list[muspy.Note]) -> list[muspy.Note]:
    """スライディングウィンドウ由来の重複ノートを除去する。"""
    by_key: dict[tuple[int, int], muspy.Note] = {}
    for note in notes:
        key = (note.time, note.pitch)
        existing = by_key.get(key)
        if existing is None or note.duration > existing.duration:
            by_key[key] = note
    return sorted(by_key.values(), key=lambda note: (note.time, note.pitch))


def export_guitar_music(
    music: muspy.Music,
    *,
    program: int = GUITAR_PROGRAM,
    include_drums: bool = False,
) -> muspy.Music:
    """DAW 向けに非ドラムを 1 本のギタートラックにまとめる。"""
    tonal_notes: list[muspy.Note] = []
    drum_tracks: list[muspy.Track] = []
    for track in music.tracks:
        if track.is_drum:
            drum_tracks.append(track)
        else:
            tonal_notes.extend(track)

    guitar_track = muspy.Track(
        program=program,
        is_drum=False,
        name=guitar_track_name(program),
    )
    guitar_track.extend(deduplicate_notes(tonal_notes))

    tracks = [guitar_track]
    if include_drums:
        tracks.extend(drum_tracks)

    return muspy.Music(
        resolution=music.resolution,
        tracks=tracks,
        tempos=getattr(music, "tempos", None) or [],
    )


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_unet()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def postprocess_guitar_tonal(
    raw_chw: np.ndarray,
    input_chw: np.ndarray,
) -> np.ndarray:
    """ギター層のみ・入力オンセット準拠・ギター音域に制限する。"""
    constrained = constrain_guitar_from_input_onsets(
        raw_chw,
        input_chw,
        GUITAR_CATEGORY,
    )
    guitar_only = np.zeros_like(constrained)
    guitar_only[GUITAR_CATEGORY] = constrained[GUITAR_CATEGORY]
    return filter_pitch_range_chw(
        guitar_only,
        pitch_min=GUITAR_PITCH_MIN,
        pitch_max=GUITAR_PITCH_MAX,
    )


def postprocess_guitar_free(output_chw: np.ndarray) -> np.ndarray:
    """骨格モード用: 新規発音を許可し、ギター層・音域だけ整える。"""
    guitar_only = np.zeros_like(output_chw)
    guitar_only[GUITAR_CATEGORY] = output_chw[GUITAR_CATEGORY]
    return filter_pitch_range_chw(
        guitar_only,
        pitch_min=GUITAR_PITCH_MIN,
        pitch_max=GUITAR_PITCH_MAX,
    )


def predict_patches(
    model: torch.nn.Module,
    patches: list[MidiPatch],
    device: torch.device,
    *,
    input_mode: str = "onset_to_full",
    guitar_only: bool = True,
) -> list[MidiPatch]:
    predicted: list[MidiPatch] = []
    with torch.no_grad():
        for patch in patches:
            source_tonal = patch.tonal_chw
            if input_mode == "full":
                model_input = source_tonal
            else:
                model_input = make_input_tonal(source_tonal, input_mode)

            tensor = torch.from_numpy(normalize_pianoroll(model_input)).unsqueeze(0).to(device)
            output = model(tensor).squeeze(0).cpu().numpy()

            if guitar_only:
                masked = np.zeros_like(output)
                masked[GUITAR_CATEGORY] = output[GUITAR_CATEGORY]
                output = masked

            if guitar_only and input_mode == "onset_to_full":
                output_chw = postprocess_guitar_tonal(output, source_tonal)
            else:
                output_chw = denormalize_pianoroll_inference(output)
                if guitar_only:
                    output_chw = postprocess_guitar_free(output_chw)

            predicted.append(
                MidiPatch(
                    tonal=output_chw.transpose(1, 2, 0),
                    drum=np.zeros_like(patch.drum),
                    bar_index=patch.bar_index,
                )
            )
    return predicted


def run_inference(
    midi_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    identity: bool = False,
    input_mode: str = "onset_to_full",
    guitar_only: bool = True,
    guitar_program: int = GUITAR_OVERDRIVE_PROGRAM,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patches = midi_to_patches(midi_path)

    if identity:
        output_patches = patches
    else:
        model = load_model(checkpoint_path, device)
        output_patches = predict_patches(
            model, patches, device, input_mode=input_mode, guitar_only=guitar_only
        )

    music = patches_to_music(output_patches)
    if guitar_only:
        music = export_guitar_music(
            remap_tonal_program(music, guitar_program),
            program=guitar_program,
            include_drums=False,
        )
    save_music(music, output_path)

    note_count = sum(len(track.notes) for track in music.tracks)
    print(f"推論結果を保存しました: {output_path}")
    print(f"ノート数: {note_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--midi",
        type=Path,
        default=SCRIPT_DIR / "midi" / "test1.mid",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "guitar-techs" / "unet_last.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="未指定時は midi/<入力名>_generated.mid",
    )
    parser.add_argument(
        "--input-mode",
        choices=PAIR_MODES,
        default="onset_to_full",
        help="onset_to_full=発音点固定, downbeat_chord/root_per_bar/melody_line=骨格→音追加",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="ギター層以外のチャンネルも出力に含める",
    )
    parser.add_argument(
        "--identity",
        action="store_true",
        help="モデルを使わずパッチをそのまま MIDI に戻す",
    )
    parser.add_argument(
        "--guitar-program",
        type=int,
        default=GUITAR_OVERDRIVE_PROGRAM,
        help="DAW 出力の GM program（29=オーバードライブ, 27=クリーン, 30=ディストーション）",
    )
    args = parser.parse_args()

    output_path = args.output or generated_midi_path(args.midi)

    run_inference(
        args.midi,
        args.checkpoint,
        output_path,
        identity=args.identity,
        input_mode=args.input_mode,
        guitar_only=not args.all_categories,
        guitar_program=args.guitar_program,
    )


if __name__ == "__main__":
    main()
