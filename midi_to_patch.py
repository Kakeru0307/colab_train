"""MIDI を ViTex 互換の 128x128 パッチ列に変換する。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import muspy
import numpy as np

TICKS_PER_BAR = 16
PATCH_BARS = 8
PATCH_TICKS = PATCH_BARS * TICKS_PER_BAR
NUM_CATEGORIES = 11


@dataclass
class MidiPatch:
    tonal: np.ndarray
    drum: np.ndarray
    bar_index: int

    @property
    def tonal_chw(self) -> np.ndarray:
        """(11, 128, 128) = C, time, pitch"""
        return self.tonal.transpose(2, 0, 1)

    @property
    def drum_chw(self) -> np.ndarray:
        """(1, 128, 128) = C, time, pitch"""
        return self.drum[np.newaxis, ...]


def _category(program: int) -> int:
    category = program // 8
    return min(category, NUM_CATEGORIES - 1)


def _collect_tonal_notes(music: muspy.Music) -> np.ndarray:
    notes_list: list[list[int]] = []
    for track in music.tracks:
        if track.is_drum:
            continue
        category = _category(track.program)
        for note in track:
            duration = note.duration + int(note.duration == 0)
            notes_list.append([note.time, note.pitch, category, duration])

    if not notes_list:
        return np.zeros((0, 4), dtype=np.int32)

    notes = np.array(notes_list, dtype=np.int32)
    return notes[notes[:, 0].argsort()]


def _collect_drum_notes(music: muspy.Music) -> np.ndarray:
    notes_list: list[list[int]] = []
    for track in music.tracks:
        if not track.is_drum:
            continue
        for note in track:
            duration = note.duration + int(note.duration == 0)
            notes_list.append([note.time, note.pitch, duration])

    if not notes_list:
        return np.zeros((0, 3), dtype=np.int32)

    notes = np.array(notes_list, dtype=np.int32)
    return notes[notes[:, 0].argsort()]


def _render_tonal_patch(tonal_notes: np.ndarray, bar_index: int) -> np.ndarray:
    """(128, 128, 11) = time, pitch, category"""
    pianoroll = np.zeros((PATCH_TICKS, 128, NUM_CATEGORIES), dtype=np.int32)
    start_tick = bar_index * TICKS_PER_BAR

    for onset, pitch, category, duration in tonal_notes:
        w = int(onset - start_tick)
        if w < 0 or w >= PATCH_TICKS:
            continue
        dw = int(duration - 1)
        end_w = min(w + dw, PATCH_TICKS)
        pianoroll[w, pitch, category] = 1
        if end_w > w + 1:
            pianoroll[w + 1 : end_w, pitch, category] = 2

    return pianoroll


def _render_drum_patch(drum_notes: np.ndarray, bar_index: int) -> np.ndarray:
    """(128, 128) = time, pitch"""
    pianoroll = np.zeros((PATCH_TICKS, 128), dtype=np.int32)
    start_tick = bar_index * TICKS_PER_BAR

    for onset, pitch, duration in drum_notes:
        w = int(onset - start_tick)
        if w < 0 or w >= PATCH_TICKS:
            continue
        dw = int(duration - 1)
        end_w = min(w + dw, PATCH_TICKS)
        pianoroll[w, pitch] = 1
        if end_w > w + 1:
            pianoroll[w + 1 : end_w, pitch] = 2

    return pianoroll


def midi_to_patches(midi_path: str | Path) -> list[MidiPatch]:
    music = muspy.read_midi(midi_path).adjust_resolution(4)
    tonal_notes = _collect_tonal_notes(music)
    drum_notes = _collect_drum_notes(music)

    end_time = music.get_end_time()
    bar_num = (end_time // TICKS_PER_BAR) + 1
    num_patches = max(0, bar_num - PATCH_BARS + 1)

    patches: list[MidiPatch] = []
    for bar_index in range(num_patches):
        tonal = _render_tonal_patch(tonal_notes, bar_index)
        drum = _render_drum_patch(drum_notes, bar_index)
        patches.append(MidiPatch(tonal=tonal, drum=drum, bar_index=bar_index))

    return patches


def normalize_pianoroll(pianoroll: np.ndarray) -> np.ndarray:
    """0/1/2 を 0.0/0.5/1.0 に変換。"""
    return pianoroll.astype(np.float32) / 2.0


def denormalize_pianoroll(values: np.ndarray) -> np.ndarray:
    """連続値を 0/1/2 に戻す。"""
    quantized = np.zeros_like(values, dtype=np.int32)
    quantized[values >= 0.25] = 1
    quantized[values >= 0.75] = 2
    return quantized


def save_patches(patches: list[MidiPatch], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for patch in patches:
        stem = f"bar{patch.bar_index:04d}"
        np.save(output_dir / f"{stem}_tonal.npy", patch.tonal_chw)
        np.save(output_dir / f"{stem}_drum.npy", patch.drum_chw)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    midi_path = script_dir / "midi" / "test1.mid"
    out_dir = script_dir / "data" / "patches" / "test1"

    patch_list = midi_to_patches(midi_path)
    save_patches(patch_list, out_dir)
    first = patch_list[0]
    print(f"パッチ数: {len(patch_list)}")
    print(f"tonal shape: {first.tonal_chw.shape}, drum shape: {first.drum_chw.shape}")
    print(f"保存先: {out_dir}")
