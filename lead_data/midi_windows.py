"""MIDI / note 列を 8 小節ウィンドウに切る。"""

from __future__ import annotations

import muspy

from makeData.builder import build_music, make_guitar_track
from makeData.constants import DEFAULT_BARS, DEFAULT_VELOCITY, TICKS_PER_BAR


def music_notes_ticks(music: muspy.Music) -> list[tuple[int, int, int]]:
    """(onset, duration, pitch) in muspy ticks (resolution=4 → 16 ticks/bar)."""
    notes: list[tuple[int, int, int]] = []
    for track in music.tracks:
        if track.is_drum:
            continue
        for note in track.notes:
            notes.append((int(note.time), max(1, int(note.duration)), int(note.pitch)))
    notes.sort(key=lambda x: x[0])
    return notes


def music_bpm(music: muspy.Music, default: float = 120.0) -> float:
    if music.tempos:
        # muspy Tempo.qpm
        try:
            return float(music.tempos[0].qpm)
        except Exception:
            pass
    return float(default)


def total_bars_from_notes(notes: list[tuple[int, int, int]]) -> int:
    if not notes:
        return 0
    end = max(on + dur for on, dur, _ in notes)
    return max(1, (end + TICKS_PER_BAR - 1) // TICKS_PER_BAR)


def slice_notes_window(
    notes: list[tuple[int, int, int]],
    *,
    start_bar: int,
    bars: int = DEFAULT_BARS,
) -> list[tuple[int, int, int]]:
    start = start_bar * TICKS_PER_BAR
    end = start + bars * TICKS_PER_BAR
    out: list[tuple[int, int, int]] = []
    for on, dur, pitch in notes:
        note_end = on + dur
        if note_end <= start or on >= end:
            continue
        new_on = max(on, start) - start
        new_end = min(note_end, end) - start
        if new_end > new_on:
            out.append((new_on, new_end - new_on, pitch))
    return out


def notes_to_music(
    notes: list[tuple[int, int, int]],
    *,
    bars: int,
    bpm: float,
) -> muspy.Music:
    track = make_guitar_track("LeadTarget")
    for on, dur, pitch in notes:
        track.append(
            muspy.Note(
                time=int(on),
                pitch=int(pitch),
                duration=max(1, int(dur)),
                velocity=DEFAULT_VELOCITY,
            )
        )
    return build_music(track, bars=bars, tempo=float(bpm))


def iter_bar_windows(n_bars: int, *, window: int = DEFAULT_BARS, stride: int = 4):
    if n_bars < window:
        return
    start = 0
    while start + window <= n_bars:
        yield start
        start += stride
    # 末尾を取りこぼさない（stride でアライメントされていない最後のウィンドウ）
    last = n_bars - window
    if last > 0 and last % stride != 0:
        yield last
