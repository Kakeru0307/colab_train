"""BarChord 列から lead 用の downbeat 骨格 MIDI を作る。"""

from __future__ import annotations

import muspy

from lead_data.chord_estimate import BarChord
from makeData.builder import add_chord, build_music, make_guitar_track
from makeData.constants import CHORD_INTERVALS, KEY_TO_SEMITONE, TICKS_PER_BAR


def _chord_pitches(chord: BarChord, *, octave: int = 4) -> list[int]:
    root_pc = KEY_TO_SEMITONE[chord.root]
    base = 12 * octave + root_pc
    return [min(127, base + iv) for iv in CHORD_INTERVALS.get(chord.quality, [0, 4, 7])]


def build_skeleton_from_bar_chords(
    chords: list[BarChord],
    *,
    bpm: float,
    onset_duration: int = 2,
) -> muspy.Music:
    bars = len(chords)
    if bars < 8:
        raise ValueError("骨格は 8 小節以上必要です")
    track = make_guitar_track("LeadSkeleton")
    for bar, chord in enumerate(chords):
        pitches = _chord_pitches(chord)
        add_chord(
            track,
            pitches,
            time=bar * TICKS_PER_BAR,
            duration=onset_duration,
        )
    return build_music(track, bars=bars, tempo=float(bpm))
