"""makeData 共通定数。"""

from __future__ import annotations

DEFAULT_RESOLUTION = 4
TICKS_PER_BAR = 16
BEAT_TICKS = 4
DEFAULT_BARS = 8
DEFAULT_VELOCITY = 80

KEYS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

KEY_TO_SEMITONE = {
    "C": 0,
    "Db": 1,
    "D": 2,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "Gb": 6,
    "G": 7,
    "Ab": 8,
    "A": 9,
    "Bb": 10,
    "B": 11,
}

CHORD_QUALITIES = ("maj", "min", "7", "m7")

CHORD_INTERVALS: dict[str, list[int]] = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "m7": [0, 3, 7, 10],
}

SCALE_MODES = ("major", "natural_minor")

SCALE_INTERVALS: dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
}

PATTERN_TYPES = ("chord_strum", "arpeggio", "scale_up", "scale_down")

BPM_RANGE = (60, 150)

# ギターらしい音域（MIDI note number）
GUITAR_PITCH_MIN = 40
GUITAR_PITCH_MAX = 76
