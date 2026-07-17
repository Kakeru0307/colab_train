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

# 旧: 単一コード／スケール往復。新規生成の主戦力は progression_*
PATTERN_TYPES = (
    "progression_strum",
    "progression_arpeggio",
    "chord_strum",
    "arpeggio",
    "scale_up",
    "scale_down",
)

# 進行を主、単一コード／スケールはごく少数（比較・多様性用）
PATTERN_WEIGHTS: dict[str, float] = {
    "progression_strum": 0.48,
    "progression_arpeggio": 0.42,
    "chord_strum": 0.04,
    "arpeggio": 0.03,
    "scale_up": 0.015,
    "scale_down": 0.015,
}

# Stage 1 目標（8 小節 MIDI ≒ 1 パッチ）※ Colab ディスク対策で長尺は使わない
DEFAULT_SYNTHETIC_COUNT = 6000

BPM_RANGE = (60, 150)

# 既定は 8 小節のみ（16/24/32 は 1 MIDI から多数パッチが出てディスクを圧迫する）
BAR_LENGTH_CHOICES = (8,)

# ギターらしい音域（MIDI note number）
GUITAR_PITCH_MIN = 40
GUITAR_PITCH_MAX = 76
