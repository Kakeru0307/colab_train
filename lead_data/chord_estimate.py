"""ノート列から小節ごとのコード（root + quality）を推定する。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from makeData.constants import CHORD_INTERVALS, KEYS, KEY_TO_SEMITONE, TICKS_PER_BAR

# lead pair 骨格で使う quality に限定
_EST_QUALITIES = ("maj", "min", "7", "m7")


@dataclass(frozen=True)
class BarChord:
    root: str  # KEYS の要素
    quality: str


def _pitch_class(pitch: int) -> int:
    return int(pitch) % 12


def estimate_key_from_notes(notes: list[tuple[int, int, int]]) -> str:
    """notes: (onset_tick, duration_tick, pitch). 最頻ピッチクラスをキー候補にする。"""
    if not notes:
        return "C"
    weights = np.zeros(12, dtype=np.float64)
    for _on, dur, pitch in notes:
        weights[_pitch_class(pitch)] += max(1, int(dur))
    return KEYS[int(np.argmax(weights))]


def estimate_chords_per_bar(
    notes: list[tuple[int, int, int]],
    *,
    bars: int,
    key: str | None = None,
) -> list[BarChord]:
    """各小節のノートから triad/7th を当てる。ノートが無い小節は前のコードを引き継ぐ。"""
    key = key or estimate_key_from_notes(notes)
    key_pc = KEY_TO_SEMITONE[key]
    templates: list[tuple[str, set[int]]] = []
    for root_name in KEYS:
        root_pc = KEY_TO_SEMITONE[root_name]
        for qual in _EST_QUALITIES:
            pcs = {(_pitch_class(root_pc + iv)) for iv in CHORD_INTERVALS[qual]}
            templates.append((f"{root_name}:{qual}", pcs))

    out: list[BarChord] = []
    prev = BarChord(root=key, quality="maj")
    for bar in range(bars):
        start = bar * TICKS_PER_BAR
        end = start + TICKS_PER_BAR
        chroma = np.zeros(12, dtype=np.float64)
        for on, dur, pitch in notes:
            note_end = on + max(1, dur)
            if note_end <= start or on >= end:
                continue
            overlap = min(note_end, end) - max(on, start)
            if overlap > 0:
                chroma[_pitch_class(pitch)] += float(overlap)
        if chroma.sum() <= 0:
            out.append(prev)
            continue
        chroma = chroma / chroma.sum()
        best_name = f"{key}:maj"
        best_score = -1.0
        for name, pcs in templates:
            score = float(sum(chroma[pc] for pc in pcs))
            # キー内コードをわずかに優遇
            root_pc = KEY_TO_SEMITONE[name.split(":")[0]]
            if (root_pc - key_pc) % 12 in (0, 2, 4, 5, 7, 9, 11):
                score += 0.05
            if score > best_score:
                best_score = score
                best_name = name
        root, quality = best_name.split(":")
        prev = BarChord(root=root, quality=quality)
        out.append(prev)
    return out
