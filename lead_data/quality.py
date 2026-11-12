"""リード教師パッチの品質判定・ノート整形。

目的:
  モデルが「発音の山」を学べるよう、べったり伸ばし／ほぼ無音／発音が
  極端に少ない窓を落とす。SynthTab・合成の共通ゲート。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from util.midi_to_patch import MidiPatch, NUM_CATEGORIES


# resolution=4 で 1 小節 = 16 tick。リード教師の最大音価（これ以上は切る）
DEFAULT_MAX_NOTE_DURATION_TICKS = 16  # 全音符相当
DEFAULT_MIN_ONSETS = 4
DEFAULT_MIN_ONSET_RATIO = 0.12  # onsetセル / 活性セル
DEFAULT_MAX_FILL = 0.85  # 時間×音高の占有率上限（べた塗り拒否）
DEFAULT_MIN_FILL = 0.001  # 下限（ほぼ無音拒否）
DEFAULT_MIN_ACTIVE_TICKS = 8  # 何か鳴っている tick 数の下限


@dataclass(frozen=True)
class LeadQualityParams:
    min_onsets: int = DEFAULT_MIN_ONSETS
    min_onset_ratio: float = DEFAULT_MIN_ONSET_RATIO
    max_fill: float = DEFAULT_MAX_FILL
    min_fill: float = DEFAULT_MIN_FILL
    min_active_ticks: int = DEFAULT_MIN_ACTIVE_TICKS


def sharpen_lead_notes(
    notes: list[tuple[int, int, int]],
    *,
    max_duration_ticks: int = DEFAULT_MAX_NOTE_DURATION_TICKS,
) -> list[tuple[int, int, int]]:
    """同ピッチの重なりを切り、長すぎる sustain を短くする。

    戻り値: (onset, duration, pitch) の昇順リスト。
    """
    if not notes:
        return []

    by_pitch: dict[int, list[tuple[int, int]]] = {}
    for onset, duration, pitch in notes:
        by_pitch.setdefault(int(pitch), []).append(
            (int(onset), max(1, int(duration)))
        )

    cleaned: list[tuple[int, int, int]] = []
    max_dur = max(1, int(max_duration_ticks))
    for pitch, events in by_pitch.items():
        events.sort(key=lambda x: x[0])
        for i, (onset, duration) in enumerate(events):
            duration = min(duration, max_dur)
            end = onset + duration
            if i + 1 < len(events):
                next_on = events[i + 1][0]
                # 次の同ピッチ発音の直前で必ず切る（1/2 境界を残す）
                if next_on <= onset:
                    continue
                end = min(end, next_on)
            duration = max(1, end - onset)
            cleaned.append((onset, duration, pitch))

    cleaned.sort(key=lambda x: (x[0], x[2]))
    return cleaned


def patch_onset_stats(patch: MidiPatch) -> tuple[int, int, float, int]:
    """(onset数, 活性セル数, 時間×音高占有率, 活性tick数) を返す。"""
    roll = np.asarray(patch.tonal_chw)
    if roll.ndim != 3:
        return 0, 0, 0.0, 0
    if roll.shape[0] == NUM_CATEGORIES:
        # CHW → 時間×音高の占有（カテゴリは潰す）
        occupied = (roll > 0).any(axis=0)
        active_ticks = int((roll > 0).any(axis=(0, 2)).sum())
    else:
        # THW
        occupied = (roll > 0).any(axis=2)
        active_ticks = int((roll > 0).any(axis=(1, 2)).sum())
    onset = int((roll == 1).sum())
    active = int((roll > 0).sum())
    fill = float(occupied.mean()) if occupied.size else 0.0
    return onset, active, fill, active_ticks


def is_clear_lead_patch(
    patch: MidiPatch,
    params: LeadQualityParams | None = None,
) -> bool:
    """発音がはっきりしたリード教師パッチか。"""
    p = params or LeadQualityParams()
    onset, active, fill, active_ticks = patch_onset_stats(patch)
    if onset < int(p.min_onsets):
        return False
    if active_ticks < int(p.min_active_ticks):
        return False
    if fill < float(p.min_fill):
        return False
    if fill > float(p.max_fill):
        return False
    if active <= 0:
        return False
    onset_ratio = float(onset) / float(active)
    if onset_ratio < float(p.min_onset_ratio):
        return False
    return True
