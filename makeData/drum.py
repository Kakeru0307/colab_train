"""ドラムパターン合成（GM キット・energy で密度変化）。"""

from __future__ import annotations

import random
from typing import Literal

import muspy

from .builder import add_note, build_music
from .constants import (
    BEAT_TICKS,
    DEFAULT_BARS,
    DEFAULT_VELOCITY,
    DRUM_CLOSED_HAT,
    DRUM_CRASH_1,
    DRUM_CRASH_2,
    DRUM_KICK,
    DRUM_OPEN_HAT,
    DRUM_PEDAL_HAT,
    DRUM_RIDE,
    DRUM_SIDE_STICK,
    DRUM_SNARE,
    DRUM_TOM_FILLS,
    TICKS_PER_BAR,
)

EnergyLevel = Literal["low", "mid", "high"]


def make_drum_track(name: str = "Drums") -> muspy.Track:
    return muspy.Track(program=0, is_drum=True, name=name)


def extract_kick_times(music: muspy.Music) -> set[int]:
    """ドラムトラックからキック onset tick の集合を返す。"""
    times: set[int] = set()
    for track in music.tracks:
        if not track.is_drum:
            continue
        for note in track.notes:
            if int(note.pitch) == DRUM_KICK:
                times.add(int(note.time))
    return times


def _hit(
    track: muspy.Track,
    *,
    time: int,
    pitch: int,
    velocity: int = DEFAULT_VELOCITY,
    duration: int = 1,
) -> None:
    add_note(track, time=time, pitch=pitch, duration=duration, velocity=velocity)


def _add_tom_fill(
    track: muspy.Track,
    *,
    bar_t: int,
    energy: EnergyLevel,
    rng: random.Random,
) -> None:
    """8小節ブロック最終小節の後半に tom fill。"""
    if energy == "low":
        return
    start = bar_t + (2 * BEAT_TICKS if energy == "mid" else BEAT_TICKS)
    toms = list(DRUM_TOM_FILLS)
    step = 2 if energy == "mid" else 1
    vel = 85 if energy == "mid" else 100
    t = start
    while t < bar_t + TICKS_PER_BAR:
        _hit(track, time=t, pitch=rng.choice(toms), velocity=vel)
        t += step


def generate_drum_pattern(
    *,
    bpm: float = 120.0,
    bars: int = DEFAULT_BARS,
    energy: EnergyLevel = "mid",
    section_start_crash: bool = True,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> muspy.Music:
    """GM キットのドラムパターンを生成する。

    energy は楽器制限ではなく密度で差をつける。
    fill は 8 小節ブロック最終小節のみ（low はなし）。
    """
    rng = rng or random.Random(seed)
    track = make_drum_track()

    for bar in range(bars):
        bar_t = bar * TICKS_PER_BAR
        is_last = bar == bars - 1
        is_first = bar == 0

        # crash: high は必ず、mid はセクション頭のみ
        if is_first and section_start_crash:
            if energy == "high":
                _hit(track, time=bar_t, pitch=DRUM_CRASH_1, velocity=110)
            elif energy == "mid" and rng.random() < 0.7:
                _hit(
                    track,
                    time=bar_t,
                    pitch=rng.choice((DRUM_CRASH_1, DRUM_CRASH_2)),
                    velocity=95,
                )

        if energy == "low":
            # 4分 closed hat、kick は1拍目、snare は 2+4（弱め）
            for beat in range(4):
                _hit(
                    track,
                    time=bar_t + beat * BEAT_TICKS,
                    pitch=DRUM_CLOSED_HAT,
                    velocity=rng.randint(55, 70),
                )
            _hit(track, time=bar_t, pitch=DRUM_KICK, velocity=rng.randint(75, 90))
            for beat in (1, 3):
                _hit(
                    track,
                    time=bar_t + beat * BEAT_TICKS,
                    pitch=DRUM_SNARE,
                    velocity=rng.randint(55, 70),
                )
            # ride も薄く使える（楽器制限なし）
            if rng.random() < 0.3:
                _hit(
                    track,
                    time=bar_t + 2 * BEAT_TICKS,
                    pitch=DRUM_RIDE,
                    velocity=60,
                )
            continue

        if energy == "mid":
            # 8分 closed hat、kick 1+3、snare 2+4
            for i in range(8):
                pitch = DRUM_CLOSED_HAT
                if i == 7 and rng.random() < 0.35:
                    pitch = DRUM_OPEN_HAT
                _hit(
                    track,
                    time=bar_t + i * 2,
                    pitch=pitch,
                    velocity=rng.randint(65, 85),
                )
            for beat in (0, 2):
                _hit(
                    track,
                    time=bar_t + beat * BEAT_TICKS,
                    pitch=DRUM_KICK,
                    velocity=rng.randint(85, 100),
                )
            for beat in (1, 3):
                _hit(
                    track,
                    time=bar_t + beat * BEAT_TICKS,
                    pitch=DRUM_SNARE,
                    velocity=rng.randint(80, 95),
                )
            if rng.random() < 0.2:
                _hit(
                    track,
                    time=bar_t + BEAT_TICKS + 2,
                    pitch=DRUM_PEDAL_HAT,
                    velocity=60,
                )
            if is_last:
                _add_tom_fill(track, bar_t=bar_t, energy=energy, rng=rng)
            continue

        # high: 8分 closed + アップビート open、kick 4つ打ち or シンコペ
        for i in range(8):
            if i % 2 == 1 and rng.random() < 0.55:
                pitch = DRUM_OPEN_HAT
                vel = rng.randint(80, 100)
            else:
                pitch = DRUM_CLOSED_HAT
                vel = rng.randint(70, 90)
            _hit(track, time=bar_t + i * 2, pitch=pitch, velocity=vel)

        if rng.random() < 0.5:
            # 4つ打ち
            for beat in range(4):
                _hit(
                    track,
                    time=bar_t + beat * BEAT_TICKS,
                    pitch=DRUM_KICK,
                    velocity=rng.randint(95, 115),
                )
        else:
            # シンコペ: 1, &, 3, 4&
            for tick in (0, 2, 2 * BEAT_TICKS, 3 * BEAT_TICKS + 2):
                _hit(
                    track,
                    time=bar_t + tick,
                    pitch=DRUM_KICK,
                    velocity=rng.randint(95, 115),
                )

        for beat in (1, 3):
            _hit(
                track,
                time=bar_t + beat * BEAT_TICKS,
                pitch=DRUM_SNARE,
                velocity=rng.randint(90, 110),
            )
            # ghost note
            if rng.random() < 0.4:
                _hit(
                    track,
                    time=bar_t + beat * BEAT_TICKS + 2,
                    pitch=rng.choice((DRUM_SNARE, DRUM_SIDE_STICK)),
                    velocity=rng.randint(40, 55),
                )

        if is_last:
            _add_tom_fill(track, bar_t=bar_t, energy=energy, rng=rng)

    return build_music(track, bars=bars, tempo=float(bpm))
