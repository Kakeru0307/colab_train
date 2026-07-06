"""ストローク・アルペジオ・音階パターン。"""

from __future__ import annotations

import random

import muspy

from .builder import add_chord, add_note, build_music, make_guitar_track
from .constants import BEAT_TICKS, DEFAULT_BARS, TICKS_PER_BAR
from .voicings import chord_pitches, scale_pitches


def generate_chord_strum(
    *,
    key: str,
    quality: str,
    bpm: int,
    bars: int = DEFAULT_BARS,
    beats_per_bar: int = 4,
) -> muspy.Music:
    track = make_guitar_track("chord_strum")
    pitches = chord_pitches(key, quality)
    step = TICKS_PER_BAR // beats_per_bar
    duration = max(step - 1, BEAT_TICKS - 1)

    for bar in range(bars):
        for beat in range(beats_per_bar):
            time = bar * TICKS_PER_BAR + beat * step
            add_chord(track, pitches, time=time, duration=duration)

    return build_music(track, bars=bars, tempo=float(bpm))


def generate_arpeggio(
    *,
    key: str,
    quality: str,
    bpm: int,
    bars: int = DEFAULT_BARS,
) -> muspy.Music:
    track = make_guitar_track("arpeggio")
    pitches = chord_pitches(key, quality)
    notes_per_bar = len(pitches)
    step = TICKS_PER_BAR // max(notes_per_bar, 1)
    duration = max(step - 1, 2)

    for bar in range(bars):
        for index, pitch in enumerate(pitches):
            time = bar * TICKS_PER_BAR + index * step
            add_note(track, time=time, pitch=pitch, duration=duration)

    return build_music(track, bars=bars, tempo=float(bpm))


def generate_scale(
    *,
    key: str,
    mode: str,
    bpm: int,
    direction: str,
    bars: int = DEFAULT_BARS,
) -> muspy.Music:
    track = make_guitar_track(f"scale_{direction}")
    pitches = scale_pitches(key, mode)
    if direction == "scale_down":
        pitches = list(reversed(pitches))

    step = BEAT_TICKS
    duration = max(step - 1, 2)
    time = 0
    pitch_index = 0

    while time < bars * TICKS_PER_BAR:
        pitch = pitches[pitch_index % len(pitches)]
        add_note(track, time=time, pitch=pitch, duration=duration)
        pitch_index += 1
        time += step

    return build_music(track, bars=bars, tempo=float(bpm))


def generate_random_phrase(
    *,
    rng: random.Random,
    bpm: int | None = None,
    bars: int = DEFAULT_BARS,
) -> tuple[muspy.Music, dict]:
    from .constants import BPM_RANGE, CHORD_QUALITIES, KEYS, PATTERN_TYPES, SCALE_MODES

    pattern = rng.choice(PATTERN_TYPES)
    key = rng.choice(KEYS)
    bpm_value = bpm if bpm is not None else rng.randint(*BPM_RANGE)

    if pattern == "chord_strum":
        quality = rng.choice(CHORD_QUALITIES)
        music = generate_chord_strum(
            key=key, quality=quality, bpm=bpm_value, bars=bars
        )
        meta = {
            "pattern": pattern,
            "key": key,
            "quality": quality,
            "bpm": bpm_value,
            "bars": bars,
        }
    elif pattern == "arpeggio":
        quality = rng.choice(CHORD_QUALITIES)
        music = generate_arpeggio(key=key, quality=quality, bpm=bpm_value, bars=bars)
        meta = {
            "pattern": pattern,
            "key": key,
            "quality": quality,
            "bpm": bpm_value,
            "bars": bars,
        }
    else:
        mode = rng.choice(SCALE_MODES)
        music = generate_scale(
            key=key,
            mode=mode,
            bpm=bpm_value,
            direction=pattern,
            bars=bars,
        )
        meta = {
            "pattern": pattern,
            "key": key,
            "mode": mode,
            "bpm": bpm_value,
            "bars": bars,
        }

    return music, meta
