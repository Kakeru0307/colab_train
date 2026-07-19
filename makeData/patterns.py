"""ストローク・アルペジオ・音階・進行パターン。"""

from __future__ import annotations

import random

import muspy

from .builder import add_chord, add_note, build_music, make_guitar_track
from .constants import (
    BAR_LENGTH_CHOICES,
    BEAT_TICKS,
    DEFAULT_BARS,
    DEFAULT_VELOCITY,
    PATTERN_WEIGHTS,
    STRUM_ARTICULATION_WEIGHTS,
    TICKS_PER_BAR,
)
from .progressions import (
    PROGRESSIONS,
    ProgressionSpec,
    progression_chord_pitch_sets,
)
from .voicings import chord_pitches, scale_pitches


def choose_strum_articulation(rng: random.Random) -> str:
    names = list(STRUM_ARTICULATION_WEIGHTS.keys())
    weights = [STRUM_ARTICULATION_WEIGHTS[n] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def _strum_beat_duration(
    articulation: str,
    *,
    beat: int,
    beats_per_bar: int,
    step: int,
    rng: random.Random,
    rest_beats: frozenset[int],
) -> int | None:
    """ストローク1拍の duration。None は休符（鳴らさない）。"""
    full = max(step - 1, 1)
    short = max(1, min(2, step // 2))
    half = max(1, step // 2)

    if articulation == "solid":
        return full
    if articulation == "staccato":
        return short
    if articulation == "mixed":
        return short if (beat % 2 == 1) else full
    if articulation == "sustained":
        # 2 拍に 1 回、長めに伸ばす（バラード寄り）
        if beat % 2 != 0:
            return None
        return max(2 * step - 1, full)
    if articulation == "rests":
        if beat in rest_beats:
            return None
        return short if rng.random() < 0.35 else full
    return full


def _choose_rest_beats(rng: random.Random, beats_per_bar: int) -> frozenset[int]:
    """1曲で固定する休符拍（少なくとも1拍は鳴らす）。"""
    if beats_per_bar <= 1:
        return frozenset()
    # よくある型: 4拍目休み / 2と4休み / 3拍目休み
    candidates = (
        frozenset({beats_per_bar - 1}),
        frozenset({1, 3}) if beats_per_bar >= 4 else frozenset({1}),
        frozenset({2}) if beats_per_bar >= 3 else frozenset({1}),
    )
    return rng.choice(candidates)


def generate_chord_strum(
    *,
    key: str,
    quality: str,
    bpm: int,
    bars: int = DEFAULT_BARS,
    beats_per_bar: int = 4,
    articulation: str = "solid",
    rng: random.Random | None = None,
) -> muspy.Music:
    rng = rng or random.Random()
    track = make_guitar_track("chord_strum")
    pitches = chord_pitches(key, quality)
    step = TICKS_PER_BAR // beats_per_bar
    rest_beats = (
        _choose_rest_beats(rng, beats_per_bar) if articulation == "rests" else frozenset()
    )

    for bar in range(bars):
        for beat in range(beats_per_bar):
            duration = _strum_beat_duration(
                articulation,
                beat=beat,
                beats_per_bar=beats_per_bar,
                step=step,
                rng=rng,
                rest_beats=rest_beats,
            )
            if duration is None:
                continue
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


def generate_progression_strum(
    *,
    spec: ProgressionSpec,
    key: str,
    bpm: int,
    bars: int = DEFAULT_BARS,
    beats_per_bar: int = 4,
    bars_per_chord: int = 1,
    articulation: str = "solid",
    rng: random.Random | None = None,
) -> muspy.Music:
    """進行に沿って、コードごとにストローク伴奏を生成する。

    articulation（1曲=1ノリ）:
      solid / staccato / mixed / sustained / rests
    """
    rng = rng or random.Random()
    track = make_guitar_track(f"prog_strum_{spec.name}")
    pitch_sets = progression_chord_pitch_sets(spec, key)
    step = TICKS_PER_BAR // beats_per_bar
    rest_beats = (
        _choose_rest_beats(rng, beats_per_bar) if articulation == "rests" else frozenset()
    )

    for bar in range(bars):
        chord_index = (bar // bars_per_chord) % len(pitch_sets)
        pitches = pitch_sets[chord_index]
        for beat in range(beats_per_bar):
            duration = _strum_beat_duration(
                articulation,
                beat=beat,
                beats_per_bar=beats_per_bar,
                step=step,
                rng=rng,
                rest_beats=rest_beats,
            )
            if duration is None:
                continue
            time = bar * TICKS_PER_BAR + beat * step
            add_chord(track, pitches, time=time, duration=duration)

    return build_music(track, bars=bars, tempo=float(bpm))


def generate_progression_arpeggio(
    *,
    spec: ProgressionSpec,
    key: str,
    bpm: int,
    bars: int = DEFAULT_BARS,
    bars_per_chord: int = 1,
) -> muspy.Music:
    """進行に沿って、コードごとにアルペジオ伴奏を生成する。"""
    track = make_guitar_track(f"prog_arp_{spec.name}")
    pitch_sets = progression_chord_pitch_sets(spec, key)

    for bar in range(bars):
        chord_index = (bar // bars_per_chord) % len(pitch_sets)
        pitches = pitch_sets[chord_index]
        notes_per_bar = len(pitches)
        step = TICKS_PER_BAR // max(notes_per_bar, 1)
        duration = max(step - 1, 2)
        for index, pitch in enumerate(pitches):
            time = bar * TICKS_PER_BAR + index * step
            add_note(track, time=time, pitch=pitch, duration=duration)

    return build_music(track, bars=bars, tempo=float(bpm))


def generate_progression_lead(
    *,
    spec: ProgressionSpec,
    key: str,
    bpm: int,
    bars: int = DEFAULT_BARS,
    bars_per_chord: int = 1,
    rng: random.Random | None = None,
    rest_prob: float = 0.18,
    leap_prob: float = 0.15,
) -> muspy.Music:
    """進行のスケール上で単音リードフレーズを生成する。

    - 拍頭はコードトーンに着地（和声感）
    - 裏拍はスケール音で順次進行（時々跳躍）
    - 休符で息継ぎ（フレーズ感）
    monophonic（各時刻 1 音）。
    """
    rng = rng or random.Random()
    track = make_guitar_track(f"prog_lead_{spec.name}")
    scale = scale_pitches(key, spec.mode, base_octave=4)  # リードは高めの音域
    if not scale:
        scale = scale_pitches(key, spec.mode)
    pitch_sets = progression_chord_pitch_sets(spec, key)
    chord_pc_sets = [set(p % 12 for p in ps) for ps in pitch_sets]

    step = TICKS_PER_BAR // 8  # 8 分刻み
    idx = len(scale) // 2  # 中央から開始

    for bar in range(bars):
        chord_pcs = chord_pc_sets[(bar // bars_per_chord) % len(chord_pc_sets)]
        for e in range(8):
            pos = bar * TICKS_PER_BAR + e * step
            is_strong = (e * step) % BEAT_TICKS == 0  # 拍頭
            if not is_strong and rng.random() < rest_prob:
                continue
            if is_strong:
                cand = [i for i in range(len(scale)) if scale[i] % 12 in chord_pcs]
                if cand:
                    idx = min(cand, key=lambda i: abs(i - idx))
            else:
                span = rng.randint(2, 4) if rng.random() < leap_prob else 1
                idx = max(0, min(len(scale) - 1, idx + rng.choice((-1, 1)) * span))
            add_note(
                track,
                time=pos,
                pitch=scale[idx],
                duration=step,
                velocity=DEFAULT_VELOCITY,
            )

    return build_music(track, bars=bars, tempo=float(bpm))


def choose_pattern(rng: random.Random) -> str:
    patterns = list(PATTERN_WEIGHTS.keys())
    weights = [PATTERN_WEIGHTS[p] for p in patterns]
    return rng.choices(patterns, weights=weights, k=1)[0]


def choose_bars(rng: random.Random, *, fixed_bars: int | None = None) -> int:
    if fixed_bars is not None:
        return fixed_bars
    return rng.choice(BAR_LENGTH_CHOICES)


def choose_progression(rng: random.Random) -> ProgressionSpec:
    weights = [spec.weight for spec in PROGRESSIONS]
    return rng.choices(list(PROGRESSIONS), weights=weights, k=1)[0]


def choose_key_for_progression(rng: random.Random, spec: ProgressionSpec) -> str:
    from .constants import KEYS

    return rng.choice(KEYS)


def generate_random_phrase(
    *,
    rng: random.Random,
    bpm: int | None = None,
    bars: int | None = None,
) -> tuple[muspy.Music, dict]:
    from .constants import BPM_RANGE, CHORD_QUALITIES, KEYS, SCALE_MODES

    pattern = choose_pattern(rng)
    bars = choose_bars(rng, fixed_bars=bars)
    bpm_value = bpm if bpm is not None else rng.randint(*BPM_RANGE)

    if pattern in ("progression_strum", "progression_arpeggio"):
        spec = choose_progression(rng)
        key = choose_key_for_progression(rng, spec)
        bars_per_chord = rng.choice((1, 1, 1, 2))
        if pattern == "progression_strum":
            articulation = choose_strum_articulation(rng)
            music = generate_progression_strum(
                spec=spec,
                key=key,
                bpm=bpm_value,
                bars=bars,
                bars_per_chord=bars_per_chord,
                articulation=articulation,
                rng=rng,
            )
            return music, {
                "pattern": pattern,
                "progression": spec.name,
                "family": spec.family,
                "mode": spec.mode,
                "key": key,
                "bars_per_chord": bars_per_chord,
                "articulation": articulation,
                "bpm": bpm_value,
                "bars": bars,
            }
        music = generate_progression_arpeggio(
            spec=spec,
            key=key,
            bpm=bpm_value,
            bars=bars,
            bars_per_chord=bars_per_chord,
        )
        return music, {
            "pattern": pattern,
            "progression": spec.name,
            "family": spec.family,
            "mode": spec.mode,
            "key": key,
            "bars_per_chord": bars_per_chord,
            "bpm": bpm_value,
            "bars": bars,
        }

    key = rng.choice(KEYS)

    if pattern == "chord_strum":
        quality = rng.choice(CHORD_QUALITIES)
        articulation = choose_strum_articulation(rng)
        music = generate_chord_strum(
            key=key,
            quality=quality,
            bpm=bpm_value,
            bars=bars,
            articulation=articulation,
            rng=rng,
        )
        meta = {
            "pattern": pattern,
            "key": key,
            "quality": quality,
            "articulation": articulation,
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
