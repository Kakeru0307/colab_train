"""ストローク・アルペジオ・音階・進行パターン。"""

from __future__ import annotations

import random

import muspy

from .builder import add_chord, add_note, build_music, make_guitar_track
from .constants import (
    ATTACKS_PER_BAR_WEIGHTS,
    BAR_LENGTH_CHOICES,
    BEAT_TICKS,
    DEFAULT_BARS,
    DEFAULT_VELOCITY,
    PATTERN_WEIGHTS,
    PLACEMENT_WEIGHTS,
    STRUM_ARTICULATION_WEIGHTS,
    TICKS_PER_BAR,
)
from .progressions import (
    PROGRESSIONS,
    ProgressionSpec,
    progression_chord_pitch_sets,
)
from .rhythm import sample_attacks_for_bpm
from .voicings import chord_pitches, scale_pitches


def choose_strum_articulation(rng: random.Random) -> str:
    names = list(STRUM_ARTICULATION_WEIGHTS.keys())
    weights = [STRUM_ARTICULATION_WEIGHTS[n] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def choose_attacks_per_bar(rng: random.Random) -> int:
    ns = list(ATTACKS_PER_BAR_WEIGHTS.keys())
    weights = [ATTACKS_PER_BAR_WEIGHTS[n] for n in ns]
    return int(rng.choices(ns, weights=weights, k=1)[0])


def choose_placement(rng: random.Random) -> str:
    names = list(PLACEMENT_WEIGHTS.keys())
    weights = [PLACEMENT_WEIGHTS[n] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def resolve_onset_ticks(
    n: int,
    placement: str,
    *,
    ticks_per_bar: int = TICKS_PER_BAR,
    rng: random.Random,
) -> list[int]:
    """1小節内の onset tick（昇順・重複なし）。"""
    n = max(1, min(int(n), ticks_per_bar))
    all_ticks = list(range(ticks_per_bar))

    if placement == "sparse_random":
        return sorted(rng.sample(all_ticks, n))

    if placement == "offbeat":
        odds = [t for t in all_ticks if t % 2 == 1]
        evens = [t for t in all_ticks if t % 2 == 0]
        if n <= len(odds):
            # 均等に裏から取る
            step = len(odds) / n
            picked = [odds[min(len(odds) - 1, int(i * step))] for i in range(n)]
            # unique 化
            out: list[int] = []
            for t in picked:
                if t not in out:
                    out.append(t)
            for t in odds:
                if len(out) >= n:
                    break
                if t not in out:
                    out.append(t)
            return sorted(out[:n])
        return sorted(odds + evens[: n - len(odds)])

    if placement == "front":
        # 常に小節前半 (0..7)。N>8 は前半を埋め、余りを後半から補充
        front_pool = list(range(ticks_per_bar // 2))
        if n <= len(front_pool):
            step = len(front_pool) / n
            picked: list[int] = []
            for i in range(n):
                t = front_pool[min(len(front_pool) - 1, int(i * step))]
                if t not in picked:
                    picked.append(t)
            for t in front_pool:
                if len(picked) >= n:
                    break
                if t not in picked:
                    picked.append(t)
            return sorted(picked[:n])
        back_pool = list(range(ticks_per_bar // 2, ticks_per_bar))
        extra = n - len(front_pool)
        step = len(back_pool) / extra
        extras: list[int] = []
        for i in range(extra):
            t = back_pool[min(len(back_pool) - 1, int(i * step))]
            if t not in extras:
                extras.append(t)
        for t in back_pool:
            if len(extras) >= extra:
                break
            if t not in extras:
                extras.append(t)
        return sorted(front_pool + extras[:extra])

    if placement == "back":
        # 常に小節後半 (8..15)。N>8 は後半を埋め、余りを前半から補充
        back_pool = list(range(ticks_per_bar // 2, ticks_per_bar))
        if n <= len(back_pool):
            step = len(back_pool) / n
            picked = []
            for i in range(n):
                t = back_pool[min(len(back_pool) - 1, int(i * step))]
                if t not in picked:
                    picked.append(t)
            for t in back_pool:
                if len(picked) >= n:
                    break
                if t not in picked:
                    picked.append(t)
            return sorted(picked[:n])
        front_pool = list(range(ticks_per_bar // 2))
        extra = n - len(back_pool)
        step = len(front_pool) / extra
        extras = []
        for i in range(extra):
            t = front_pool[min(len(front_pool) - 1, int(i * step))]
            if t not in extras:
                extras.append(t)
        for t in front_pool:
            if len(extras) >= extra:
                break
            if t not in extras:
                extras.append(t)
        return sorted(extras[:extra] + back_pool)

    # even（既定）
    if n == 1:
        return [0]
    picked = []
    for i in range(n):
        t = int(round(i * (ticks_per_bar - 1) / (n - 1))) if n > 1 else 0
        t = max(0, min(ticks_per_bar - 1, t))
        if t not in picked:
            picked.append(t)
    for t in all_ticks:
        if len(picked) >= n:
            break
        if t not in picked:
            picked.append(t)
    return sorted(picked[:n])


def _strum_onset_duration(
    articulation: str,
    *,
    onset_index: int,
    onset_ticks: list[int],
    ticks_per_bar: int,
    rng: random.Random,
    rest_indices: frozenset[int],
) -> int | None:
    """onset 間ギャップに基づく duration。None は休符。"""
    n = len(onset_ticks)
    tick = onset_ticks[onset_index]
    if onset_index + 1 < n:
        gap = onset_ticks[onset_index + 1] - tick
    else:
        gap = max(1, ticks_per_bar - tick)

    full = max(gap - 1, 1)
    short = max(1, min(2, max(gap // 2, 1)))

    if articulation == "solid":
        return full
    if articulation == "staccato":
        return short
    if articulation == "mixed":
        return short if (onset_index % 2 == 1) else full
    if articulation == "sustained":
        if onset_index % 2 != 0:
            return None
        return max(full, min(ticks_per_bar - tick, max(gap * 2 - 1, full)))
    if articulation == "rests":
        if onset_index in rest_indices:
            return None
        return short if rng.random() < 0.35 else full
    return full


def _choose_rest_indices(rng: random.Random, n: int) -> frozenset[int]:
    if n <= 1:
        return frozenset()
    candidates = (
        frozenset({n - 1}),
        frozenset({i for i in range(n) if i % 2 == 1}) or frozenset({n - 1}),
        frozenset({n // 2}),
    )
    chosen = rng.choice(candidates)
    # 全部休みにしない
    if len(chosen) >= n:
        return frozenset({n - 1})
    return chosen


def _write_strum_bars(
    track: muspy.Track,
    *,
    pitch_sets: list[list[int]] | list,
    bars: int,
    bars_per_chord: int,
    onset_ticks: list[int],
    articulation: str,
    rng: random.Random,
) -> None:
    rest_indices = (
        _choose_rest_indices(rng, len(onset_ticks))
        if articulation == "rests"
        else frozenset()
    )
    for bar in range(bars):
        chord_index = (bar // bars_per_chord) % len(pitch_sets)
        pitches = pitch_sets[chord_index]
        for oi, tick in enumerate(onset_ticks):
            duration = _strum_onset_duration(
                articulation,
                onset_index=oi,
                onset_ticks=onset_ticks,
                ticks_per_bar=TICKS_PER_BAR,
                rng=rng,
                rest_indices=rest_indices,
            )
            if duration is None:
                continue
            add_chord(
                track,
                pitches,
                time=bar * TICKS_PER_BAR + tick,
                duration=duration,
            )


def generate_chord_strum(
    *,
    key: str,
    quality: str,
    bpm: int,
    bars: int = DEFAULT_BARS,
    articulation: str = "solid",
    attacks_per_bar: int | None = None,
    placement: str | None = None,
    onset_ticks: list[int] | None = None,
    rng: random.Random | None = None,
) -> muspy.Music:
    rng = rng or random.Random()
    track = make_guitar_track("chord_strum")
    pitches = chord_pitches(key, quality)
    if onset_ticks is None:
        n = attacks_per_bar if attacks_per_bar is not None else choose_attacks_per_bar(rng)
        place = placement if placement is not None else choose_placement(rng)
        onset_ticks = resolve_onset_ticks(n, place, rng=rng)
    _write_strum_bars(
        track,
        pitch_sets=[pitches],
        bars=bars,
        bars_per_chord=1,
        onset_ticks=onset_ticks,
        articulation=articulation,
        rng=rng,
    )
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
    bars_per_chord: int = 1,
    articulation: str = "solid",
    attacks_per_bar: int | None = None,
    placement: str | None = None,
    onset_ticks: list[int] | None = None,
    rng: random.Random | None = None,
) -> muspy.Music:
    """進行に沿ってストローク伴奏を生成（1曲=1ノリの N×配置）。"""
    rng = rng or random.Random()
    track = make_guitar_track(f"prog_strum_{spec.name}")
    pitch_sets = progression_chord_pitch_sets(spec, key)
    if onset_ticks is None:
        n = attacks_per_bar if attacks_per_bar is not None else choose_attacks_per_bar(rng)
        place = placement if placement is not None else choose_placement(rng)
        onset_ticks = resolve_onset_ticks(n, place, rng=rng)
    _write_strum_bars(
        track,
        pitch_sets=pitch_sets,
        bars=bars,
        bars_per_chord=bars_per_chord,
        onset_ticks=onset_ticks,
        articulation=articulation,
        rng=rng,
    )
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
    """進行のスケール上で単音リードフレーズを生成する。"""
    rng = rng or random.Random()
    track = make_guitar_track(f"prog_lead_{spec.name}")
    scale = scale_pitches(key, spec.mode, base_octave=4)
    if not scale:
        scale = scale_pitches(key, spec.mode)
    pitch_sets = progression_chord_pitch_sets(spec, key)
    chord_pc_sets = [set(p % 12 for p in ps) for ps in pitch_sets]

    step = TICKS_PER_BAR // 8
    idx = len(scale) // 2

    for bar in range(bars):
        chord_pcs = chord_pc_sets[(bar // bars_per_chord) % len(chord_pc_sets)]
        for e in range(8):
            pos = bar * TICKS_PER_BAR + e * step
            is_strong = (e * step) % BEAT_TICKS == 0
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
            attacks = sample_attacks_for_bpm(bpm_value, rng)
            placement = choose_placement(rng)
            onset_ticks = resolve_onset_ticks(attacks, placement, rng=rng)
            music = generate_progression_strum(
                spec=spec,
                key=key,
                bpm=bpm_value,
                bars=bars,
                bars_per_chord=bars_per_chord,
                articulation=articulation,
                onset_ticks=onset_ticks,
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
                "attacks_per_bar": attacks,
                "placement": placement,
                "onset_ticks": onset_ticks,
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
        pitch_sets = progression_chord_pitch_sets(spec, key)
        arp_notes = len(pitch_sets[0]) if pitch_sets else 0
        return music, {
            "pattern": pattern,
            "progression": spec.name,
            "family": spec.family,
            "mode": spec.mode,
            "key": key,
            "bars_per_chord": bars_per_chord,
            "arp_notes_per_bar": arp_notes,
            "bpm": bpm_value,
            "bars": bars,
        }

    key = rng.choice(KEYS)

    if pattern == "chord_strum":
        quality = rng.choice(CHORD_QUALITIES)
        articulation = choose_strum_articulation(rng)
        attacks = sample_attacks_for_bpm(bpm_value, rng)
        placement = choose_placement(rng)
        onset_ticks = resolve_onset_ticks(attacks, placement, rng=rng)
        music = generate_chord_strum(
            key=key,
            quality=quality,
            bpm=bpm_value,
            bars=bars,
            articulation=articulation,
            onset_ticks=onset_ticks,
            rng=rng,
        )
        meta = {
            "pattern": pattern,
            "key": key,
            "quality": quality,
            "articulation": articulation,
            "attacks_per_bar": attacks,
            "placement": placement,
            "onset_ticks": onset_ticks,
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
            "arp_notes_per_bar": len(chord_pitches(key, quality)),
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
