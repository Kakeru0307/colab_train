"""GuitarSet annotation (JAMS) → lead pairs.

想定レイアウト:
  <root>/**/*.jams   （annotation.zip 展開など）
solo を優先し、chord 注釈があれば骨格に使う。無ければノート推定。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from lead_data.chord_estimate import BarChord, estimate_chords_per_bar, estimate_key_from_notes
from lead_data.midi_windows import (
    iter_bar_windows,
    notes_to_music,
    slice_notes_window,
    total_bars_from_notes,
)
from lead_data.skeleton import build_skeleton_from_bar_chords
from lead_data.write_pairs import save_lead_pair_song
from makeData.constants import DEFAULT_BARS, KEYS, KEY_TO_SEMITONE, TICKS_PER_BAR

# GuitarSet chord like "B:maj", "G:min7", "N", "X"
_CHORD_RE = re.compile(
    r"^([A-G](?:b|#)?)\s*:\s*(maj|min|major|minor|7|maj7|min7|m7|dim|aug)?",
    re.I,
)


def _norm_root(token: str) -> str | None:
    t = token.strip().replace("#", "s")  # temporary
    # map sharps to flats used in KEYS
    sharp_to_flat = {
        "C#": "Db",
        "D#": "Eb",
        "F#": "Gb",
        "G#": "Ab",
        "A#": "Bb",
        "Cs": "Db",
        "Ds": "Eb",
        "Fs": "Gb",
        "Gs": "Ab",
        "As": "Bb",
    }
    raw = token.strip()
    if raw in KEY_TO_SEMITONE:
        return raw
    if raw in sharp_to_flat:
        return sharp_to_flat[raw]
    # try without accidental variants
    for k in KEYS:
        if k.lower() == raw.lower():
            return k
    return None


def _parse_chord_label(label: str) -> BarChord | None:
    s = (label or "").strip()
    if not s or s.upper() in {"N", "X", "NONE"}:
        return None
    m = _CHORD_RE.match(s.replace(" ", ""))
    if not m:
        # bare root
        root = _norm_root(s.split(":")[0])
        if root:
            return BarChord(root=root, quality="maj")
        return None
    root = _norm_root(m.group(1))
    if not root:
        return None
    q = (m.group(2) or "maj").lower()
    if q in ("major", "maj"):
        quality = "maj"
    elif q in ("minor", "min"):
        quality = "min"
    elif q in ("m7", "min7"):
        quality = "m7"
    elif q in ("7", "maj7"):
        quality = "7" if q == "7" else "maj"
    else:
        quality = "maj"
    return BarChord(root=root, quality=quality)


def _load_jams(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jams_tempo(jams: dict, default: float = 120.0) -> float:
    for ann in jams.get("annotations", []):
        ns = ann.get("namespace", "")
        if "tempo" in ns:
            for d in ann.get("data", []):
                try:
                    return float(d.get("value", default))
                except (TypeError, ValueError):
                    continue
    return default


def _jams_notes_with_bpm(jams: dict, bpm: float, resolution: int = 4) -> list[tuple[int, int, int]]:
    """time(sec) → ticks: beat = 60/bpm sec, ticks_per_beat = resolution."""
    ticks_per_sec = (bpm / 60.0) * resolution
    notes: list[tuple[int, int, int]] = []
    for ann in jams.get("annotations", []):
        ns = str(ann.get("namespace", ""))
        if "note_midi" not in ns:
            continue
        for d in ann.get("data", []):
            try:
                pitch = int(round(float(d["value"])))
                t0 = float(d["time"])
                dur = float(d["duration"])
            except (KeyError, TypeError, ValueError):
                continue
            on = int(round(t0 * ticks_per_sec))
            nd = max(1, int(round(dur * ticks_per_sec)))
            notes.append((on, nd, pitch))
    notes.sort(key=lambda x: x[0])
    return notes


def _jams_chords_per_bar(jams: dict, *, bars: int, bpm: float, resolution: int = 4) -> list[BarChord] | None:
    ticks_per_sec = (bpm / 60.0) * resolution
    events: list[tuple[int, BarChord]] = []
    for ann in jams.get("annotations", []):
        ns = str(ann.get("namespace", ""))
        if "chord" not in ns:
            continue
        for d in ann.get("data", []):
            ch = _parse_chord_label(str(d.get("value", "")))
            if ch is None:
                continue
            t0 = float(d.get("time", 0.0))
            on = int(round(t0 * ticks_per_sec))
            events.append((on, ch))
    if not events:
        return None
    events.sort(key=lambda x: x[0])
    out: list[BarChord] = []
    for bar in range(bars):
        start = bar * TICKS_PER_BAR
        # last chord that started at or before bar start
        cur = events[0][1]
        for on, ch in events:
            if on <= start:
                cur = ch
            else:
                break
        out.append(cur)
    return out


def import_guitarset(
    annotation_root: Path,
    pairs_dir: Path,
    *,
    solo_only: bool = True,
    max_songs: int | None = None,
    window_bars: int = DEFAULT_BARS,
    stride_bars: int = 4,
    min_onsets: int = 1,
) -> dict:
    jams_files = sorted(annotation_root.rglob("*.jams"))
    if solo_only:
        jams_files = [p for p in jams_files if "solo" in p.stem.lower()]
    stats = {"source": "guitarset", "songs": [], "total_patches": 0, "files": 0}
    pairs_dir.mkdir(parents=True, exist_ok=True)

    used = 0
    for path in jams_files:
        if max_songs is not None and used >= max_songs:
            break
        try:
            jams = _load_jams(path)
        except Exception as exc:
            print(f"[guitarset] skip {path.name}: {exc}")
            continue
        bpm = _jams_tempo(jams)
        notes = _jams_notes_with_bpm(jams, bpm)
        if len(notes) < min_onsets:
            continue
        n_bars = total_bars_from_notes(notes)
        if n_bars < window_bars:
            continue
        stem = path.stem
        # コード注釈は song 単位で一度だけ取得する（ループ内で毎回呼ばない）
        full_chords = _jams_chords_per_bar(jams, bars=n_bars, bpm=bpm)
        song_patches = 0
        for wi, start_bar in enumerate(iter_bar_windows(n_bars, window=window_bars, stride=stride_bars)):
            win = slice_notes_window(notes, start_bar=start_bar, bars=window_bars)
            if len(win) < min_onsets:
                continue
            key = estimate_key_from_notes(win)
            if full_chords is not None:
                chords = full_chords[start_bar : start_bar + window_bars]
                if len(chords) < window_bars:
                    chords = estimate_chords_per_bar(win, bars=window_bars, key=key)
            else:
                chords = estimate_chords_per_bar(win, bars=window_bars, key=key)
            try:
                target = notes_to_music(win, bars=window_bars, bpm=bpm)
                inp = build_skeleton_from_bar_chords(chords, bpm=bpm)
            except Exception as exc:
                print(f"[guitarset] build fail {stem}@{start_bar}: {exc}")
                continue
            song_id = f"gset_{stem}_{wi:02d}"
            n = save_lead_pair_song(
                pairs_dir,
                song_id,
                input_music=inp,
                target_music=target,
                bpm=bpm,
                min_onsets=min_onsets,
                zero_power_mask=True,
            )
            song_patches += n
        if song_patches > 0:
            used += 1
            stats["files"] += 1
            stats["total_patches"] += song_patches
            stats["songs"].append({"id": stem, "patches": song_patches})
            print(f"[guitarset] {stem}: {song_patches} patches")

    return stats
