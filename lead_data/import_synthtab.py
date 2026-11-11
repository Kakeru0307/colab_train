"""SynthTab JAMS / MIDI → lead pairs.

想定:
  <root>/**/*.mid  または  **/*.midi
  （all_jams_midi の MIDI 側。伴奏っぽいトラックも残してよい）
骨格はノートからコード推定。power mask はゼロ。

選別方針:
  - 32 小節以上のトラックのみ使用（曲の流れを持つ演奏を優先）
  - lick / intro / exercise 系の短い練習フレーズを除外できる
  - 60,000 トラック全体に対しても適用可能
"""

from __future__ import annotations

from pathlib import Path

import muspy

from lead_data.chord_estimate import estimate_chords_per_bar, estimate_key_from_notes
from lead_data.midi_windows import (
    iter_bar_windows,
    music_bpm,
    music_notes_ticks,
    notes_to_music,
    slice_notes_window,
    total_bars_from_notes,
)
from lead_data.skeleton import build_skeleton_from_bar_chords
from lead_data.write_pairs import save_lead_pair_song
from makeData.constants import DEFAULT_BARS

# ファイル名にこれらが含まれる場合は除外（短い練習フレーズ・曲の一部）
_DEFAULT_EXCLUDE_KEYWORDS: tuple[str, ...] = ("lick", "intro", "exercise", "lesson")


def _iter_midi_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.mid")) + list(root.rglob("*.midi"))
    return sorted(set(files))


def _load_music(path: Path) -> muspy.Music:
    music = muspy.read_midi(path)
    # muspy.read_midi は MIDI ファイルの native resolution を保持する（480 / 960 等）。
    # downstream は resolution=4 (16 ticks/bar) を前提にしているので正規化する。
    if music.resolution != 4:
        muspy.adjust_resolution(music, 4)
    return music


def _is_excluded(path: Path, exclude_keywords: tuple[str, ...]) -> bool:
    """ファイル名（stem）に除外キーワードが含まれるか判定する。大小文字不問。"""
    stem_lower = path.stem.lower()
    return any(kw in stem_lower for kw in exclude_keywords)


def import_synthtab(
    midi_root: Path,
    pairs_dir: Path,
    *,
    max_files: int | None = None,
    window_bars: int = DEFAULT_BARS,
    stride_bars: int = 8,
    min_onsets: int = 1,
    min_notes: int = 4,
    min_bars: int = 32,
    exclude_keywords: tuple[str, ...] = _DEFAULT_EXCLUDE_KEYWORDS,
) -> dict:
    """曲の流れを持つ長尺トラックのみ使用する。

    min_bars: この小節数未満のトラックはスキップ（デフォルト 32）。
    exclude_keywords: ファイル名にこれらが含まれる場合は除外。
    空タプル () を渡すとキーワードフィルタを無効にできる。
    """
    files = _iter_midi_files(midi_root)
    stats = {
        "source": "synthtab",
        "songs": [],
        "total_patches": 0,
        "files": 0,
        "skipped_short": 0,
        "skipped_keyword": 0,
    }
    pairs_dir.mkdir(parents=True, exist_ok=True)

    used = 0
    for path in files:
        if max_files is not None and used >= max_files:
            break

        # キーワード除外
        if exclude_keywords and _is_excluded(path, exclude_keywords):
            stats["skipped_keyword"] += 1
            continue

        try:
            music = _load_music(path)
        except Exception as exc:
            print(f"[synthtab] skip {path.name}: {exc}")
            continue

        bpm = music_bpm(music)
        notes = music_notes_ticks(music)
        if len(notes) < min_notes:
            continue

        n_bars = total_bars_from_notes(notes)

        # 短尺除外（min_bars 未満はスキップ）
        if n_bars < min_bars:
            stats["skipped_short"] += 1
            continue

        try:
            rel = path.relative_to(midi_root)
        except ValueError:
            rel = Path(path.stem)
        tag = str(rel.with_suffix("")).replace("\\", "_").replace("/", "_")[:80]

        song_patches = 0
        for wi, start_bar in enumerate(iter_bar_windows(n_bars, window=window_bars, stride=stride_bars)):
            win = slice_notes_window(notes, start_bar=start_bar, bars=window_bars)
            if len(win) < min_onsets:
                continue
            key = estimate_key_from_notes(win)
            chords = estimate_chords_per_bar(win, bars=window_bars, key=key)
            try:
                target = notes_to_music(win, bars=window_bars, bpm=bpm)
                inp = build_skeleton_from_bar_chords(chords, bpm=bpm)
            except Exception as exc:
                print(f"[synthtab] build fail {tag}@{start_bar}: {exc}")
                continue
            song_id = f"stab_{tag}_{wi:02d}"
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
            stats["songs"].append({"id": tag, "patches": song_patches, "bars": n_bars})
            if used % 50 == 0:
                print(f"[synthtab] {used} files ... patches={stats['total_patches']}")

    print(
        f"[synthtab] done  files={stats['files']}  patches={stats['total_patches']}  "
        f"skipped_short={stats['skipped_short']}  skipped_keyword={stats['skipped_keyword']}"
    )
    return stats
