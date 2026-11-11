"""AnimeTAB Clips (MusicXML) → lead pairs + section ラベル。

ファイル名例: ``Song_0^8I.xml`` → bars 0–8, role I(intro)
マップ: I→intro, A→verse, B→chorus, C→bridge
"""

from __future__ import annotations

import re
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
from song_form import ANIMETAB_CLIP_TO_ROLE, section_role_to_id

# stem: ..._start^endX  (X in IABC)
_CLIP_RE = re.compile(
    r"_(?P<start>\d+)\^(?P<end>\d+)(?P<label>[IABC])$",
    re.IGNORECASE,
)


def _iter_clip_xml(root: Path) -> list[Path]:
    files = list(root.rglob("*.xml")) + list(root.rglob("*.musicxml"))
    return sorted(set(files))


def parse_animetab_clip_name(path: Path) -> tuple[int, int, str] | None:
    """(start_bar, end_bar, I/A/B/C) or None."""
    m = _CLIP_RE.search(path.stem)
    if not m:
        return None
    start = int(m.group("start"))
    end = int(m.group("end"))
    label = m.group("label").upper()
    if end <= start:
        return None
    return start, end, label


def _load_music(path: Path) -> muspy.Music:
    music = muspy.read_musicxml(str(path))
    if music.resolution != 4:
        muspy.adjust_resolution(music, 4)
    return music


def import_animetab(
    clips_dir: Path,
    pairs_dir: Path,
    *,
    max_files: int | None = None,
    window_bars: int = DEFAULT_BARS,
    stride_bars: int = 8,
    min_onsets: int = 1,
    min_notes: int = 4,
) -> dict:
    files = _iter_clip_xml(clips_dir)
    stats = {
        "source": "animetab",
        "songs": [],
        "total_patches": 0,
        "files": 0,
        "skipped_parse": 0,
        "by_role": {k: 0 for k in ANIMETAB_CLIP_TO_ROLE},
    }
    pairs_dir.mkdir(parents=True, exist_ok=True)

    used = 0
    for path in files:
        if max_files is not None and used >= max_files:
            break
        parsed = parse_animetab_clip_name(path)
        if parsed is None:
            stats["skipped_parse"] += 1
            continue
        _start, _end, clip_label = parsed
        role = ANIMETAB_CLIP_TO_ROLE.get(clip_label)
        if role is None:
            stats["skipped_parse"] += 1
            continue
        section_id = section_role_to_id(role)

        try:
            music = _load_music(path)
        except Exception as exc:
            print(f"[animetab] skip {path.name}: {exc}")
            continue

        bpm = music_bpm(music)
        notes = music_notes_ticks(music)
        if len(notes) < min_notes:
            continue
        n_bars = max(1, total_bars_from_notes(notes))

        try:
            rel = path.relative_to(clips_dir)
        except ValueError:
            rel = Path(path.stem)
        tag = str(rel.with_suffix("")).replace("\\", "_").replace("/", "_")[:80]

        song_patches = 0
        for wi, start_bar in enumerate(
            iter_bar_windows(n_bars, window=window_bars, stride=stride_bars)
        ):
            win = slice_notes_window(notes, start_bar=start_bar, bars=window_bars)
            if len(win) < min_onsets:
                continue
            key = estimate_key_from_notes(win)
            chords = estimate_chords_per_bar(win, bars=window_bars, key=key)
            try:
                target = notes_to_music(win, bars=window_bars, bpm=bpm)
                inp = build_skeleton_from_bar_chords(chords, bpm=bpm)
            except Exception as exc:
                print(f"[animetab] build fail {tag}@{start_bar}: {exc}")
                continue
            song_id = f"atab_{tag}_{wi:02d}"
            n = save_lead_pair_song(
                pairs_dir,
                song_id,
                input_music=inp,
                target_music=target,
                bpm=bpm,
                min_onsets=min_onsets,
                zero_power_mask=True,
                section_id=section_id,
            )
            song_patches += n

        if song_patches > 0:
            used += 1
            stats["files"] += 1
            stats["total_patches"] += song_patches
            stats["by_role"][clip_label] = stats["by_role"].get(clip_label, 0) + 1
            stats["songs"].append(
                {
                    "id": tag,
                    "patches": song_patches,
                    "bars": n_bars,
                    "role": role,
                    "clip": clip_label,
                }
            )
            if used % 50 == 0:
                print(f"[animetab] {used} files ... patches={stats['total_patches']}")

    print(
        f"[animetab] done  files={stats['files']}  patches={stats['total_patches']}  "
        f"skipped_parse={stats['skipped_parse']}  by_role={stats['by_role']}"
    )
    return stats


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="AnimeTAB Clips → lead pairs")
    p.add_argument("--clips-dir", type=Path, required=True)
    p.add_argument("--pairs-dir", type=Path, required=True)
    p.add_argument("--max-files", type=int, default=None)
    args = p.parse_args()
    import_animetab(args.clips_dir, args.pairs_dir, max_files=args.max_files)


if __name__ == "__main__":
    main()
