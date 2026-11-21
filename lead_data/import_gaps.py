"""GAPS classical guitar MIDI → lead pairs（arp 高スコア寄り選別）。

Zenodo gaps_v1 の ``*.mid`` を想定。全 240 一括はせず、単音＋分散和音スコア上位を取る。
power mask はゼロ。骨格はノートからのコード推定。
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
from lead_data.quality import DEFAULT_MIN_ONSETS, sharpen_lead_notes
from lead_data.skeleton import build_skeleton_from_bar_chords
from lead_data.write_pairs import save_lead_pair_song
from makeData.constants import DEFAULT_BARS

SONG_ID_PREFIX = "gaps_"


def _iter_midi_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.mid")) + list(root.rglob("*.midi"))
    return sorted(set(files))


def _load_music(path: Path) -> muspy.Music:
    music = muspy.read_midi(path)
    if music.resolution != 4:
        muspy.adjust_resolution(music, 4)
    return music


def arp_score_notes(notes: list) -> dict[str, float]:
    """単音率・短間隔音高変化・分散和音窓から arp 寄りスコアを返す。"""
    if len(notes) < 8:
        return {"score": 0.0, "mono": 0.0, "arp_h": 0.0, "broken": 0.0}
    by_t: dict[int, list[int]] = {}
    for n in notes:
        # music_notes_ticks → (onset, duration, pitch)
        if isinstance(n, tuple) and len(n) >= 3:
            time, _dur, pitch = int(n[0]), int(n[1]), int(n[2])
        else:
            time, pitch = int(n.time), int(n.pitch)
        by_t.setdefault(time, []).append(pitch)
    times = sorted(by_t)
    if len(times) < 8:
        return {"score": 0.0, "mono": 0.0, "arp_h": 0.0, "broken": 0.0}

    mono = sum(1 for t in times if len(by_t[t]) == 1) / len(times)
    tops = [by_t[t][0] if len(by_t[t]) == 1 else max(by_t[t]) for t in times]

    succ = max(len(times) - 1, 1)
    short_chg = 0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        if dt <= 2 and tops[i] != tops[i + 1]:
            short_chg += 1
    arp_h = short_chg / succ

    win_ok = 0
    win_n = 0
    for i in range(len(tops) - 3):
        w = tops[i : i + 4]
        win_n += 1
        if len(set(w)) >= 3 and (max(w) - min(w)) >= 7:
            win_ok += 1
    broken = win_ok / max(win_n, 1)

    score = 0.40 * mono + 0.35 * arp_h + 0.25 * broken
    return {"score": float(score), "mono": float(mono), "arp_h": float(arp_h), "broken": float(broken)}


def rank_gaps_midis(midi_root: Path) -> list[dict]:
    """全 MIDI をスコア付きで降順リスト化。"""
    rows: list[dict] = []
    for path in _iter_midi_files(midi_root):
        try:
            music = _load_music(path)
        except Exception as exc:
            rows.append(
                {
                    "path": path,
                    "score": -1.0,
                    "mono": 0.0,
                    "arp_h": 0.0,
                    "broken": 0.0,
                    "error": str(exc),
                }
            )
            continue
        notes = music_notes_ticks(music)
        m = arp_score_notes(notes)
        rows.append(
            {
                "path": path,
                "score": m["score"],
                "mono": m["mono"],
                "arp_h": m["arp_h"],
                "broken": m["broken"],
                "n_notes": len(notes),
                "bars": total_bars_from_notes(notes) if notes else 0,
            }
        )
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    return rows


def select_gaps_files(
    midi_root: Path,
    *,
    top_k: int = 100,
    min_score: float | None = None,
) -> list[dict]:
    ranked = [r for r in rank_gaps_midis(midi_root) if float(r["score"]) >= 0.0]
    if min_score is not None:
        ranked = [r for r in ranked if float(r["score"]) >= float(min_score)]
    if top_k is not None and top_k > 0:
        ranked = ranked[: int(top_k)]
    return ranked


def write_gaps_pick_tsv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["rank\tstem\tscore\tmono\tarp_h\tbroken\tbars\tn_notes"]
    for i, r in enumerate(rows, start=1):
        stem = Path(r["path"]).stem
        lines.append(
            f"{i}\t{stem}\t{r['score']:.4f}\t{r['mono']:.4f}\t{r['arp_h']:.4f}\t"
            f"{r['broken']:.4f}\t{r.get('bars', 0)}\t{r.get('n_notes', 0)}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def import_gaps(
    midi_root: Path,
    pairs_dir: Path,
    *,
    top_k: int = 100,
    min_score: float | None = None,
    pick_tsv: Path | None = None,
    window_bars: int = DEFAULT_BARS,
    stride_bars: int = 8,
    min_onsets: int = DEFAULT_MIN_ONSETS,
    min_notes: int = 4,
    min_bars: int = 8,
    sharpen: bool = True,
) -> dict:
    selected = select_gaps_files(midi_root, top_k=top_k, min_score=min_score)
    if pick_tsv is not None:
        write_gaps_pick_tsv(selected, pick_tsv)

    stats: dict = {
        "source": "gaps",
        "songs": [],
        "total_patches": 0,
        "files": 0,
        "selected": len(selected),
        "top_k": top_k,
        "min_score": min_score,
        "skipped_short": 0,
        "skipped_quality": 0,
        "song_ids": [],
    }
    pairs_dir.mkdir(parents=True, exist_ok=True)

    for path_info in selected:
        path = Path(path_info["path"])
        try:
            music = _load_music(path)
        except Exception as exc:
            print(f"[gaps] skip {path.name}: {exc}")
            continue

        bpm = music_bpm(music)
        notes = music_notes_ticks(music)
        if len(notes) < min_notes:
            continue
        n_bars = total_bars_from_notes(notes)
        if n_bars < min_bars:
            stats["skipped_short"] += 1
            continue

        tag = path.stem.replace("\\", "_").replace("/", "_")[:80]
        song_patches = 0
        for wi, start_bar in enumerate(
            iter_bar_windows(n_bars, window=window_bars, stride=stride_bars)
        ):
            win = slice_notes_window(notes, start_bar=start_bar, bars=window_bars)
            if sharpen:
                win = sharpen_lead_notes(win)
            if len(win) < min_onsets:
                stats["skipped_quality"] += 1
                continue
            key = estimate_key_from_notes(win)
            chords = estimate_chords_per_bar(win, bars=window_bars, key=key)
            try:
                target = notes_to_music(win, bars=window_bars, bpm=bpm)
                inp = build_skeleton_from_bar_chords(chords, bpm=bpm)
            except Exception as exc:
                print(f"[gaps] build fail {tag}@{start_bar}: {exc}")
                continue
            song_id = f"{SONG_ID_PREFIX}{tag}_{wi:02d}"
            n = save_lead_pair_song(
                pairs_dir,
                song_id,
                input_music=inp,
                target_music=target,
                bpm=bpm,
                min_onsets=min_onsets,
                zero_power_mask=True,
            )
            if n == 0:
                stats["skipped_quality"] += 1
            else:
                song_patches += n
                stats["song_ids"].append(song_id)

        if song_patches > 0:
            stats["files"] += 1
            stats["total_patches"] += song_patches
            stats["songs"].append(
                {
                    "id": tag,
                    "patches": song_patches,
                    "bars": n_bars,
                    "score": path_info.get("score"),
                }
            )
            if stats["files"] % 25 == 0:
                print(f"[gaps] {stats['files']} files ... patches={stats['total_patches']}")

    print(
        f"[gaps] done  selected={stats['selected']}  files={stats['files']}  "
        f"patches={stats['total_patches']}  skipped_short={stats['skipped_short']}  "
        f"skipped_quality={stats['skipped_quality']}"
    )
    return stats


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="GAPS MIDI → lead pairs（arp 上位選別）")
    p.add_argument("--midi-dir", type=Path, required=True)
    p.add_argument("--pairs-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--pick-tsv", type=Path, default=None)
    p.add_argument("--min-bars", type=int, default=8)
    args = p.parse_args()
    import_gaps(
        args.midi_dir,
        args.pairs_dir,
        top_k=args.top_k,
        min_score=args.min_score,
        pick_tsv=args.pick_tsv,
        min_bars=args.min_bars,
    )


if __name__ == "__main__":
    main()
