"""all_jams_midi.zip → synthtab_pick10000.tsv（機械スコア＋クォータ充足）。

採用判断そのものはしない。mono / 高音 / ペンタ系の列挙と枠埋めのみ。
境界の目視は生成後に人間が行う。

例:
  python lead_data/build_synthtab_pick10000.py \\
    --zip lead_data/all_jams_midi.zip \\
    --out lead_data/synthtab_pick10000.tsv
"""

from __future__ import annotations

import argparse
import collections
import csv
import random
import struct
import time
from pathlib import Path

KEYSWITCH_MAX = 35
TARGET_ST = 9500
QUOTA_PENTA = 2800
QUOTA_HIGH = 1800
MONO_MIN = 0.5
MIN_NOTES = 24
MOVE_MIN = 0.12  # 極端なチャグ／単音ペダル落とし
# mono≥0.5 プールでは厳しい高音が少ない。n67/pmax の緩いゲート＋上位埋めで ~15% を確保
HIGH_N67_MIN = 4
HIGH_PMAX_MIN = 70
SEED = 42

SCALES: dict[str, tuple[int, ...]] = {
    "minor_pent": (0, 3, 5, 7, 10),
    "major_pent": (0, 2, 4, 7, 9),
    "egyptian": (0, 2, 5, 7, 10),
    "dorian_pent": (0, 2, 3, 7, 9),
    "mixolydian_pent": (0, 2, 4, 7, 10),
    "phrygian_pent": (0, 1, 3, 7, 8),
    "locrian_pent": (0, 3, 5, 6, 10),
    "hirajoshi": (0, 2, 3, 7, 8),
    "insen": (0, 1, 5, 7, 10),
    "iwato": (0, 1, 5, 6, 10),
    "kumoi": (0, 2, 3, 7, 9),
    "yo": (0, 2, 5, 7, 9),
    "prometheus": (0, 2, 4, 6, 9),
    "blues_min": (0, 3, 5, 6, 7, 10),
    "blues_maj": (0, 2, 3, 4, 7, 9),
}


def _parse_track_events(data: bytes) -> list[tuple[int, int, int]]:
    """(abs_tick, on01, pitch) for one SMF."""
    if data[:4] != b"MThd":
        return []
    ev: list[tuple[int, int, int]] = []
    i = 0
    while True:
        idx = data.find(b"MTrk", i)
        if idx < 0 or idx + 8 > len(data):
            break
        length = struct.unpack(">I", data[idx + 4 : idx + 8])[0]
        tr = data[idx + 8 : idx + 8 + length]
        i = idx + 8 + length
        pos = 0
        running = None
        abs_t = 0
        while pos < len(tr):
            delta = 0
            while pos < len(tr):
                b = tr[pos]
                pos += 1
                delta = (delta << 7) | (b & 0x7F)
                if not (b & 0x80):
                    break
            abs_t += delta
            if pos >= len(tr):
                break
            status = tr[pos]
            if status & 0x80:
                running = status
                pos += 1
            else:
                status = running
                if status is None:
                    break
            stype = status & 0xF0
            if stype in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                if pos + 1 >= len(tr):
                    break
                pitch = tr[pos]
                v = tr[pos + 1]
                pos += 2
                if stype == 0x90 and v > 0:
                    ev.append((abs_t, 1, pitch))
                elif stype == 0x80 or (stype == 0x90 and v == 0):
                    ev.append((abs_t, 0, pitch))
            elif stype in (0xC0, 0xD0):
                pos += 1
            elif status == 0xFF:
                if pos + 1 >= len(tr):
                    break
                ln = tr[pos + 1]
                pos += 2 + ln
            elif status in (0xF0, 0xF7):
                ln = 0
                while pos < len(tr):
                    b = tr[pos]
                    pos += 1
                    ln = (ln << 7) | (b & 0x7F)
                    if not (b & 0x80):
                        break
                pos += ln
            else:
                break
    return ev


def _score_events(ev: list[tuple[int, int, int]]) -> dict | None:
    if not ev:
        return None
    ev = sorted(ev)
    sounding: collections.Counter[int] = collections.Counter()
    mono = poly = 0
    pitches: list[int] = []
    pc_hist: collections.Counter[int] = collections.Counter()
    for _t, on, pitch in ev:
        if pitch <= KEYSWITCH_MAX:
            if on:
                sounding[pitch] += 1
            else:
                if sounding[pitch]:
                    sounding[pitch] -= 1
                    if sounding[pitch] == 0:
                        del sounding[pitch]
            continue
        if on:
            others = sum(1 for pp, c in sounding.items() if pp > KEYSWITCH_MAX and c > 0)
            if others == 0:
                mono += 1
            else:
                poly += 1
            pitches.append(pitch)
            pc_hist[pitch % 12] += 1
            sounding[pitch] += 1
        else:
            if sounding[pitch]:
                sounding[pitch] -= 1
                if sounding[pitch] == 0:
                    del sounding[pitch]
    notes = mono + poly
    if notes < MIN_NOTES:
        return None
    mono_r = mono / notes
    # pitch motion among successive onsets
    moves = 0
    for a, b in zip(pitches, pitches[1:]):
        if a != b:
            moves += 1
    move_r = moves / max(1, len(pitches) - 1)
    pmax = max(pitches) if pitches else 0
    n77 = sum(1 for p in pitches if p >= 77)
    n67 = sum(1 for p in pitches if p >= 67)
    high = (n67 >= HIGH_N67_MIN) or (pmax >= HIGH_PMAX_MIN)

    # penta / blues family
    best_name = ""
    best_cov = -1.0
    best_n = 0
    best_t = 0
    for name, iv in SCALES.items():
        for t in range(12):
            s = {(t + x) % 12 for x in iv}
            inn = sum(v for pc, v in pc_hist.items() if pc in s)
            n_used = sum(1 for pc in s if pc_hist[pc] > 0)
            cov = inn / notes
            if cov > best_cov or (cov == best_cov and n_used > best_n):
                best_cov, best_n, best_t, best_name = cov, n_used, t, name
    n_need = 5 if len(SCALES[best_name]) >= 6 else 4
    s_best = {(best_t + x) % 12 for x in SCALES[best_name]}
    outside = sum(v for pc, v in pc_hist.items() if pc not in s_best) / notes
    penta_strict = best_cov >= 0.90 and best_n >= n_need and outside <= 0.10
    penta_weak = best_cov >= 0.85 and best_n >= n_need and outside <= 0.15

    # head for eye digest
    head = ",".join(str(p) for p in pitches[:16])
    return {
        "notes": notes,
        "mono": round(mono_r, 4),
        "move": round(move_r, 4),
        "pmax": pmax,
        "n77": n77,
        "n67": n67,
        "high": int(high),
        "penta_strict": int(penta_strict),
        "penta_weak": int(penta_weak),
        "penta_name": best_name,
        "penta_cov": round(best_cov, 4),
        "head": head,
    }


def _label_inst(track_key: str) -> tuple[str, str]:
    label = Path(track_key.replace("\\", "/")).name
    lab = label.lower()
    if "nylon" in lab:
        inst = "nylon"
    elif "steel" in lab:
        inst = "steel"
    elif "jazz" in lab:
        inst = "jazz"
    elif "clean" in lab:
        inst = "clean"
    elif "muted" in lab:
        inst = "muted"
    elif "overdriven" in lab:
        inst = "overdriven"
    elif "distortion" in lab:
        inst = "distortion"
    else:
        inst = "other"
    return label, inst


def scan_zip(zip_path: Path) -> list[dict]:
    import zipfile

    z = zipfile.ZipFile(zip_path)
    mids = [n for n in z.namelist() if n.lower().endswith((".mid", ".midi"))]
    by: dict[str, list[str]] = collections.defaultdict(list)
    for n in mids:
        by["/".join(n.split("/")[:-1])].append(n)
    parents = sorted(by.keys())
    rows: list[dict] = []
    t0 = time.time()
    for i, key in enumerate(parents):
        ev: list[tuple[int, int, int]] = []
        for fn in by[key]:
            try:
                ev.extend(_parse_track_events(z.read(fn)))
            except Exception:
                continue
        sc = _score_events(ev)
        if sc is None:
            continue
        label, inst = _label_inst(key)
        row = {
            "track_key": key.replace("/", "\\") if "\\" in key or True else key,
            "jams": "",
            "n_strings": str(len(by[key])),
            "label": label,
            "inst": inst,
            **{k: str(sc[k]) for k in (
                "notes", "mono", "move", "pmax", "n77", "n67",
                "high", "penta_strict", "penta_weak", "penta_name", "penta_cov", "head",
            )},
        }
        # normalize track_key to forward-ish consistent with existing tsv (backslashes)
        row["track_key"] = key.replace("/", "\\")
        rows.append(row)
        if (i + 1) % 2000 == 0:
            print(
                f"[scan] {i+1}/{len(parents)} kept={len(rows)} "
                f"{time.time()-t0:.0f}s",
                flush=True,
            )
    print(f"[scan] done parents={len(parents)} scored={len(rows)} {time.time()-t0:.0f}s", flush=True)
    return rows


def select_pick(scored: list[dict], *, target: int, quota_penta: int, quota_high: int) -> list[dict]:
    rng = random.Random(SEED)
    pool = []
    for r in scored:
        mono = float(r["mono"])
        move = float(r["move"])
        if mono < MONO_MIN:
            continue
        if move < MOVE_MIN and int(r["notes"]) > 80:
            # 動かない長尺はチャグ／ペダル寄りとして除外
            continue
        pool.append(r)

    def prio_penta(r: dict) -> tuple:
        return (
            -int(r["penta_strict"]),
            -int(r["penta_weak"]),
            -float(r["penta_cov"]),
            -float(r["mono"]),
            -float(r["move"]),
        )

    def prio_high(r: dict) -> tuple:
        return (-int(r["n77"]), -int(r["n67"]), -int(r["pmax"]), -float(r["mono"]))

    def prio_fill(r: dict) -> tuple:
        return (-float(r["mono"]), -float(r["move"]), -int(r["notes"]))

    chosen: dict[str, dict] = {}
    flags: dict[str, set[str]] = collections.defaultdict(set)

    penta_cands = sorted(
        [r for r in pool if int(r["penta_strict"]) or int(r["penta_weak"])],
        key=prio_penta,
    )
    for r in penta_cands:
        if sum(1 for s in flags.values() if "penta" in s) >= quota_penta:
            break
        k = r["track_key"]
        chosen[k] = r
        flags[k].add("penta")

    # 高音はフラグだけでなく n67/pmax 上位から枠を埋める（母集団上限まで）
    for r in sorted(pool, key=prio_high):
        if sum(1 for s in flags.values() if "high" in s) >= quota_high:
            break
        if int(r["n67"]) < HIGH_N67_MIN and int(r["pmax"]) < HIGH_PMAX_MIN:
            continue
        k = r["track_key"]
        chosen[k] = r
        flags[k].add("high")

    rest = [r for r in pool if r["track_key"] not in chosen]
    rest.sort(key=prio_fill)
    # slight shuffle among equal-ish tiers
    rng.shuffle(rest)
    rest.sort(key=prio_fill)
    for r in rest:
        if len(chosen) >= target:
            break
        chosen[r["track_key"]] = r
        flags[r["track_key"]].add("fill")

    out = []
    for k, r in chosen.items():
        row = dict(r)
        row["quota"] = "+".join(sorted(flags[k])) if flags[k] else "fill"
        out.append(row)
    out.sort(key=lambda r: r["track_key"])
    n_penta = sum(1 for r in out if "penta" in r["quota"])
    n_high = sum(1 for r in out if "high" in r["quota"])
    print(
        f"[select] total={len(out)} penta_flag={n_penta} high_flag={n_high} "
        f"pool={len(pool)}",
        flush=True,
    )
    return out


def write_tsv(path: Path, rows: list[dict]) -> None:
    cols = [
        "track_key", "jams", "n_strings", "label", "inst",
        "notes", "mono", "move", "pmax", "n77", "n67",
        "high", "penta_strict", "penta_weak", "penta_name", "penta_cov",
        "quota", "head",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=cols, delimiter="\t", extrasaction="ignore", restval=""
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {path} rows={len(rows)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build synthtab_pick10000.tsv")
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "synthtab_pick10000.tsv",
    )
    p.add_argument("--target", type=int, default=TARGET_ST)
    p.add_argument("--quota-penta", type=int, default=QUOTA_PENTA)
    p.add_argument("--quota-high", type=int, default=QUOTA_HIGH)
    p.add_argument("--score-cache", type=Path, default=None, help="optional scored TSV reuse")
    args = p.parse_args()
    if not args.zip.is_file():
        raise SystemExit(f"zip not found: {args.zip}")

    if args.score_cache and args.score_cache.is_file():
        with args.score_cache.open(encoding="utf-8", newline="") as f:
            scored = list(csv.DictReader(f, delimiter="\t"))
        print(f"[cache] loaded {len(scored)} from {args.score_cache}", flush=True)
    else:
        scored = scan_zip(args.zip)
        cache = args.score_cache or args.out.with_name("synthtab_score_all.tsv")
        write_tsv(cache, [{**r, "quota": ""} for r in scored])

    pick = select_pick(
        scored,
        target=args.target,
        quota_penta=args.quota_penta,
        quota_high=args.quota_high,
    )
    write_tsv(args.out, pick)


if __name__ == "__main__":
    main()
