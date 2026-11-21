"""SynthTab pick TSV + all_jams_midi.zip → 統合 MIDI ディレクトリ.

弦別 MIDI を1トラックにマージし、keyswitch（pitch<=35）を除去する。
出力は import_synthtab / build_lead_pairs_mixed が読める平坦な *.mid。

例:
  python lead_data/materialize_synthtab_pick.py \\
    --zip /content/all_jams_midi.zip \\
    --pick lead_data/synthtab_pick5000.tsv \\
    --out /content/synthtab_pick5000
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import zipfile
from pathlib import Path

import pretty_midi as pm

KEYSWITCH_MAX = 35
DEFAULT_PROGRAM = 30  # Distortion Guitar


def _safe_stem(label: str, index: int) -> str:
    stem = re.sub(r"[^\w\-]+", "_", label, flags=re.UNICODE).strip("_")
    stem = stem[:100] or f"track_{index:04d}"
    return f"{index:04d}_{stem}"


def _program_from_label(label: str) -> int:
    lab = label.lower()
    if "nylon" in lab:
        return 24
    if "steel" in lab:
        return 25
    if "jazz" in lab:
        return 26
    if "clean" in lab:
        return 27
    if "muted" in lab:
        return 28
    if "overdriven" in lab:
        return 29
    if "distortion" in lab:
        return 30
    return DEFAULT_PROGRAM


def _load_pick(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise SystemExit(f"empty pick: {path}")
    if "track_key" not in rows[0]:
        raise SystemExit(f"need track_key column in {path}")
    return rows


def _string_members(z: zipfile.ZipFile, track_key: str) -> list[str]:
    key = track_key.replace("\\", "/").rstrip("/") + "/"
    out = [
        n
        for n in z.namelist()
        if n.startswith(key) and re.search(r"string_[1-6]\.mid$", n, re.I)
    ]
    return sorted(out)


def merge_track(z: zipfile.ZipFile, track_key: str, label: str) -> pm.PrettyMIDI | None:
    members = _string_members(z, track_key)
    if not members:
        return None
    notes: list[pm.Note] = []
    for name in members:
        m = pm.PrettyMIDI(io.BytesIO(z.read(name)))
        for inst in m.instruments:
            for n in inst.notes:
                if int(n.pitch) > KEYSWITCH_MAX:
                    notes.append(n)
    if len(notes) < 8:
        return None
    out = pm.PrettyMIDI()
    inst = pm.Instrument(program=_program_from_label(label), is_drum=False, name="guitar")
    for n in notes:
        inst.notes.append(
            pm.Note(
                velocity=int(max(1, min(127, n.velocity))),
                pitch=int(n.pitch),
                start=float(n.start),
                end=float(max(n.end, n.start + 0.02)),
            )
        )
    out.instruments.append(inst)
    return out


def materialize(
    zip_path: Path,
    pick_path: Path,
    out_dir: Path,
    *,
    limit: int | None = None,
) -> dict:
    rows = _load_pick(pick_path)
    if limit is not None:
        rows = rows[: max(0, limit)]
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"written": 0, "skipped": 0, "fail": 0, "out_dir": str(out_dir)}
    with zipfile.ZipFile(zip_path) as z:
        for i, row in enumerate(rows, start=1):
            key = row["track_key"]
            label = row.get("label") or Path(key).name
            try:
                midi = merge_track(z, key, label)
            except Exception as exc:
                stats["fail"] += 1
                print(f"[fail] {label[:60]}: {exc}")
                continue
            if midi is None:
                stats["skipped"] += 1
                continue
            path = out_dir / f"{_safe_stem(label, i)}.mid"
            midi.write(str(path))
            stats["written"] += 1
            if stats["written"] % 250 == 0:
                print(f"[materialize] {stats['written']} ...", flush=True)
    print(
        f"[materialize] done written={stats['written']} "
        f"skipped={stats['skipped']} fail={stats['fail']} -> {out_dir}"
    )
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Materialize SynthTab pick to flat MIDI dir")
    p.add_argument("--zip", type=Path, required=True, help="all_jams_midi.zip")
    p.add_argument(
        "--pick",
        type=Path,
        default=Path(__file__).resolve().parent / "synthtab_pick5000.tsv",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="smoke: first N only")
    args = p.parse_args()
    if not args.zip.is_file():
        raise SystemExit(f"zip not found: {args.zip}")
    if not args.pick.is_file():
        raise SystemExit(f"pick not found: {args.pick}")
    materialize(args.zip, args.pick, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
