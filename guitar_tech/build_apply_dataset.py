"""Build apply dataset: EG-Solo + Guitar-TECHS + friend MIDI + sparse synthetic fill."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .carriers import generate_carrier, notes_to_midi
from .constants import (
    APPLY_DATASET_DIR,
    DEFAULT_PROGRAM,
    EGSOLO_ROOT,
    GUITAR_TECHS_ROOT,
    TECH_ROOT,
)
from .import_egsolo_tech import import_egsolo
from .import_friend_guitar import import_friend_guitar
from .import_guitar_techs import import_guitar_techs
from .params import TECH_NAMES
from .render import assign_techs, labels_to_jsonable, render_target


def _append_synthetic(
    out_dir: Path,
    *,
    n_carriers: int,
    applies_per_carrier: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    carrier_dir = out_dir / "carrier"
    target_dir = out_dir / "target"
    labels_dir = out_dir / "labels"
    for d in (carrier_dir, target_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    tech_note_counts: Counter[str] = Counter()
    n_notes_total = 0
    n_pairs = 0
    mf_path = out_dir / "manifest_synth.jsonl"
    with mf_path.open("w", encoding="utf-8") as mf:
        for c_i in range(n_carriers):
            notes, _ = generate_carrier(rng, program=DEFAULT_PROGRAM)
            carrier_name = f"synth_c{c_i:05d}.mid"
            notes_to_midi(notes, program=DEFAULT_PROGRAM).write(
                str(carrier_dir / carrier_name)
            )
            for a_i in range(applies_per_carrier):
                pair_id = f"synth_{c_i:05d}_{a_i:02d}"
                labels = assign_techs(notes, rng)
                target = render_target(notes, labels, program=DEFAULT_PROGRAM, rng=rng)
                target.write(str(target_dir / f"{pair_id}.mid"))
                for lb in labels:
                    tech_note_counts[lb.tech] += 1
                    n_notes_total += 1
                (labels_dir / f"{pair_id}.json").write_text(
                    json.dumps(
                        {
                            "id": pair_id,
                            "source": "synth",
                            "notes": [
                                {
                                    "pitch": n.pitch,
                                    "start": round(n.start, 5),
                                    "end": round(n.end, 5),
                                    "velocity": n.velocity,
                                }
                                for n in notes
                            ],
                            "labels": labels_to_jsonable(labels),
                            "tech_names": list(TECH_NAMES),
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                mf.write(
                    json.dumps(
                        {
                            "id": pair_id,
                            "carrier": f"carrier/{carrier_name}",
                            "labels": f"labels/{pair_id}.json",
                            "source": "synth",
                            "n_notes": len(notes),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_pairs += 1
    return {
        "n_pairs": n_pairs,
        "n_note_labels": n_notes_total,
        "tech_note_counts": dict(tech_note_counts),
    }


def build(
    *,
    out_dir: Path = APPLY_DATASET_DIR,
    n_synth_carriers: int = 200,
    applies_per_carrier: int = 2,
    seed: int = 0,
    skip_egsolo: bool = False,
    skip_techs: bool = False,
    skip_friend: bool = False,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}

    if not skip_egsolo and (EGSOLO_ROOT / "tech_midi").is_dir():
        try:
            summaries["egsolo"] = import_egsolo(egsolo_root=EGSOLO_ROOT, out_dir=out_dir)
            print("egsolo", summaries["egsolo"])
        except Exception as exc:
            print(f"egsolo import skipped: {exc}")
    if not skip_techs and GUITAR_TECHS_ROOT.is_dir():
        try:
            summaries["guitar_techs"] = import_guitar_techs(
                root=GUITAR_TECHS_ROOT, out_dir=out_dir, max_per_tech=30
            )
            print("techs", summaries["guitar_techs"])
        except Exception as exc:
            print(f"techs import skipped: {exc}")
    if not skip_friend and TECH_ROOT.is_dir():
        try:
            summaries["friend"] = import_friend_guitar(root=TECH_ROOT, out_dir=out_dir)
            print("friend", summaries["friend"])
        except Exception as exc:
            print(f"friend import skipped: {exc}")

    summaries["synth"] = _append_synthetic(
        out_dir,
        n_carriers=n_synth_carriers,
        applies_per_carrier=applies_per_carrier,
        seed=seed,
    )
    print("synth", summaries["synth"])

    # aggregate
    total_notes = 0
    tech_counts: Counter[str] = Counter()
    n_pairs = 0
    for s in summaries.values():
        n_pairs += int(s.get("n_pairs", 0))
        total_notes += int(s.get("n_note_labels", 0))
        tech_counts.update(s.get("tech_note_counts") or {})
    normal_n = tech_counts.get("normal", 0)
    summary = {
        "n_pairs": n_pairs,
        "n_note_labels": total_notes,
        "normal_ratio": (normal_n / total_notes) if total_notes else 0.0,
        "tech_note_counts": dict(tech_counts),
        "sources": summaries,
        "seed": seed,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"wrote {out_dir}: pairs={n_pairs} normal_ratio={summary['normal_ratio']:.2f}"
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build guitar tech apply dataset")
    parser.add_argument("--out-dir", type=Path, default=APPLY_DATASET_DIR)
    parser.add_argument("--n-synth-carriers", type=int, default=200)
    parser.add_argument("--applies-per-carrier", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-egsolo", action="store_true")
    parser.add_argument("--skip-techs", action="store_true")
    parser.add_argument("--skip-friend", action="store_true")
    args = parser.parse_args()
    build(
        out_dir=args.out_dir,
        n_synth_carriers=args.n_synth_carriers,
        applies_per_carrier=args.applies_per_carrier,
        seed=args.seed,
        skip_egsolo=args.skip_egsolo,
        skip_techs=args.skip_techs,
        skip_friend=args.skip_friend,
    )


if __name__ == "__main__":
    main()
