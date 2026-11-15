"""Import commissioned friend guitar MIDI (folder name → tech)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pretty_midi as pm

from .carriers import NoteEvent
from .constants import TECH_ROOT
from .eligibility import apply_eligibility
from .features import pretty_midi_to_notes
from .import_egsolo_tech import vel_scales_from_notes
from .params import TECH_NAMES
from .render import NoteLabel, labels_to_jsonable, render_target

# Japanese folder names from 依頼仕様
FOLDER_TO_TECH: dict[str, str] = {
    "スライド": "slide",
    "ハンマリング": "hammer",
    "プリング": "pull",
    "パームミュート": "mute",
    "ゴーストノート": "ghost",
    "トリル": "vibrato",
    "トレモロピッキング": "tremolo",
    "タッピング": "tap",
    "ナチュラルハーモニクス": "harmonic",
    "浅いチョーキング": "bend",
    "チョーキング": "bend",
    "強いチョーキング": "choke_strong",
    "プリベンド": "prebend",
    "ベンドリリース": "bend_release",
    "ビブラート": "vibrato",
    "アーミング": "arming",
}
# skip: ピンチハーモニクス, スタッカート
SKIP_FOLDERS = frozenset({"ピンチハーモニクス", "スタッカート"})


def _pb_curve_params(src: pm.PrettyMIDI) -> dict:
    """Store raw pitch-bend points for later sampling / stats."""
    pts: list[tuple[float, int]] = []
    for inst in src.instruments:
        for b in inst.pitch_bends:
            pts.append((float(b.time), int(b.pitch)))
    pts.sort()
    if len(pts) > 64:
        step = max(1, len(pts) // 64)
        pts = pts[::step]
    return {"pb_curve": pts} if pts else {}


def import_friend_guitar(
    *,
    root: Path = TECH_ROOT,
    out_dir: Path,
    write_targets: bool = True,
) -> dict:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    out_dir = Path(out_dir)
    carrier_dir = out_dir / "carrier"
    target_dir = out_dir / "target"
    labels_dir = out_dir / "labels"
    for d in (carrier_dir, target_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    tech_counts: Counter[str] = Counter()
    n_notes = 0
    written = 0
    manifest = out_dir / "manifest_friend.jsonl"

    with manifest.open("w", encoding="utf-8") as mf:
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            name = folder.name
            if name in SKIP_FOLDERS:
                continue
            tech = FOLDER_TO_TECH.get(name)
            if tech is None:
                continue
            for mid in sorted(folder.glob("*.mid")):
                src = pm.PrettyMIDI(str(mid))
                notes = pretty_midi_to_notes(src)
                if not notes:
                    continue
                techs = [tech] * len(notes)
                scales = vel_scales_from_notes(notes, ["normal"] * len(notes))
                # for friend clips all notes share tech; use clip mean as ref
                mean_v = sum(n.velocity for n in notes) / len(notes)
                scales = [
                    max(0.15, min(1.5, float(n.velocity) / max(1.0, mean_v)))
                    for n in notes
                ]
                pb = _pb_curve_params(src)
                labels = [
                    NoteLabel(i, tech, {"vel_scale": scales[i], **pb})
                    for i in range(len(notes))
                ]
                apply_eligibility(labels, notes)
                pair_id = f"friend_{name}_{mid.stem}"
                carrier_path = carrier_dir / f"{pair_id}.mid"
                src.write(str(carrier_path))
                if write_targets:
                    # strip PB from carrier-like render input: use notes only
                    target = render_target(notes, labels, program=29, source=src)
                    target.write(str(target_dir / f"{pair_id}.mid"))
                payload = {
                    "id": pair_id,
                    "source": "friend",
                    "folder": name,
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
                }
                (labels_dir / f"{pair_id}.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                for lb in labels:
                    tech_counts[lb.tech] += 1
                    n_notes += 1
                mf.write(
                    json.dumps(
                        {
                            "id": pair_id,
                            "carrier": f"carrier/{pair_id}.mid",
                            "labels": f"labels/{pair_id}.json",
                            "source": "friend",
                            "tech": tech,
                            "n_notes": len(notes),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1

    summary = {
        "source": "friend",
        "n_pairs": written,
        "n_note_labels": n_notes,
        "tech_note_counts": dict(tech_counts),
    }
    (out_dir / "summary_friend.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    import argparse

    from .constants import APPLY_DATASET_DIR

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=TECH_ROOT)
    p.add_argument("--out-dir", type=Path, default=APPLY_DATASET_DIR)
    args = p.parse_args()
    print(import_friend_guitar(root=args.root, out_dir=args.out_dir))


if __name__ == "__main__":
    main()
