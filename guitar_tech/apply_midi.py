"""Apply learned technique classifier + parametric renderer to a MIDI file."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pretty_midi as pm
import torch

from .apply_model import TechApplyNet
from .carriers import NoteEvent
from .constants import CHECKPOINT_DIR, DEFAULT_PROGRAM, OUTPUT_DIR
from .features import note_features, pretty_midi_to_notes
from .params import TECH_NAMES, sample_params
from .render import NoteLabel, render_target


def load_apply_model(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[TechApplyNet, list[str]]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = blob.get("meta") or {}
    names = list(meta.get("tech_names") or TECH_NAMES)
    model = TechApplyNet(len(names), hidden=int(meta.get("hidden", 128)))
    model.load_state_dict(blob["model_state_dict"])
    model.to(device)
    model.eval()
    return model, names


@torch.inference_mode()
def predict_labels(
    notes: list[NoteEvent],
    model: TechApplyNet,
    tech_names: list[str],
    *,
    device: torch.device,
    rng: random.Random,
    none_bias: float = 0.0,
) -> list[NoteLabel]:
    if not notes:
        return []
    feats = torch.from_numpy(note_features(notes)).float().to(device)
    logits = model(feats)
    if none_bias:
        logits = logits.clone()
        logits[:, 0] = logits[:, 0] + float(none_bias)
    pred = logits.argmax(dim=-1).cpu().tolist()
    labels: list[NoteLabel] = []
    for i, tid in enumerate(pred):
        tech = tech_names[int(tid)] if 0 <= int(tid) < len(tech_names) else "none"
        params = sample_params(tech, rng) if tech != "none" else {}
        labels.append(NoteLabel(note_index=i, tech=tech, params=params))
    return labels


def apply_techniques_to_midi(
    midi_path: Path | str,
    *,
    checkpoint: Path | None = None,
    out_path: Path | None = None,
    seed: int = 0,
    none_bias: float = 0.0,
    program: int | None = None,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = checkpoint or (CHECKPOINT_DIR / "guitar_tech_apply_last.pt")
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"apply checkpoint missing: {ckpt}")

    model, tech_names = load_apply_model(Path(ckpt), device)
    src = pm.PrettyMIDI(str(midi_path))
    notes = pretty_midi_to_notes(src)
    rng = random.Random(seed)
    labels = predict_labels(
        notes, model, tech_names, device=device, rng=rng, none_bias=none_bias
    )
    prog = DEFAULT_PROGRAM if program is None else int(program)
    # preserve first non-drum program if present
    if program is None:
        for inst in src.instruments:
            if not inst.is_drum:
                prog = int(inst.program)
                break
    out_midi = render_target(notes, labels, program=prog, rng=rng)

    out = Path(out_path) if out_path else (
        OUTPUT_DIR / f"{Path(midi_path).stem}_tech.mid"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out_midi.write(str(out))

    meta_path = out.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "source": str(midi_path),
                "checkpoint": str(ckpt),
                "labels": [
                    {"note_index": lb.note_index, "tech": lb.tech, "params": lb.params}
                    for lb in labels
                ],
                "tech_counts": {
                    t: sum(1 for lb in labels if lb.tech == t)
                    for t in sorted({lb.tech for lb in labels})
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply guitar techniques to MIDI")
    parser.add_argument("midi", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "guitar_tech_apply_last.pt",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--none-bias",
        type=float,
        default=0.0,
        help="none クラス logit に足すバイアス（大きいほど控えめ）",
    )
    args = parser.parse_args()
    path = apply_techniques_to_midi(
        args.midi,
        checkpoint=args.checkpoint,
        out_path=args.out,
        seed=args.seed,
        none_bias=args.none_bias,
    )
    print(f"wrote {path}")
    print(f"meta {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
