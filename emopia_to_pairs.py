"""EMOPIA → structure prior 学習ペア変換。

EMOPIA 2.2（感情4象限ラベル付きポップピアノ MIDI 1,087 クリップ）から
VA + BPM + キー + モード の structure_pairs.jsonl を作る。

  感情象限: ファイル名の接頭辞（Q1_xxxx_0.mid → Q1）
  キー・モード・テンポ: key_mode_tempo.csv（無ければ MIDI から推定）

EMOPIA に無い項目（進行・bars_per_chord）は書かない。
train_structure_prior.py --extra-jsonl はその項目を損失から除外する。

使い方:
    python emopia_to_pairs.py --download                 # data/emopia に取得して変換
    python emopia_to_pairs.py --emopia-dir path/to/EMOPIA_2.2
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# 符号は EMOPIA の 4象限（Q1=HVHA, Q2=LVHA, Q3=LVLA, Q4=HVLA）。大きさは仮。
Q_TO_VA: dict[str, tuple[float, float]] = {
    "Q1": (+0.6, +0.6),
    "Q2": (-0.6, +0.6),
    "Q3": (-0.6, -0.6),
    "Q4": (+0.6, -0.6),
}

# structure_prior.KEYS と同じ表記（フラット系）に揃える
_ENHARMONIC = {
    "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb",
    "Cb": "B", "Fb": "E", "E#": "F", "B#": "C",
}
KEYS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

BPM_LO, BPM_HI = 60.0, 150.0

_Q_RE = re.compile(r"^(Q[1-4])_", re.IGNORECASE)


def _fold_bpm(bpm: float) -> float:
    """倍・半テンポの取り違えを考慮して BPM_LO〜BPM_HI に折り込む。"""
    if bpm <= 0:
        return 0.0
    while bpm > BPM_HI:
        bpm /= 2.0
    while bpm < BPM_LO:
        bpm *= 2.0
    return min(BPM_HI, bpm)


def _norm_key(raw: str) -> tuple[str | None, str | None]:
    """'C#', 'c#', 'Db major', 'a minor' 等 → (KEYS 表記, mode or None)。"""
    s = raw.strip()
    if not s:
        return None, None
    mode: str | None = None
    low = s.lower()
    if "minor" in low or low.endswith("m") and not low.endswith("maj"):
        mode = "natural_minor"
    elif "major" in low:
        mode = "major"
    m = re.match(r"([A-Ga-g])([#b♯♭]?)", s)
    if not m:
        return None, mode
    letter, acc = m.group(1), m.group(2).replace("♯", "#").replace("♭", "b")
    if mode is None and letter.islower():
        mode = "natural_minor"
    name = letter.upper() + acc
    name = _ENHARMONIC.get(name, name)
    return (name if name in KEYS else None), mode


def _norm_mode(raw: str) -> str | None:
    low = raw.strip().lower()
    if low in ("major", "maj", "ionian"):
        return "major"
    if low in ("minor", "min", "aeolian", "natural_minor"):
        return "natural_minor"
    return None


def _find_col(header: list[str], *needles: str) -> str | None:
    for col in header:
        c = col.strip().lower()
        if any(n in c for n in needles):
            return col
    return None


def _load_key_mode_tempo(emopia_dir: Path) -> dict[str, dict[str, str]]:
    """key_mode_tempo.csv を {クリップ stem: 行} にする。列名は推定する。"""
    paths = list(emopia_dir.rglob("key_mode_tempo.csv"))
    if not paths:
        print("[情報] key_mode_tempo.csv なし → MIDI から推定します")
        return {}
    path = paths[0]
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    print(f"key_mode_tempo.csv: {path}  列={header}")
    id_col = _find_col(header, "name", "file", "id", "clip") or (header[0] if header else None)
    key_col = _find_col(header, "key")
    mode_col = _find_col(header, "mode")
    tempo_col = _find_col(header, "tempo", "bpm")
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if id_col is None:
            break
        stem = Path(str(row.get(id_col, "")).strip()).stem
        if not stem:
            continue
        out[stem] = {
            "key": str(row.get(key_col, "")) if key_col else "",
            "mode": str(row.get(mode_col, "")) if mode_col else "",
            "tempo": str(row.get(tempo_col, "")) if tempo_col else "",
        }
    return out


def _from_midi(midi_path: Path) -> tuple[float | None, str | None, str | None]:
    """MIDI から (bpm, key, mode)。取れない項目は None。"""
    try:
        import muspy

        music = muspy.read_midi(str(midi_path))
    except Exception:
        return None, None, None
    bpm = float(music.tempos[0].qpm) if music.tempos else None
    key = mode = None
    if music.key_signatures:
        ks = music.key_signatures[0]
        if ks.root is not None:
            key = KEYS[int(ks.root) % 12]
        if ks.mode is not None:
            mode = "natural_minor" if str(ks.mode).lower().startswith("min") or ks.mode == 1 else "major"
    return bpm, key, mode


def download(root: Path) -> Path:
    import muspy

    root.mkdir(parents=True, exist_ok=True)
    muspy.EMOPIADataset(root, download_and_extract=True)
    return root


def convert(emopia_dir: Path, out_path: Path) -> None:
    midis = sorted(p for p in emopia_dir.rglob("*.mid") if not p.name.startswith("._"))
    if not midis:
        raise FileNotFoundError(f"MIDI が見つかりません: {emopia_dir}")
    kmt = _load_key_mode_tempo(emopia_dir)

    records: list[dict] = []
    stats: Counter[str] = Counter()
    for midi in midis:
        m = _Q_RE.match(midi.name)
        if not m:
            stats["skip_no_q"] += 1
            continue
        q = m.group(1).upper()
        valence, arousal = Q_TO_VA[q]

        bpm: float | None = None
        key: str | None = None
        mode: str | None = None
        meta = kmt.get(midi.stem)
        if meta:
            key, key_mode = _norm_key(meta["key"])
            mode = _norm_mode(meta["mode"]) or key_mode
            try:
                bpm = float(meta["tempo"]) if meta["tempo"] else None
            except ValueError:
                bpm = None
            stats["csv_hit"] += 1
        if bpm is None or key is None or mode is None:
            m_bpm, m_key, m_mode = _from_midi(midi)
            bpm = bpm if bpm is not None else m_bpm
            key = key or m_key
            mode = mode or m_mode
        if bpm is None:
            stats["skip_no_bpm"] += 1
            continue

        structure: dict[str, object] = {"bpm": round(_fold_bpm(bpm), 1), "bars": 8}
        if key:
            structure["key"] = key
        if mode:
            structure["mode"] = mode
        records.append({
            "source": "emopia",
            "clip": midi.stem,
            "q_label": q,
            "va": {"valence": valence, "arousal": arousal},
            "structure": structure,
        })
        stats[q] += 1
        stats["has_key"] += int(key is not None)
        stats["has_mode"] += int(mode is not None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"書き出し: {len(records)} 件 → {out_path}")
    print(f"内訳: {dict(stats)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EMOPIA → structure prior 学習ペア")
    parser.add_argument("--emopia-dir", type=Path, default=SCRIPT_DIR / "data" / "emopia")
    parser.add_argument("--download", action="store_true", help="MuSpy 経由で EMOPIA 2.2 を取得")
    parser.add_argument(
        "--out",
        type=Path,
        default=SCRIPT_DIR / "data" / "emopia_pairs" / "structure_pairs.jsonl",
    )
    args = parser.parse_args()

    if args.download:
        download(args.emopia_dir)
    convert(args.emopia_dir, args.out)


if __name__ == "__main__":
    main()
