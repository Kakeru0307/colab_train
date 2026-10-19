"""SynthTab + GuitarSet + 合成 lead pairs を混合して data/pairs/lead を作る。

本決め:
  実データ 8〜9 割 / 合成 1〜2 割
  SynthTab は伴奏っぽいものも残す
  実データの power mask はゼロ
  旧 pairs は上書き前に消してよい
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from generate_lead_pairs import generate_lead_pairs
from lead_data.import_guitarset import import_guitarset
from lead_data.import_synthtab import import_synthtab

SCRIPT_DIR = Path(__file__).resolve().parent


def _count_patches(pairs_dir: Path) -> int:
    target = pairs_dir / "target"
    if not target.is_dir():
        return 0
    return sum(1 for _ in target.rglob("*_tonal.npy"))


def main() -> None:
    p = argparse.ArgumentParser(description="混合 lead pairs を構築")
    p.add_argument("--pairs-dir", type=Path, default=SCRIPT_DIR / "data" / "pairs" / "lead")
    p.add_argument("--synthtab-dir", type=Path, default=None, help="SynthTab MIDI ルート")
    p.add_argument("--guitarset-dir", type=Path, default=None, help="GuitarSet JAMS ルート")
    p.add_argument("--real-ratio", type=float, default=0.85, help="実パッチ目標比率 (0.8–0.9)")
    p.add_argument("--synth-count", type=int, default=None, help="合成曲数（省略時は実に合わせて自動）")
    p.add_argument("--synthtab-max-files", type=int, default=None)
    p.add_argument("--guitarset-max-songs", type=int, default=None)
    p.add_argument("--clear", action="store_true", help="先に pairs-dir を削除")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-synthtab", action="store_true")
    p.add_argument("--skip-guitarset", action="store_true")
    p.add_argument("--skip-synthetic", action="store_true")
    args = p.parse_args()

    pairs_dir = args.pairs_dir
    if args.clear and pairs_dir.exists():
        print(f"remove {pairs_dir}")
        shutil.rmtree(pairs_dir)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"mode": "lead_mixed", "parts": {}, "real_ratio_target": args.real_ratio}

    # --- real ---
    if not args.skip_guitarset:
        if args.guitarset_dir and args.guitarset_dir.is_dir():
            print("=== GuitarSet ===")
            st = import_guitarset(
                args.guitarset_dir,
                pairs_dir,
                max_songs=args.guitarset_max_songs,
            )
            manifest["parts"]["guitarset"] = st
        else:
            print("GuitarSet: dir なし → スキップ", args.guitarset_dir)

    if not args.skip_synthtab:
        if args.synthtab_dir and args.synthtab_dir.is_dir():
            print("=== SynthTab ===")
            st = import_synthtab(
                args.synthtab_dir,
                pairs_dir,
                max_files=args.synthtab_max_files,
            )
            manifest["parts"]["synthtab"] = st
        else:
            print("SynthTab: dir なし → スキップ", args.synthtab_dir)

    real_patches = _count_patches(pairs_dir)
    print(f"real patches so far: {real_patches}")

    # --- synthetic to hit ratio ---
    if not args.skip_synthetic:
        ratio = min(0.9, max(0.8, float(args.real_ratio)))
        if args.synth_count is not None:
            synth_songs = args.synth_count
        elif real_patches <= 0:
            synth_songs = 600
            print("実パッチ 0 → 合成のみ仮に", synth_songs, "曲")
        else:
            # real / (real+synth) ≈ ratio  → synth ≈ real * (1-ratio)/ratio
            # 1 song ≈ 1 patch (8 bars) for synthetic
            synth_songs = max(1, int(round(real_patches * (1.0 - ratio) / ratio)))
        print(f"=== synthetic ({synth_songs} songs) ===")
        # write into temp then merge — generate_lead_pairs writes song ids lead#####
        st = generate_lead_pairs(pairs_dir, count=synth_songs, seed=args.seed)
        manifest["parts"]["synthetic"] = {
            "songs": len(st.get("songs", [])),
            "total_patches": st.get("total_patches", 0),
        }

    total = _count_patches(pairs_dir)
    real_final = total - int(manifest.get("parts", {}).get("synthetic", {}).get("total_patches", 0))
    manifest["total_patches"] = total
    manifest["real_patches_approx"] = max(0, real_final)
    if total > 0:
        manifest["real_ratio_actual"] = round(real_final / total, 4)
    out = pairs_dir / "manifest_mixed.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(
        f"total={total}  real~{manifest.get('real_patches_approx')}  "
        f"ratio~{manifest.get('real_ratio_actual')}"
    )


if __name__ == "__main__":
    main()
