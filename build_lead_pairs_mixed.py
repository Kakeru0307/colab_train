"""SynthTab + GAPS(+任意 AnimeTAB) + 合成 lead pairs を混合して data/pairs/lead を作る。

本決め:
  実データ中心。合成は --skip-synthetic で省略可
  GAPS は arp 高スコア寄り選別後、実効 5–8%（上限 10%）へ必要なら複製
  実データの power mask はゼロ
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from generate_lead_pairs import generate_lead_pairs
from lead_data.import_animetab import import_animetab
from lead_data.import_gaps import SONG_ID_PREFIX as GAPS_PREFIX
from lead_data.import_gaps import import_gaps
from lead_data.import_synthtab import import_synthtab
from lead_data.oversample_pairs import oversample_prefix

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
    p.add_argument("--animetab-dir", type=Path, default=None, help="AnimeTAB Clips ルート")
    p.add_argument("--gaps-dir", type=Path, default=None, help="GAPS MIDI ルート")
    p.add_argument(
        "--synthtab-sectioned",
        action="store_true",
        help="SynthTab を SongFormer 区間ラベル付きで取り込む",
    )
    p.add_argument("--songformer-device", type=str, default="cuda:0")
    p.add_argument("--songformer-cache-dir", type=Path, default=None)
    p.add_argument("--real-ratio", type=float, default=0.85, help="実パッチ目標比率 (0.8–0.9)")
    p.add_argument("--synth-count", type=int, default=None, help="合成曲数（省略時は実に合わせて自動）")
    p.add_argument("--synthtab-max-files", type=int, default=None)
    p.add_argument(
        "--synthtab-min-bars",
        type=int,
        default=32,
        help="SynthTab 取り込みの最小小節数（選別済み短尺ソロは 8 推奨）",
    )
    p.add_argument(
        "--synthtab-no-keyword-filter",
        action="store_true",
        help="ファイル名キーワード除外を無効化（選別済み zip 向け）",
    )
    p.add_argument("--animetab-max-files", type=int, default=None)
    p.add_argument("--gaps-top-k", type=int, default=100, help="GAPS arp スコア上位何本を使うか")
    p.add_argument("--gaps-min-score", type=float, default=None)
    p.add_argument(
        "--gaps-pick-tsv",
        type=Path,
        default=None,
        help="選別結果 TSV（省略時 lead_data/gaps_pick_topk.tsv）",
    )
    p.add_argument("--gaps-min-bars", type=int, default=8)
    p.add_argument(
        "--gaps-target-ratio",
        type=float,
        default=0.06,
        help="GAPS 実効パッチ目標比率（min 未満なら複製）",
    )
    p.add_argument("--gaps-min-ratio", type=float, default=0.05)
    p.add_argument("--gaps-max-ratio", type=float, default=0.10)
    p.add_argument("--skip-gaps-oversample", action="store_true")
    p.add_argument("--clear", action="store_true", help="先に pairs-dir を削除")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-synthtab", action="store_true")
    p.add_argument("--skip-animetab", action="store_true")
    p.add_argument("--skip-gaps", action="store_true")
    p.add_argument("--skip-synthetic", action="store_true")
    args = p.parse_args()

    pairs_dir = args.pairs_dir
    if args.clear and pairs_dir.exists():
        print(f"remove {pairs_dir}")
        shutil.rmtree(pairs_dir)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"mode": "lead_mixed", "parts": {}, "real_ratio_target": args.real_ratio}

    if not args.skip_animetab:
        if args.animetab_dir and args.animetab_dir.is_dir():
            print("=== AnimeTAB ===")
            at = import_animetab(
                args.animetab_dir,
                pairs_dir,
                max_files=args.animetab_max_files,
            )
            manifest["parts"]["animetab"] = at
        else:
            print("AnimeTAB: dir なし → スキップ", args.animetab_dir)

    # --- real（SynthTab）---
    if not args.skip_synthtab:
        if args.synthtab_dir and args.synthtab_dir.is_dir():
            if args.synthtab_sectioned:
                print("=== SynthTab + SongFormer sections ===")
                from lead_data.label_synthtab_sections import import_synthtab_with_sections

                st = import_synthtab_with_sections(
                    args.synthtab_dir,
                    pairs_dir,
                    device=args.songformer_device,
                    max_files=args.synthtab_max_files,
                    cache_dir=args.songformer_cache_dir,
                )
            else:
                print("=== SynthTab ===")
                st_kw: dict = {
                    "max_files": args.synthtab_max_files,
                    "min_bars": args.synthtab_min_bars,
                }
                if args.synthtab_no_keyword_filter:
                    st_kw["exclude_keywords"] = ()
                st = import_synthtab(args.synthtab_dir, pairs_dir, **st_kw)
            manifest["parts"]["synthtab"] = st
        else:
            print("SynthTab: dir なし → スキップ", args.synthtab_dir)

    # --- GAPS（arp 選別）---
    if not args.skip_gaps:
        if args.gaps_dir and args.gaps_dir.is_dir():
            print("=== GAPS ===")
            pick_tsv = args.gaps_pick_tsv
            if pick_tsv is None:
                pick_tsv = SCRIPT_DIR / "lead_data" / "gaps_pick_topk.tsv"
            gp = import_gaps(
                args.gaps_dir,
                pairs_dir,
                top_k=args.gaps_top_k,
                min_score=args.gaps_min_score,
                pick_tsv=pick_tsv,
                min_bars=args.gaps_min_bars,
            )
            manifest["parts"]["gaps"] = {
                k: v for k, v in gp.items() if k != "song_ids"
            }
            manifest["parts"]["gaps"]["pick_tsv"] = str(pick_tsv)

            if not args.skip_gaps_oversample:
                print("=== GAPS oversample ===")
                os_stats = oversample_prefix(
                    pairs_dir,
                    prefix=GAPS_PREFIX,
                    min_ratio=args.gaps_min_ratio,
                    target_ratio=args.gaps_target_ratio,
                    max_ratio=args.gaps_max_ratio,
                )
                manifest["parts"]["gaps_oversample"] = os_stats
        else:
            print("GAPS: dir なし → スキップ", args.gaps_dir)

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
            synth_songs = max(1, int(round(real_patches * (1.0 - ratio) / ratio)))
        print(f"=== synthetic ({synth_songs} songs) ===")
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
