"""pairs 内の特定 prefix 曲を複製して実効比率を上げる。"""

from __future__ import annotations

import math
import shutil
from pathlib import Path


def _count_tonal(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob("*_tonal.npy"))


def count_patches_by_prefix(pairs_dir: Path, prefix: str) -> tuple[int, int, list[str]]:
    """(prefix_patches, total_patches, song_ids_with_prefix)."""
    target = pairs_dir / "target"
    if not target.is_dir():
        return 0, 0, []
    song_ids = sorted(p.name for p in target.iterdir() if p.is_dir())
    pref_ids = [s for s in song_ids if s.startswith(prefix)]
    # OS 複製（_os02 等）は元 prefix 曲に含める
    pref_patches = sum(_count_tonal(target / s) for s in pref_ids)
    total = sum(_count_tonal(target / s) for s in song_ids)
    return pref_patches, total, pref_ids


def oversample_prefix(
    pairs_dir: Path,
    *,
    prefix: str,
    min_ratio: float = 0.05,
    target_ratio: float = 0.06,
    max_ratio: float = 0.10,
) -> dict:
    """prefix 曲のパッチ比率が min_ratio 未満なら複製して target_ratio 付近へ。

    複製先 id: ``{song_id}_os{n:02d}``（n>=2）。上限 max_ratio。
    すでに _os 付きの id はコピー元にしない（元曲だけ倍化）。
    """
    pref_patches, total, pref_ids = count_patches_by_prefix(pairs_dir, prefix)
    base_ids = [s for s in pref_ids if "_os" not in s]
    other = max(0, total - pref_patches)
    ratio = (pref_patches / total) if total > 0 else 0.0

    stats: dict = {
        "prefix": prefix,
        "before_patches": pref_patches,
        "before_total": total,
        "before_ratio": round(ratio, 4),
        "copies": 0,
        "added_patches": 0,
        "repeat_factor": 1,
        "action": "none",
    }

    if total <= 0 or pref_patches <= 0 or not base_ids:
        stats["action"] = "skip_empty"
        print(f"[oversample] {prefix}: empty → skip")
        return stats

    if ratio >= min_ratio:
        stats["action"] = "already_ok"
        print(
            f"[oversample] {prefix}: ratio={ratio:.4f} >= min={min_ratio} → no copy"
        )
        return stats

    # G' / (other + G') = target  →  G' = target * other / (1 - target)
    tgt = min(max_ratio, max(min_ratio, float(target_ratio)))
    if tgt >= 1.0:
        tgt = 0.99
    want = int(math.ceil(tgt * other / (1.0 - tgt)))
    # cap by max_ratio
    want_max = int(math.floor(max_ratio * other / (1.0 - max_ratio)))
    want = min(want, want_max) if want_max > 0 else want
    if want <= pref_patches:
        stats["action"] = "want_le_have"
        print(f"[oversample] {prefix}: want={want} <= have={pref_patches} → no copy")
        return stats

    factor = int(math.ceil(want / pref_patches))
    factor = max(2, factor)
    # safety: after factor, ratio <= max_ratio
    while factor > 1:
        g_after = pref_patches * factor
        r_after = g_after / (other + g_after)
        if r_after <= max_ratio + 1e-9:
            break
        factor -= 1
    if factor <= 1:
        stats["action"] = "capped_no_copy"
        print(f"[oversample] {prefix}: max_ratio cap → no copy")
        return stats

    input_root = pairs_dir / "input"
    target_root = pairs_dir / "target"
    added = 0
    copies = 0
    for rep in range(2, factor + 1):
        for sid in base_ids:
            new_id = f"{sid}_os{rep:02d}"
            src_i = input_root / sid
            src_t = target_root / sid
            dst_i = input_root / new_id
            dst_t = target_root / new_id
            if not src_t.is_dir() or dst_t.exists():
                continue
            shutil.copytree(src_t, dst_t)
            if src_i.is_dir():
                shutil.copytree(src_i, dst_i)
            n = _count_tonal(dst_t)
            added += n
            copies += 1

    after_p, after_t, _ = count_patches_by_prefix(pairs_dir, prefix)
    stats.update(
        {
            "action": "copied",
            "repeat_factor": factor,
            "copies": copies,
            "added_patches": added,
            "after_patches": after_p,
            "after_total": after_t,
            "after_ratio": round(after_p / after_t, 4) if after_t else 0.0,
        }
    )
    print(
        f"[oversample] {prefix}: ×{factor}  {ratio:.4f}→{stats['after_ratio']}  "
        f"added={added} copies={copies}"
    )
    return stats
