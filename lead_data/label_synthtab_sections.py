"""SynthTab MIDI → 音声化 → SongFormer → section 付き lead pairs。

Phase0 スパイク: ``--spike N`` で数曲だけ推論しラベル分布を表示。
本実装: ``--pairs-dir`` へ section 付き pair を書き出す。

依存（Colab 想定）:
  - pretty_midi / fluidsynth or pretty_midi.synthesize
  - SongFormer (transformers + HF ASLP-lab/SongFormer)
  - pip: muq omegaconf ema-pytorch loguru x-transformers huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from lead_data.chord_estimate import estimate_chords_per_bar, estimate_key_from_notes
from lead_data.import_synthtab import _iter_midi_files, _is_excluded, _load_music
from lead_data.midi_windows import (
    iter_bar_windows,
    music_bpm,
    music_notes_ticks,
    notes_to_music,
    slice_notes_window,
    total_bars_from_notes,
)
from lead_data.skeleton import build_skeleton_from_bar_chords
from lead_data.write_pairs import save_lead_pair_song
from makeData.constants import DEFAULT_BARS
from song_form import (
    SECTION_ROLES,
    map_songformer_label,
    section_role_to_id,
)

_DEFAULT_EXCLUDE_KEYWORDS: tuple[str, ...] = ("lick", "intro", "exercise", "lesson")


_SOUNDFONT_CANDIDATES = (
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/content/FluidR3_GM.sf2",
)


def _find_soundfont() -> Path | None:
    for p in _SOUNDFONT_CANDIDATES:
        path = Path(p)
        if path.is_file():
            return path
    return None


def render_midi_to_wav(midi_path: Path, wav_path: Path, *, sample_rate: int = 24000) -> None:
    """MIDI → WAV。優先: fluidsynth CLI、次点: pretty_midi.synthesize。"""
    import shutil
    import subprocess

    import soundfile as sf

    sf2 = _find_soundfont()
    fluidsynth = shutil.which("fluidsynth")
    if fluidsynth is not None and sf2 is not None:
        cmd = [
            fluidsynth,
            "-ni",
            str(sf2),
            str(midi_path),
            "-F",
            str(wav_path),
            "-r",
            str(int(sample_rate)),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        if wav_path.is_file() and wav_path.stat().st_size > 0:
            return

    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    try:
        audio = pm.synthesize(fs=sample_rate)
    except Exception as exc:
        raise RuntimeError(
            "MIDI レンダ失敗。Colab では "
            "`apt-get install -y fluidsynth fluid-soundfont-gm` が必要です。"
        ) from exc
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    sf.write(str(wav_path), audio.astype(np.float32), sample_rate)


_SONGFORMER_PIP = (
    "muq",
    "omegaconf",
    "transformers",
    "huggingface_hub",
    "ema-pytorch",
    "loguru",
    "x-transformers",
)


def _ensure_pkg_inits(root: Path, rel: str) -> None:
    """namespace package だと後段の dataset.py に負けるので通常パッケージにする。"""
    cur = root
    for part in Path(rel).parts:
        cur = cur / part
        if not cur.is_dir():
            return
        init = cur / "__init__.py"
        if not init.is_file():
            init.write_text("", encoding="utf-8")


def _evict_module_tree(name: str, keep_under: Path) -> None:
    import sys

    keep = keep_under.resolve()
    for key in list(sys.modules):
        if key != name and not key.startswith(name + "."):
            continue
        origin = getattr(sys.modules[key], "__file__", None) or ""
        try:
            if origin and Path(origin).resolve().is_relative_to(keep):
                continue
        except (ValueError, OSError):
            pass
        sys.modules.pop(key, None)


def _ensure_msaf_stub() -> None:
    """推論では msaf を使わない。未インストールならダミーを置く。"""
    import sys
    import types

    try:
        import msaf  # noqa: F401
        return
    except ImportError:
        pass
    msaf = types.ModuleType("msaf")
    msaf_eval = types.ModuleType("msaf.eval")
    msaf_eval.compute_results = lambda *a, **k: {}
    msaf.eval = msaf_eval
    sys.modules["msaf"] = msaf
    sys.modules["msaf.eval"] = msaf_eval


def _prepare_songformer_path(local_dir: str) -> None:
    """HF スナップショットを sys.path 先頭へ。repo の dataset.py / model.py と衝突させない。"""
    import sys

    root = Path(local_dir).resolve()
    for rel in (
        "dataset",
        "postprocessing",
        "musicfm",
        "musicfm/model",
        "musicfm/modules",
    ):
        _ensure_pkg_inits(root, rel)
    _evict_module_tree("dataset", root)
    _evict_module_tree("model", root)
    _evict_module_tree("musicfm", root)
    _evict_module_tree("postprocessing", root)

    repo_root = Path(__file__).resolve().parents[1]
    blocked = {repo_root.resolve(), Path.cwd().resolve()}
    cleaned: list[str] = []
    for p in sys.path:
        if p in ("", "."):
            continue
        try:
            if Path(p).resolve() in blocked:
                continue
        except (OSError, RuntimeError):
            pass
        cleaned.append(p)
    sys.path[:] = cleaned
    sys.path.insert(0, str(root))


def load_songformer(device: str = "cuda:0"):
    """HF SongFormer をロード。"""
    import os

    missing = []
    for pip_name, mod in (
        ("muq", "muq"),
        ("omegaconf", "omegaconf"),
        ("transformers", "transformers"),
        ("huggingface_hub", "huggingface_hub"),
        ("ema-pytorch", "ema_pytorch"),
        ("loguru", "loguru"),
        ("x-transformers", "x_transformers"),
    ):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        raise ImportError(
            "SongFormer 依存が不足: "
            + ", ".join(missing)
            + " → pip install "
            + " ".join(_SONGFORMER_PIP)
        )

    import inspect

    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    snap_kw: dict = {
        "repo_id": "ASLP-lab/SongFormer",
        "repo_type": "model",
        "ignore_patterns": ["SongFormer.pt", "SongFormer.safetensors"],
    }
    snap_params = inspect.signature(snapshot_download).parameters
    if "local_dir_use_symlinks" in snap_params:
        snap_kw["local_dir_use_symlinks"] = False
    if "resume_download" in snap_params:
        snap_kw["resume_download"] = True
    local_dir = snapshot_download(**snap_kw)
    _prepare_songformer_path(local_dir)
    _ensure_msaf_stub()
    os.environ["SONGFORMER_LOCAL_DIR"] = local_dir
    model = AutoModel.from_pretrained(
        local_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model.to(device)
    model.eval()
    return model


def run_songformer(model, wav_path: Path) -> list[dict]:
    """戻り値: [{start, end, label}, ...]"""
    result = model(str(wav_path))
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and "segments" in result:
        return list(result["segments"])
    raise TypeError(f"unexpected SongFormer output: {type(result)}")


def _seconds_per_bar(bpm: float) -> float:
    # 4/4 想定: 1 bar = 4 beats
    return 4.0 * 60.0 / max(1e-6, float(bpm))


def majority_role_for_window(
    segments: list[dict],
    *,
    start_bar: int,
    bars: int,
    bpm: float,
) -> str | None:
    """窓内で占有時間が最大の role。silence のみなら None。"""
    spb = _seconds_per_bar(bpm)
    win_start = start_bar * spb
    win_end = (start_bar + bars) * spb
    scores: Counter[str] = Counter()
    for seg in segments:
        label = str(seg.get("label", ""))
        role = map_songformer_label(label)
        if role is None:
            continue
        s = float(seg["start"])
        e = float(seg["end"])
        overlap = max(0.0, min(e, win_end) - max(s, win_start))
        if overlap > 0:
            scores[role] += overlap
    if not scores:
        return None
    return scores.most_common(1)[0][0]


def spike_songformer(
    midi_root: Path,
    *,
    n: int = 5,
    device: str = "cuda:0",
    out_json: Path | None = None,
    min_bars: int = 8,
    exclude_keywords: tuple[str, ...] = (),
) -> dict:
    """数曲だけ SongFormer を回してラベル分布を返す。

    確認用のため既定ではキーワード除外なし・min_bars=8。
    大きい MIDI から優先して試す。
    """
    files = _iter_midi_files(midi_root)
    files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    model = load_songformer(device=device)
    label_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    per_file: list[dict] = []
    skipped = {"keyword": 0, "short": 0, "load": 0, "infer": 0}

    used = 0
    for path in files:
        if used >= n:
            break
        if exclude_keywords and _is_excluded(path, exclude_keywords):
            skipped["keyword"] += 1
            continue
        try:
            music = _load_music(path)
            notes = music_notes_ticks(music)
            n_bars = total_bars_from_notes(notes)
            if n_bars < min_bars:
                skipped["short"] += 1
                continue
            bpm = music_bpm(music)
        except Exception as exc:
            skipped["load"] += 1
            print(f"[spike] skip load {path.name}: {exc}")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "t.wav"
            try:
                render_midi_to_wav(path, wav)
                segs = run_songformer(model, wav)
            except Exception as exc:
                skipped["infer"] += 1
                print(f"[spike] fail {path.name}: {exc}")
                continue

        for seg in segs:
            raw = str(seg.get("label", ""))
            label_counts[raw] += 1
            role = map_songformer_label(raw)
            if role is not None:
                role_counts[role] += 1

        per_file.append(
            {
                "file": path.name,
                "bpm": bpm,
                "bars": n_bars,
                "segments": segs,
            }
        )
        used += 1
        print(f"[spike] {used}/{n} {path.name} bars={n_bars} segs={len(segs)}")

    if used == 0:
        raise RuntimeError(
            f"スパイク 0 件。midi_root={midi_root} skipped={skipped} "
            f"files={len(files)}"
        )

    summary = {
        "files": used,
        "skipped": skipped,
        "raw_labels": dict(label_counts),
        "roles": dict(role_counts),
        "section_roles": list(SECTION_ROLES),
        "per_file": per_file,
    }
    print("[spike] raw_labels:", dict(label_counts))
    print("[spike] roles:", dict(role_counts))
    print("[spike] skipped:", skipped)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[spike] wrote {out_json}")
    return summary


def import_synthtab_with_sections(
    midi_root: Path,
    pairs_dir: Path,
    *,
    device: str = "cuda:0",
    max_files: int | None = None,
    window_bars: int = DEFAULT_BARS,
    stride_bars: int = 8,
    min_onsets: int = 1,
    min_notes: int = 4,
    min_bars: int = 32,
    cache_dir: Path | None = None,
) -> dict:
    """SongFormer ラベル付きで SynthTab を取り込む。"""
    files = _iter_midi_files(midi_root)
    model = load_songformer(device=device)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "source": "synthtab_songformer",
        "songs": [],
        "total_patches": 0,
        "files": 0,
        "skipped_short": 0,
        "skipped_keyword": 0,
        "no_section": 0,
        "by_role": {r: 0 for r in SECTION_ROLES},
    }
    pairs_dir.mkdir(parents=True, exist_ok=True)

    used = 0
    for path in files:
        if max_files is not None and used >= max_files:
            break
        if _is_excluded(path, _DEFAULT_EXCLUDE_KEYWORDS):
            stats["skipped_keyword"] += 1
            continue
        try:
            music = _load_music(path)
        except Exception as exc:
            print(f"[synthtab+sf] skip {path.name}: {exc}")
            continue

        bpm = music_bpm(music)
        notes = music_notes_ticks(music)
        if len(notes) < min_notes:
            continue
        n_bars = total_bars_from_notes(notes)
        if n_bars < min_bars:
            stats["skipped_short"] += 1
            continue

        cache_path = None
        if cache_dir is not None:
            cache_path = cache_dir / f"{path.stem}.json"

        if cache_path is not None and cache_path.is_file():
            segs = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            with tempfile.TemporaryDirectory() as tmp:
                wav = Path(tmp) / "t.wav"
                try:
                    render_midi_to_wav(path, wav)
                    segs = run_songformer(model, wav)
                except Exception as exc:
                    print(f"[synthtab+sf] songformer fail {path.name}: {exc}")
                    continue
            if cache_path is not None:
                cache_path.write_text(
                    json.dumps(segs, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

        try:
            rel = path.relative_to(midi_root)
        except ValueError:
            rel = Path(path.stem)
        tag = str(rel.with_suffix("")).replace("\\", "_").replace("/", "_")[:80]

        song_patches = 0
        for wi, start_bar in enumerate(
            iter_bar_windows(n_bars, window=window_bars, stride=stride_bars)
        ):
            win = slice_notes_window(notes, start_bar=start_bar, bars=window_bars)
            if len(win) < min_onsets:
                continue
            role = majority_role_for_window(
                segs, start_bar=start_bar, bars=window_bars, bpm=bpm
            )
            section_id = section_role_to_id(role) if role is not None else None
            if section_id is None:
                stats["no_section"] += 1
            key = estimate_key_from_notes(win)
            chords = estimate_chords_per_bar(win, bars=window_bars, key=key)
            try:
                target = notes_to_music(win, bars=window_bars, bpm=bpm)
                inp = build_skeleton_from_bar_chords(chords, bpm=bpm)
            except Exception as exc:
                print(f"[synthtab+sf] build fail {tag}@{start_bar}: {exc}")
                continue
            song_id = f"stab_sf_{tag}_{wi:02d}"
            n = save_lead_pair_song(
                pairs_dir,
                song_id,
                input_music=inp,
                target_music=target,
                bpm=bpm,
                min_onsets=min_onsets,
                zero_power_mask=True,
                section_id=section_id,
            )
            song_patches += n
            if role is not None:
                stats["by_role"][role] = stats["by_role"].get(role, 0) + n

        if song_patches > 0:
            used += 1
            stats["files"] += 1
            stats["total_patches"] += song_patches
            stats["songs"].append({"id": tag, "patches": song_patches, "bars": n_bars})
            if used % 20 == 0:
                print(
                    f"[synthtab+sf] {used} files ... patches={stats['total_patches']} "
                    f"by_role={stats['by_role']}"
                )

    print(
        f"[synthtab+sf] done  files={stats['files']}  patches={stats['total_patches']}  "
        f"by_role={stats['by_role']}  no_section_windows={stats['no_section']}"
    )
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="SynthTab × SongFormer section labeling")
    p.add_argument("--midi-dir", type=Path, required=True)
    p.add_argument("--pairs-dir", type=Path, default=None)
    p.add_argument("--spike", type=int, default=0, help=">0 なら数曲スパイクのみ")
    p.add_argument("--spike-out", type=Path, default=None)
    p.add_argument(
        "--spike-min-bars",
        type=int,
        default=8,
        help="スパイク時の最小小節数（確認用は 8）",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    args = p.parse_args()

    if args.spike > 0:
        spike_songformer(
            args.midi_dir,
            n=args.spike,
            device=args.device,
            out_json=args.spike_out,
            min_bars=args.spike_min_bars,
        )
        return
    if args.pairs_dir is None:
        raise SystemExit("--pairs-dir が必要です（または --spike）")
    import_synthtab_with_sections(
        args.midi_dir,
        args.pairs_dir,
        device=args.device,
        max_files=args.max_files,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
