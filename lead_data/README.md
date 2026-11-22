# lead_data — 外部ソースから lead pairs を作る

方針: `docs/データ/リードフレーズデータ候補.md`

- GuitarSet: `import_guitarset.py`（JAMS、solo 優先、chord 注釈→骨格）
- SynthTab: `import_synthtab.py`（MIDI、伴奏トラックも可、ノートからコード推定）
- GAPS: `import_gaps.py`（クラシック MIDI・arp スコア上位選別）+ `oversample_pairs.py`
- 一括: リポジトリ直下 `build_lead_pairs_mixed.py`

Colab の `GEN_LEAD` 既定: **`synthtab_pick5000.zip`** + **`gaps_v1_midi.zip`**（arp 上位 100、5–8% OS）+ 旧 zip から EGSolo/GuitarSet。合成なし・一から学習。

```bash
python build_lead_pairs_mixed.py \
  --pairs-dir data/pairs/lead \
  --clear \
  --skip-synthetic \
  --synthtab-dir /content/synthtab_pick5000 \
  --synthtab-min-bars 8 \
  --synthtab-no-keyword-filter \
  --gaps-dir /content/gaps_midi \
  --gaps-top-k 100 \
  --gaps-target-ratio 0.06 \
  --gaps-min-ratio 0.05 \
  --gaps-max-ratio 0.10

python train.py --pairs-dir data/pairs/lead --cvae --require-power-cond \
  --lr 1e-3 --epochs 40 ...
```

### lead pairs（Colab → Drive）

- **pair 化は Colab**（`GEN_LEAD`）。完了後 Drive `LEAD_PAIRS_DRIVE_DIR`（既定 `MyDrive/colab_train_lead_pairs`）へ保存
- ファイル名: **`M-D-HH.zip`**（例 `11-22-19.zip`）。同時刻衝突時は `M-D-HH-MM.zip`
- 識別: 同フォルダの **`latest.txt`**（最新 zip 名）。次回 `FORCE_REBUILD_LEAD_PAIRS=False` なら latest を展開するだけ
- 作り直し: フラグを `True` → MIDI から再構築 → 新規 dated zip

同梱 MIDI zip（再構築用）: `synthtab_pick5000.zip` / `gaps_v1_midi.zip` / 旧 `synthtab_lead_merged.zip`

再生成 MIDI: `materialize_synthtab_pick.py` ← `all_jams_midi.zip` + `synthtab_pick5000.tsv`。
