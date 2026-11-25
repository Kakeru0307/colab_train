# lead_data — 外部ソースから lead pairs を作る

方針: `docs/データ/リードフレーズデータ候補.md`

- GuitarSet: `import_guitarset.py`（JAMS、solo 優先、chord 注釈→骨格）
- SynthTab: `import_synthtab.py`（MIDI、伴奏トラックも可、ノートからコード推定）
- GAPS: `import_gaps.py`（クラシック MIDI・arp スコア上位選別）+ `oversample_pairs.py`
- 一括: リポジトリ直下 `build_lead_pairs_mixed.py`

Colab の `GEN_LEAD` 既定: **`synthtab_pick5000.zip`**（TSV **4589**＝mono≥0.3 の3914＋高音追加675）+ 旧 merged から **`synthtab_extra_solo411.txt` の411本**と EGSolo/GuitarSet + **`gaps_v1_midi.zip`**（arp 上位 **160**、実効 **~40%**・上限 45%）。SynthTab 系合計 **5000**。合成なし・一から学習。

**pair 再利用（節約・推奨）:** `colab_train.ipynb` で次を設定してリード用セルを実行。

```text
GEN_BACKING = False
GEN_LEAD = True
GEN_BASS = False
GEN_DRUM = False
FORCE_REBUILD_LEAD_PAIRS = False
LEAD_GAPS_BOOST_REUSE = True   # 旧 pairs 展開 → GAPS top160 + OS~40% → Drive 再保存
TRAIN_CVAE = True              # boost 後に lead scratch 学習
```

ST の MIDI→pair は走らない。かかるのは旧 zip 展開・GAPS MIDI 展開・GAPS import/OS・再 zip。  
`LEAD_GAPS_BOOST_REUSE=False` かつ `FORCE_REBUILD=False` だと旧低比率 zip の展開だけなので 40% にならない。

```bash
python build_lead_pairs_mixed.py \
  --pairs-dir data/pairs/lead \
  --clear \
  --skip-synthetic \
  --synthtab-dir /content/synthtab_pick5000 \
  --synthtab-min-bars 8 \
  --synthtab-no-keyword-filter \
  --gaps-dir /content/gaps_midi \
  --gaps-top-k 160 \
  --gaps-target-ratio 0.40 \
  --gaps-min-ratio 0.38 \
  --gaps-max-ratio 0.45

python train.py --pairs-dir data/pairs/lead --cvae --require-power-cond \
  --lr 1e-3 --epochs 40 ...
```

### lead pairs（Colab → Drive）

- **pair 化は Colab**（`GEN_LEAD`）。完了後 Drive `LEAD_PAIRS_DRIVE_DIR`（既定 `MyDrive/colab_train_lead_pairs`）へ保存
- ファイル名: **`M-D-HH.zip`**（例 `11-22-19.zip`）。同時刻衝突時は `M-D-HH-MM.zip`
- 識別: 同フォルダの **`latest.txt`**（1行目=最新 zip 名、2行目以降に `build_seconds` / `build_hms`）。次回 `FORCE_REBUILD_LEAD_PAIRS=False` なら latest を展開するだけ
- 作り直し: フラグを `True` → MIDI から再構築 → 新規 dated zip

同梱 MIDI zip（再構築用）: `synthtab_pick5000.zip` / `gaps_v1_midi.zip` / 旧 `synthtab_lead_merged.zip`

再生成 MIDI: `materialize_synthtab_pick.py` ← `all_jams_midi.zip` + `synthtab_pick5000.tsv`。追加ソロ MIDI は `synthtab_lead_merged.zip` + `synthtab_extra_solo411.txt`。高音追加一覧: `synthtab_extra_high675.txt`（既に pick zip に焼込み済み）。
