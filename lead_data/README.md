# lead_data — 外部ソースから lead pairs を作る

方針: `docs/データ/リードフレーズデータ候補.md`

- GuitarSet: `import_guitarset.py`（JAMS、solo 優先、chord 注釈→骨格）
- SynthTab: `import_synthtab.py`（MIDI、伴奏トラックも可、ノートからコード推定）
- 一括: リポジトリ直下 `build_lead_pairs_mixed.py`（実 8–9 割 + 合成）

Colab の `GEN_LEAD` は GuitarSet を Zenodo から `/content` に取る。
SynthTab はリポジトリ内の `synthtab_lead_merged.zip`（1曲1ファイル、953曲）を `/content/synthtab_lead` に展開する。アップロードも Drive も不要。

```bash
python build_lead_pairs_mixed.py \
  --pairs-dir data/pairs/lead \
  --clear \
  --guitarset-dir /content/guitarset \
  --synthtab-dir /content/synthtab_lead \
  --real-ratio 0.85
```
