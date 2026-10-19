# lead_data — 外部ソースから lead pairs を作る

方針: `docs/データ/リードフレーズデータ候補.md`

- GuitarSet: `import_guitarset.py`（JAMS、solo 優先、chord 注釈→骨格）
- SynthTab: `import_synthtab.py`（MIDI、伴奏トラックも可、ノートからコード推定）
- 一括: リポジトリ直下 `build_lead_pairs_mixed.py`（実 8–9 割 + 合成）

```bash
python build_lead_pairs_mixed.py \
  --pairs-dir data/pairs/lead \
  --clear \
  --synthtab-dir /path/to/synthtab_midi \
  --guitarset-dir /path/to/guitarset_jams \
  --real-ratio 0.85
```
