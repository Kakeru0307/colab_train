# VA×BPM 整理レポート

生成: `organize_va_bpm_pairs.py --per-group 116 --seed 0`

## 学習本線セット

**`structure_pairs_balanced_116.jsonl`**（合計 **464** 件）

| グループ | source | 目標 | 実数 | BPM帯 |
|---|---|---:|---:|---|
| 高VA×BPM≥150 | bal_high_fast | 116 | 116 | {'very_fast': 116} |
| 高V低A | bal_highV_lowA | 116 | 116 | {'slow': 30, 'mid': 62, 'fast': 24} |
| 低V高A | bal_lowV_highA | 116 | 116 | {'fast': 52, 'slow': 22, 'mid': 31, 'very_fast': 11} |
| 低V低A | bal_lowV_lowA | 116 | 116 | {'slow': 13, 'mid': 74, 'fast': 29} |

プール（重複なしコア後）: high_fast=116, highV_lowA=180, lowV_highA=350, lowV_lowA=179

高V低A・低V低Aは低め BPM 優先。低V高Aは一様サンプル（≥150 専用バケットは作らない）。高VA×≥150 はプール全件が 116 未満なら全採用。

## 学習コマンド例

```text
python scripts/train_structure_prior.py \
  --jsonl data/prior_pairs/manifests/accept.jsonl \
  --extra-jsonl \
    data/emopia_pairs/structure_pairs.jsonl \
    data/va_bpm_curated/structure_pairs_balanced_116.jsonl
```

accept は進行・キー用。VA→BPM の条件つき分布は balanced_116 が主。
EMOPIA はキー／モード用（BPM マスク）。

コア除外: {'drop_high_va_slow': 17, 'drop_lowV_lowA_very_fast': 30}（計 47）
