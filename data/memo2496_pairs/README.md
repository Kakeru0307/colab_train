# memo2496_pairs

Memo2496（MusicRaw）から作った structure prior 用 JSONL。

| ファイル | 内容 |
|---|---|
| `structure_pairs.jsonl` | 学習用（2495行。進行ラベル付きは再マッチ後 ~1785） |
| `variant_candidates.jsonl` | カタログ不一致の残り（参考） |
| `estimates_rematched.jsonl` | 全推定＋再マッチ結果（参考・学習には不要） |

音源・Annotations は git に入れない（WSL `~/github/memo/`）。

## 学習への載せ方

### ローカル（推奨入口）

```powershell
cd colab_train
..\venv\Scripts\python.exe local_train_structure_prior.py
```

GPU を使うには **CUDA 版 torch** が必要（リポ venv が CPU のみだと `CUDA が見えません` と出る）。

### 手動 CLI / Colab

```powershell
..\venv\Scripts\python.exe scripts\train_structure_prior.py `
  --jsonl data/prior_pairs/manifests/accept.jsonl `
  --extra-jsonl `
    data/va_bpm_curated/structure_pairs_balanced_116.jsonl `
    data/memo2496_pairs/structure_pairs.jsonl `
    data/emopia_pairs/structure_pairs.jsonl `
  --checkpoint-dir checkpoints/structure_prior `
  --epochs 80 `
  --batch-size 32
```

- `makeData/progressions.py` に `proceed#####` 亜種入り（計759）。旧 prior ckpt は prog head 不一致のため **scratch 再学習**。
- `source=memo2496` は `EQUAL_SOURCES` 済み。
- Colab は `colab_train.ipynb` で `TRAIN_STRUCTURE_PRIOR=True`（Memo は extras に自動追加）。
