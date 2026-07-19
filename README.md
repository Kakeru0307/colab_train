# colab_train — Google Colab 用 U-Net 学習

合成 MIDI + Guitar-TECHS を **Colab 上でパッチ化 → 二段階学習** し、checkpoint を Git 経由でローカルへ渡します。

## 学習タスク（骨格モード）

| モード | 入力（骨格） | 正解（target） |
|---|---|---|
| **`downbeat_chord`**（デフォルト） | 小節頭のコード onset | 音価付きフルパート |
| `root_per_bar` | 小節ごと根音 1 音 | 同上 |
| `melody_line` | 最上位声部のメロディ | 同上 |
| `onset_to_full` | 全発音点のみ（旧タスク） | 同上 |

Stage 1 / Stage 2 は **同じ `--mode`** でパッチ化してください。骨格からノート追加を学ばせるには `downbeat_chord` 等での **再学習が必要** です（旧 `onset_to_full` チェックポイントは流用不可）。

## 含まれるもの

| パス | 内容 |
|---|---|
| `prepare_all.py` | **合成 + TECHS の一括パッチ化**（Colab 用メイン） |
| `generate_lead_pairs.py` | リード学習用ペア生成 |
| `skeleton.py` | 骨格抽出（`downbeat_chord` 等） |
| `makeData/` | 合成 MIDI 生成（Colab 上で最大 6,000 本） |
| `data/raw/guitar-techs/` | 生 MIDI（98 曲） |
| `colab_train.ipynb` | 実行ノートブック |
| `checkpoints/` | 学習結果（Git push） |
| `scripts/push_checkpoint.py` | Colab から checkpoint を GitHub へ push |

**Git に含めない:** `data/pairs/`（**Colab 上でのみ** `prepare_all.py` が生成）

推論・DAW 確認用の `generate_*.py` / `inference.py` は本リポジトリには置かず、ローカル `prttype/` を使う。

## ローカル vs Colab の役割

| 作業 | どこでやるか |
|---|---|
| コード編集（`makeData/`, `skeleton.py` 等） | ローカル → `colab_train` を Git push |
| **合成 MIDI 生成** | **Colab**（`prepare_all.py` が自動実行） |
| **パッチ化（pairs）** | **Colab のみ**（ローカルで `prepare_dataset.py` は不要） |
| Stage 1 / 2 学習 | Colab |
| 推論・DAW 確認 | ローカル `prttype/` |

ローカルで `python -m makeData.generate` して試すことはできますが、**学習用 pairs は Colab で作ってください**（容量・GPU 環境の都合）。

## Colab での実行（推奨）

1. `colab_train.ipynb` を開く
2. `REPO_URL` / `GITHUB_TOKEN` を設定
3. セル 5 の `PAIR_MODE` を確認（既定: `downbeat_chord`）
4. セルを上から実行

### パッチ化だけ手動で行う場合

```bash
pip install -q -r requirements.txt
python prepare_all.py --synthetic-count 1000 --synthetic-seed 42 --mode downbeat_chord
```

| 出力 | 内容 |
|---|---|
| `data/pairs/synthetic/` | 合成パッチ |
| `data/pairs/guitar-techs/` | TECHS ≈ 3,381 パッチ |
| `data/pairs/manifest_all.json` | サマリー（`mode` フィールド付き） |

### 学習（二段階）

```bash
# Stage 1: 合成（骨格 → フル）
python train.py --pairs-dir data/pairs/synthetic \
  --epochs 20 --batch-size 16 \
  --checkpoint-dir checkpoints/stage1 --pos-weight 10

# Stage 2: Guitar-TECHS
python train.py --pairs-dir data/pairs/guitar-techs \
  --epochs 20 --batch-size 16 --lr 1e-5 \
  --checkpoint-dir checkpoints/stage2 \
  --resume checkpoints/stage1/unet_last.pt --pos-weight 10

python scripts/push_checkpoint.py \
  --ckpt checkpoints/stage2/unet_last.pt \
  --message "Stage1+2 downbeat_chord"
```

## ローカルへ取り込み（バッキング用）

Colab で学習・push したあと、**最終配置先は `prttype/checkpoints/`** です。
推論・生成スクリプト（`generate_*.py` 等）は本リポジトリには含めず、ローカルの `prttype/` 側で行います。

```powershell
cd colab_train
git pull
# Stage1（合成のみ）を backing に配置（通常）
Copy-Item checkpoints\stage1\unet_last.pt ..\prttype\checkpoints\backing\unet_last.pt -Force
# リード学習した場合
Copy-Item checkpoints\lead\unet_last.pt ..\prttype\checkpoints\lead\unet_last.pt -Force
```

| Colab 上（学習中） | ローカル最終配置 |
|---|---|
| `checkpoints/stage1/unet_last.pt` | → **`prttype/checkpoints/backing/unet_last.pt`**（推奨） |
| `checkpoints/stage2/unet_last.pt` | （通常は不使用。リズム劣化に注意） |
| `checkpoints/lead/unet_last.pt` | → `prttype/checkpoints/lead/unet_last.pt` |

ローカル生成は `prttype/` で:

```powershell
cd ..\prttype
python generate_backing.py --progression marusa --key C --bars 8 --bpm 100
python generate_lead.py    --progression marusa --key C --bars 8 --bpm 100
python generate_song.py --progression marusa --key C --bars 8 --bpm 100
```

## 初回 GitHub セットアップ

```powershell
cd colab_train
git init
git add .
git commit -m "Add Colab training with prepare_all"
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

## オプション

```bash
# 骨格モードを変更
python prepare_all.py --mode root_per_bar

# 旧タスク（発音点固定・音価付与のみ）
python prepare_all.py --mode onset_to_full

# 合成 MIDI を必ず再生成
python prepare_all.py --synthetic-count 1000 --force-regenerate

# 小規模テスト
python prepare_all.py --synthetic-count 10 --mode downbeat_chord
```
