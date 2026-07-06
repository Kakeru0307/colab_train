# colab_train — Google Colab 用 U-Net 学習

合成 MIDI + Guitar-TECHS を **Colab 上でパッチ化 → 二段階学習** し、checkpoint を Git 経由でローカルへ渡します。

## 含まれるもの

| パス | 内容 |
|---|---|
| `prepare_all.py` | **合成 + TECHS の一括パッチ化**（Colab 用メイン） |
| `makeData/` | 合成 MIDI 生成（1,000 本） |
| `data/raw/guitar-techs/` | 生 MIDI（98 曲） |
| `colab_train.ipynb` | 実行ノートブック |
| `checkpoints/` | 学習結果（Git push） |

**Git に含めない:** `data/pairs/`（Colab 上で `prepare_all.py` が生成）

## Colab での実行（推奨）

1. `colab_train.ipynb` を開く
2. `REPO_URL` / `GITHUB_TOKEN` を設定
3. セルを上から実行

### パッチ化だけ手動で行う場合

```bash
pip install -q -r requirements.txt
python prepare_all.py --synthetic-count 1000 --synthetic-seed 42
```

| 出力 | 内容 |
|---|---|
| `data/pairs/synthetic/` | 合成 ≈ 1,000 パッチ |
| `data/pairs/guitar-techs/` | TECHS ≈ 3,381 パッチ |
| `data/pairs/manifest_all.json` | サマリー |

### 学習（二段階）

```bash
# Stage 1: 合成
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
  --message "Stage1+2 complete"
```

## ローカルへ取り込み

```powershell
cd colab_train
git pull
.\scripts\import_to_prttype.ps1
```

`import_to_prttype.ps1` は `checkpoints/unet_last.pt` を想定しているため、Stage 2 利用時は手動で:

```powershell
Copy-Item checkpoints\stage2\unet_last.pt ..\prttype\checkpoints\guitar-techs\unet_last.pt
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
# 合成 MIDI を必ず再生成
python prepare_all.py --synthetic-count 1000 --force-regenerate

# 小規模テスト
python prepare_all.py --synthetic-count 10
```
