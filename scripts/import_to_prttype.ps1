# ローカル: colab_train のチェックポイントを prttype へコピー
$RepoRoot = Split-Path $PSScriptRoot -Parent
$ProjectRoot = Split-Path $RepoRoot -Parent
$SrcCandidates = @(
    (Join-Path $RepoRoot "checkpoints\stage2\unet_last.pt"),
    (Join-Path $RepoRoot "checkpoints\unet_last.pt")
)
$Src = $SrcCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$DstDir = Join-Path $ProjectRoot "prttype\checkpoints\guitar-techs"
$Dst = Join-Path $DstDir "unet_last.pt"

if (-not $Src) {
    Write-Error "チェックポイントが見つかりません。git pull 後、checkpoints/stage2/unet_last.pt を確認してください。"
}

New-Item -ItemType Directory -Force -Path $DstDir | Out-Null
Copy-Item $Src $Dst -Force
Write-Host "コピー完了: $Dst"
Write-Host "推論例:"
Write-Host "  cd prttype"
Write-Host "  python inference.py --midi midi/test1.mid --checkpoint checkpoints/guitar-techs/unet_last.pt --output data/test1_generated.mid"
