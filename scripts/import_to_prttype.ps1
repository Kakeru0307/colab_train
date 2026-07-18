# ローカル: colab_train のチェックポイントを prttype の役割フォルダへコピー
# 既定は backing。リード用: .\import_to_prttype.ps1 -Role lead
param(
    [ValidateSet("backing", "lead")]
    [string]$Role = "backing"
)

$RepoRoot = Split-Path $PSScriptRoot -Parent
$ProjectRoot = Split-Path $RepoRoot -Parent
$SrcCandidates = @(
    (Join-Path $RepoRoot "checkpoints\stage2\unet_last.pt"),
    (Join-Path $RepoRoot "checkpoints\backing\unet_last.pt"),
    (Join-Path $RepoRoot "checkpoints\unet_last.pt")
)
$Src = $SrcCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$DstDir = Join-Path $ProjectRoot "prttype\checkpoints\$Role"
$Dst = Join-Path $DstDir "unet_last.pt"

if (-not $Src) {
    Write-Error "チェックポイントが見つかりません。git pull 後、checkpoints/stage2/unet_last.pt を確認してください。"
    exit 1
}

New-Item -ItemType Directory -Force -Path $DstDir | Out-Null
Copy-Item $Src $Dst -Force
Write-Host "コピー完了: $Dst  (role=$Role)"
Write-Host "生成例:"
Write-Host "  cd prttype"
Write-Host "  python generate_backing.py --progression marusa --key C --bars 8 --bpm 100"
