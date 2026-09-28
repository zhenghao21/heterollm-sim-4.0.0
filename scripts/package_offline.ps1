[CmdletBinding()]
param(
    [string]$OutputRoot = "",
    [switch]$SkipZip
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repoRoot "build\offline"
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)

$packageName = "heterollm-simulator-offline"
$staging = Join-Path $OutputRoot $packageName
$zipPath = Join-Path $OutputRoot ("{0}.zip" -f $packageName)

function Copy-Tree([string]$Source, [string]$Destination) {
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "找不到要打包的目录：$Source"
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
}

function Remove-TransientFiles([string]$Root) {
    if (-not (Test-Path -LiteralPath $Root)) { return }
    Get-ChildItem -LiteralPath $Root -Recurse -Force -Directory |
        Where-Object { $_.Name -in @("__pycache__", ".pytest_cache", ".mypy_cache") } |
        Sort-Object FullName -Descending |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath $Root -Recurse -Force -File |
        Where-Object { $_.Extension -in @(".pyc", ".pyo") } |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

if (Test-Path -LiteralPath $staging) {
    Remove-Item -LiteralPath $staging -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $staging | Out-Null
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

# The simulator is intentionally packaged as source plus a private CPython
# runtime.  This avoids relying on a Python installation or network access on
# the presentation computer.
$runtimeSource = Join-Path $env:USERPROFILE "AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none"
$venvPackages = Join-Path $repoRoot ".venv\Lib\site-packages"
if (-not (Test-Path -LiteralPath (Join-Path $runtimeSource "python.exe"))) {
    throw "未找到可复制的 Python 3.12 x64 运行时：$runtimeSource"
}
if (-not (Test-Path -LiteralPath $venvPackages)) {
    throw "未找到项目依赖：$venvPackages"
}

Copy-Tree $runtimeSource (Join-Path $staging "runtime")
New-Item -ItemType Directory -Force -Path (Join-Path $staging "runtime\Lib\site-packages") | Out-Null
Copy-Item -Path (Join-Path $venvPackages "*") -Destination (Join-Path $staging "runtime\Lib\site-packages") -Recurse -Force

# The editable-install marker points back to the build machine.  The launcher
# sets PYTHONPATH to the packaged source tree, so this marker must not remain.
Get-ChildItem -LiteralPath (Join-Path $staging "runtime\Lib\site-packages") -Force -File |
    Where-Object { $_.Name -like "__editable__.*.pth" } |
    Remove-Item -Force

Copy-Tree (Join-Path $repoRoot "src") (Join-Path $staging "app\src")
Copy-Tree (Join-Path $repoRoot "tools") (Join-Path $staging "app\tools")
Copy-Tree (Join-Path $repoRoot "configs") (Join-Path $staging "app\configs")
Copy-Tree (Join-Path $repoRoot "docs") (Join-Path $staging "app\docs")
Copy-Tree (Join-Path $repoRoot "tests") (Join-Path $staging "app\tests")
Copy-Item -LiteralPath (Join-Path $repoRoot "README.md") -Destination (Join-Path $staging "app\README.md") -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "pyproject.toml") -Destination (Join-Path $staging "app\pyproject.toml") -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "uv.lock") -Destination (Join-Path $staging "app\uv.lock") -Force

# Keep a small, deterministic evidence bundle for the live demonstration.  The
# complete development artifact tree contains multi-gigabyte native traces and
# is deliberately not copied into an offline presentation package.
$evidenceSource = Join-Path $repoRoot "artifacts\development\native_long_grid_135_20260915"
$evidenceDestination = Join-Path $staging "demo\evidence"
New-Item -ItemType Directory -Force -Path $evidenceDestination | Out-Null
foreach ($fileName in @(
    "stable_native_dataset.json",
    "hardware.json",
    "native_protocol.json",
    "prompts.json"
)) {
    $sourceFile = Join-Path $evidenceSource $fileName
    if (Test-Path -LiteralPath $sourceFile) {
        Copy-Item -LiteralPath $sourceFile -Destination $evidenceDestination -Force
    }
}
$r0Errors = Join-Path $evidenceSource "optimization_loop\round_000\on\errors.0001.json"
if (Test-Path -LiteralPath $r0Errors) {
    Copy-Item -LiteralPath $r0Errors -Destination (Join-Path $evidenceDestination "r0_errors.0001.json") -Force
}
$currentErrors = Join-Path $repoRoot ".tmp\native_compare_20260928_v2\errors.0001.json"
if (Test-Path -LiteralPath $currentErrors) {
    Copy-Item -LiteralPath $currentErrors -Destination (Join-Path $evidenceDestination "current_errors.0001.json") -Force
}

# Keep the bounded DeepSeek-V3 edge-assistant demonstration and its exact
# full-preset feasibility evidence with the offline package.  These are small
# JSON summaries, not native traces or model weights.
foreach ($fileName in @(
    "deepseek_v3_edge_short_scan.json",
    "deepseek_v3_edge_evidence.json",
    "deepseek_frontend_api_flow.json"
)) {
    $sourceFile = Join-Path $repoRoot ".tmp\$fileName"
    if (Test-Path -LiteralPath $sourceFile) {
        Copy-Item -LiteralPath $sourceFile -Destination (Join-Path $evidenceDestination $fileName) -Force
    }
}

Copy-Item -LiteralPath (Join-Path $PSScriptRoot "start_offline.ps1") -Destination (Join-Path $staging "Start-Simulator.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "start_offline.cmd") -Destination (Join-Path $staging "Start-Simulator.cmd") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "check_offline.ps1") -Destination (Join-Path $staging "Check-Offline.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "OFFLINE_README_ZH.md") -Destination (Join-Path $staging "README_OFFLINE_ZH.md") -Force

Remove-TransientFiles $staging

$manifest = [ordered]@{
    package = $packageName
    simulator_version = "4.0.0"
    python = "3.12.x x64 (private runtime)"
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    source = "heterollm-sim-4.0.0"
    network_required = $false
    entrypoint = "Start-Simulator.cmd"
    health_check = "Check-Offline.ps1"
    evidence = "demo/evidence"
    note = "开发 artifacts 中的大型 native trace 未打包；演示所需的冻结数据已包含。"
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $staging "OFFLINE_MANIFEST.json") -Encoding UTF8

if (-not $SkipZip) {
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path $staging -DestinationPath $zipPath -CompressionLevel Optimal
}

$stagingBytes = (Get-ChildItem -LiteralPath $staging -Recurse -Force -File | Measure-Object Length -Sum).Sum
$zipBytes = if (Test-Path -LiteralPath $zipPath) { (Get-Item -LiteralPath $zipPath).Length } else { 0 }
Write-Output ("离线包目录：{0}" -f $staging)
if ($zipBytes -gt 0) {
    Write-Output ("离线 ZIP：{0} ({1:N1} MB)" -f $zipPath, ($zipBytes / 1MB))
}
Write-Output ("解压后大小：{0:N1} MB" -f ($stagingBytes / 1MB))

