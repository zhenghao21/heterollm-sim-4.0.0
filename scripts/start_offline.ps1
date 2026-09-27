[CmdletBinding()]
param(
    [int]$Port = 8765,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
$python = Join-Path $root "runtime\python.exe"
$source = Join-Path $root "app\src"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "找不到离线 Python 运行时：$python"
}
if (-not (Test-Path -LiteralPath (Join-Path $source "heterollm_sim"))) {
    throw "找不到仿真器源码：$source"
}

$env:PYTHONPATH = $source
$arguments = @(
    "-m", "heterollm_sim.cli", "ui",
    "--host", "127.0.0.1",
    "--port", $Port.ToString()
)
if ($NoBrowser) {
    $arguments += "--no-browser"
}
Write-Host "HeteroLLM Simulator 正在启动：http://127.0.0.1:$Port/"
Write-Host "关闭窗口或按 Ctrl+C 停止。"
& $python @arguments
exit $LASTEXITCODE

