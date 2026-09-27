[CmdletBinding()]
param(
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
$python = Join-Path $root "runtime\python.exe"
$source = Join-Path $root "app\src"
$env:PYTHONPATH = $source

& $python -c "import heterollm_sim, numpy, ortools; print('simulator=' + heterollm_sim.__version__); print('numpy=' + numpy.__version__); print('ortools=' + ortools.__version__)"
if ($LASTEXITCODE -ne 0) { throw "离线 Python 依赖检查失败。" }

$healthUrl = "http://127.0.0.1:{0}/api/health" -f $Port
try {
    $response = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
    $response | ConvertTo-Json -Depth 5
    Write-Host "健康检查通过：$healthUrl"
} catch {
    Write-Host "当前没有运行中的 UI 服务，依赖检查已通过。启动 Start-Simulator.cmd 后可再次执行本脚本。"
}

