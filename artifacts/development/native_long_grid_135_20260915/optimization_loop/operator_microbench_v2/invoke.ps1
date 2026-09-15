param(
    [ValidateSet('cpu','cuda')][string]$Device = 'cpu',
    [ValidateSet('F16','Q4_0','Q4_K','Q6_K','Q8_0','IQ3_S','IQ4_XS')][string]$Quant = 'F16',
    [long]$M = 1, [long]$N = 256, [long]$K = 256,
    [int]$Threads = 16, [int]$CudaIndex = 0,
    [int]$Warmup = 3, [int]$Repeats = 20, [int]$Samples = 32,
    [uint32]$Seed = 20260914, [double]$AbsoluteTolerance = 0.05, [double]$RelativeTolerance = 0.03,
    [string]$Output = '', [switch]$Run, [switch]$Nvtx, [int]$EvictMib = 0
)
$ErrorActionPreference = 'Stop'
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'build_manifest.json') -Raw | ConvertFrom-Json
foreach($ref in @($manifest.executable) + @($manifest.inputs) + @($manifest.dependencies)) {
    if (!(Test-Path -LiteralPath $ref.path -PathType Leaf)) { throw "Missing frozen input: $($ref.path)" }
    if ((Get-FileHash -LiteralPath $ref.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ref.sha256) { throw "Build input identity changed: $($ref.path)" }
}
$argsList = @('--device',$Device,'--m',"$M",'--n',"$N",'--k',"$K",'--quant',$Quant,'--threads',"$Threads",'--cuda-index',"$CudaIndex",'--warmup',"$Warmup",'--repeats',"$Repeats",'--samples',"$Samples",'--seed',"$Seed",'--atol',$AbsoluteTolerance.ToString([Globalization.CultureInfo]::InvariantCulture),'--rtol',$RelativeTolerance.ToString([Globalization.CultureInfo]::InvariantCulture))
$argsList += $(if($Run) {'--run'} else {'--check-only'})
if ($Nvtx) { $argsList += '--nvtx' }
$argsList += @('--evict-mib',"$EvictMib")
if ($Output) { $argsList += @('--output',$Output) }
$oldPath = $env:PATH
try {
    # Process-scoped loader path; no DLL is copied or replaced.
    $env:PATH = "$($manifest.native_bin);E:\cuda\bin;$oldPath"
    & $manifest.executable.path @argsList
    $code = $LASTEXITCODE
} finally { $env:PATH = $oldPath }
if ($code -ne 0) { throw "Microbench returned nonzero status $code; preserve its JSON failure record." }
