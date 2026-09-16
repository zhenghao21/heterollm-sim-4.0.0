param(
 [Parameter(Mandatory=$true)][string]$Config,
 [ValidateSet('event','control')][string]$Mode='event',
 [string]$Output='',
 [switch]$Run,
 [switch]$Profile,
 [switch]$RootReviewed,
 [switch]$IdleWindowConfirmed
)
$ErrorActionPreference='Stop'
$manifestPath=Join-Path $PSScriptRoot 'build_manifest.json'
$manifest=Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$protocol=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'protocol.json') -Raw | ConvertFrom-Json
function Assert-Frozen {
 if($manifest.schema -cne 'operator-surface-probe-build/v1' -or @($manifest.files).Count -lt 10){throw 'Missing or invalid frozen identity'}
 foreach($ref in @($manifest.files)){
  if(!$ref.path -or !$ref.sha256 -or $ref.sha256 -notmatch '^[a-f0-9]{64}$' -or $null -eq $ref.bytes -or !(Test-Path -LiteralPath $ref.path -PathType Leaf)){throw 'Incomplete frozen identity'}
  if((Get-Item -LiteralPath $ref.path).Length -ne $ref.bytes -or (Get-FileHash -LiteralPath $ref.path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $ref.sha256){throw "Frozen input mismatch: $($ref.path)"}
 }
 foreach($name in @('protocol.json','stream_event_probe.cpp','source_reference.h','identity_lock.h','identity_lock.json','invoke.ps1','assess.py','full_raw_audit.py','operator-surface-probe.exe')){
  if(@($manifest.files | Where-Object {$_.path -ceq (Join-Path $PSScriptRoot $name)}).Count -ne 1){throw "Required frozen input missing: $name"}
 }
}
Assert-Frozen
$chosen=@($protocol.configs | Where-Object {$_.id -ceq $Config})
if($chosen.Count -ne 1){throw 'Configuration outside frozen26'}
if(!$Run){[pscustomobject]@{status='identity_verified_not_executed';config=$Config;mode=$Mode;device_access=$false;manifest_sha256=(Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()};return}
if(!$RootReviewed -or !$IdleWindowConfirmed){throw 'Root review and idle window required before GPU access'}
if(!$Output -or (Test-Path -LiteralPath $Output)){throw 'An explicit new output path is required'}
if($Profile -and $Mode -ne 'event'){throw 'Profile the same event mode as direct'}
$c=$chosen[0]
$argv=@('--run','--device','cuda','--quant',$c.quant,'--m',"$($c.M)",'--n',"$($c.N)",'--k',"$($c.K)",'--batch','1','--threads','1','--cuda-index','0','--warmup','5','--repeats','30','--samples','4096','--seed','20260914','--atol','0.05','--rtol','0.03','--evict-mib','128','--nvtx','--output',$Output)
if($Mode -eq 'control'){$argv+='--control'}
$names=@('PATH','GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_FORCE_MMQ','GGML_CUDA_FORCE_CUBLAS','CUDA_VISIBLE_DEVICES','LLAMA_TRACE_ANNOTATIONS','GGML_CUDA_DISABLE_FUSION','GGML_CUDA_CUBLAS_COMPUTE_TYPE','GGML_OP_OFFLOAD_MIN_BATCH','GGML_SCHED_DEBUG','OMP_NUM_THREADS','GGML_NO_IQ_PANEL','GGML_CPU_DISABLE_FUSION')
$previous=@{};foreach($name in $names){$previous[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
try{
 [Environment]::SetEnvironmentVariable('PATH',"$($manifest.native_bin);E:\cuda\bin;$($previous['PATH'])",'Process')
 [Environment]::SetEnvironmentVariable('GGML_CUDA_DISABLE_GRAPHS','1','Process')
 foreach($name in $names | Select-Object -Skip 2){[Environment]::SetEnvironmentVariable($name,$null,'Process')}
 if($Profile){
  $prefix=$Output+'.trace'
  if(Test-Path -LiteralPath ($prefix+'.nsys-rep')){throw 'Trace output already exists'}
  & $manifest.profiler.path profile --kill=false --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=false --output=$prefix $manifest.executable.path @argv
 }else{& $manifest.executable.path @argv}
 $code=$LASTEXITCODE
}finally{foreach($name in $names){[Environment]::SetEnvironmentVariable($name,$previous[$name],'Process')}}
Assert-Frozen
if($code -ne 0){throw "Probe failed status $code; keep all failure evidence"}
