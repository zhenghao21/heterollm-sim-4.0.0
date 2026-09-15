param(
 [string]$Config='dev_Q5_0_m1_k896',
 [ValidateSet('event','control')][string]$Mode='event',
 [string]$Output='',
 [switch]$Run
)
$ErrorActionPreference='Stop'
$manifestPath=Join-Path $PSScriptRoot 'build_manifest.json'
$manifest=Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$protocol=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'protocol.json') -Raw | ConvertFrom-Json
if($manifest.schema -cne 'stream-event-probe-build/v2' -or @($manifest.files).Count -eq 0){throw 'Missing/non-v2 identity manifest'}
foreach($ref in @($manifest.files)){
 if(!$ref.path -or !$ref.sha256 -or $ref.sha256 -cnotmatch '^[0-9a-f]{64}$'){throw 'Empty/invalid frozen identity'}
 if(!(Test-Path -LiteralPath $ref.path -PathType Leaf)){throw "Missing frozen input: $($ref.path)"}
 $item=Get-Item -LiteralPath $ref.path
 if($item.Length -ne $ref.bytes -or (Get-FileHash -LiteralPath $ref.path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $ref.sha256){throw "Frozen input mismatch: $($ref.path)"}
}
$required=@('source_reference.h','stream_event_probe.cpp','stream-event-probe.exe','protocol.json','invoke.ps1','run_frozen_matrix.ps1','assess.py','full_raw_audit.py','summarize_matrix.py','identity_lock.h','identity_lock.json')
foreach($name in $required){if(@($manifest.files | Where-Object { $_.path -ceq (Join-Path $PSScriptRoot $name) }).Count -ne 1){throw "Required input not frozen: $name"}}
if(!$manifest.executable.path -or @($manifest.files | Where-Object { $_.path -ceq $manifest.executable.path -and $_.sha256 -ceq $manifest.executable.sha256 }).Count -ne 1){throw 'Executable identity absent'}
$chosen=@($protocol.configs | Where-Object {$_.id -ceq $Config})
if($chosen.Count -ne 1){throw 'Configuration is not in the frozen 12-configuration protocol'}
if(!$Run){
 [pscustomobject]@{status='identity_verified_not_executed';config=$Config;mode=$Mode;device_access=$false;graph_compute_calls=0;frozen_files=@($manifest.files).Count;manifest_sha256=(Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()}
 return
}
if(!$Output){throw 'An explicit new output path is required'}
if(Test-Path -LiteralPath $Output){throw 'Refusing to overwrite prior output'}
$c=$chosen[0]
$argv=@('--run','--device','cuda','--quant',$c.quant,'--m',"$($c.M)",'--n',"$($c.N)",'--k',"$($c.K)",'--batch','1024','--threads','1','--cuda-index','0','--warmup','5','--repeats','30','--samples','4096','--seed','20260914','--atol','0.05','--rtol','0.03','--output',$Output)
if($Mode -eq 'control'){$argv+='--control'}
$names=@('PATH','GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_FORCE_MMQ','GGML_CUDA_FORCE_CUBLAS','CUDA_VISIBLE_DEVICES','LLAMA_TRACE_ANNOTATIONS','GGML_CUDA_DISABLE_FUSION','GGML_CUDA_CUBLAS_COMPUTE_TYPE')
$previous=@{};foreach($name in $names){$previous[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
try{
 [Environment]::SetEnvironmentVariable('PATH',"$($manifest.native_bin);E:\cuda\bin;$($previous['PATH'])",'Process')
 [Environment]::SetEnvironmentVariable('GGML_CUDA_DISABLE_GRAPHS','1','Process')
 foreach($name in $names | Select-Object -Skip 2){if(Test-Path -LiteralPath ('Env:'+$name)){Remove-Item -LiteralPath ('Env:'+$name)}}
 & $manifest.executable.path @argv
 $code=$LASTEXITCODE
}finally{foreach($name in $names){if($null -eq $previous[$name]){if(Test-Path -LiteralPath ('Env:'+$name)){Remove-Item -LiteralPath ('Env:'+$name)}}else{Set-Item -LiteralPath ('Env:'+$name) -Value $previous[$name]}}}
if($code -ne 0){throw "Probe failed, status $code; preserve failure evidence"}
