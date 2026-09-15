# Host-only environment propagation check; no CUDA/ggml links or measurements.
$ErrorActionPreference='Stop'
$names=@('GGML_CUDA_FORCE_MMQ','GGML_CUDA_FORCE_CUBLAS','CUDA_VISIBLE_DEVICES','LLAMA_TRACE_ANNOTATIONS','GGML_CUDA_DISABLE_FUSION','GGML_CUDA_CUBLAS_COMPUTE_TYPE')
$previous=@{};foreach($name in $names){$previous[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
try{
 foreach($name in $names){Set-Item -LiteralPath ('Env:'+$name) -Value 'sentinel';Remove-Item -LiteralPath ('Env:'+$name)}
 $cpp=& (Join-Path $PSScriptRoot 'environment-echo.exe') | ConvertFrom-Json
 $cppCode=$LASTEXITCODE
 $python=& 'E:/anaconda/python.exe' -c 'import os,json; keys=["GGML_CUDA_FORCE_MMQ","GGML_CUDA_FORCE_CUBLAS","CUDA_VISIBLE_DEVICES","LLAMA_TRACE_ANNOTATIONS","GGML_CUDA_DISABLE_FUSION","GGML_CUDA_CUBLAS_COMPUTE_TYPE"]; d={k:os.environ.get(k) for k in keys}; print(json.dumps({"removed_values":d,"all_absent":all(k not in os.environ for k in keys),"gpu_access":False})); raise SystemExit(0 if all(k not in os.environ for k in keys) else 1)' | ConvertFrom-Json
 $pythonCode=$LASTEXITCODE
}finally{foreach($name in $names){if($null -eq $previous[$name]){if(Test-Path -LiteralPath ('Env:'+$name)){Remove-Item -LiteralPath ('Env:'+$name)}}else{Set-Item -LiteralPath ('Env:'+$name) -Value $previous[$name]}}}
if($cppCode -ne 0 -or $pythonCode -ne 0 -or !$cpp.all_absent -or !$python.all_absent){throw 'Unset propagation failed'}
[pscustomobject]@{cpp=$cpp;python=$python;passed=$true;gpu_access=$false} | ConvertTo-Json -Depth 4
