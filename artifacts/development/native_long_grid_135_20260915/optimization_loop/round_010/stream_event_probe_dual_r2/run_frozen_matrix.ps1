param([switch]$IdleWindowConfirmed)
$ErrorActionPreference='Stop'
if(!$IdleWindowConfirmed){throw 'An independently arranged idle window must be confirmed'}
$knownBusy=@(Get-CimInstance Win32_Process | Where-Object { ($_.Name -match 'python|llama|stream-event-probe') -and ($_.CommandLine -match 'predict_stable_native_dataset|hash_read_diagnostic|llama-server|llama-cli|llama-bench|stream-event-probe.exe') })
if($knownBusy.Count){throw 'A known measurement, hash diagnostic, or LLM process is active; no process was stopped'}
$protocol=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'protocol.json') -Raw | ConvertFrom-Json
$stamp=[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$resultDir=Join-Path $PSScriptRoot "runs/$stamp"
if(Test-Path -LiteralPath $resultDir){throw 'Run directory already exists'}
New-Item -ItemType Directory -Path $resultDir | Out-Null
$initialIdentity=& (Join-Path $PSScriptRoot 'invoke.ps1')
$initialIdentity | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resultDir 'identity_before.json') -Encoding utf8
Get-CimInstance Win32_Process | Select-Object Name,ProcessId | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $resultDir 'processes_before.json') -Encoding utf8
$driver='C:/Windows/System32/nvidia-smi.exe'
$query='--query-gpu=uuid,pci.bus_id,driver_version,name,clocks.current.sm,clocks.current.memory,utilization.gpu,temperature.gpu'
& $driver $query --format=csv,noheader,nounits | Set-Content -LiteralPath (Join-Path $resultDir 'device_before.csv') -Encoding utf8
if($LASTEXITCODE -ne 0){throw 'Pre-run device snapshot failed'}
& $driver '--query-compute-apps=pid,process_name' --format=csv,noheader,nounits | Set-Content -LiteralPath (Join-Path $resultDir 'compute_processes_before.csv') -Encoding utf8
if($LASTEXITCODE -ne 0){throw 'Compute-process snapshot failed'}
$computeLines=@(Get-Content -LiteralPath (Join-Path $resultDir 'compute_processes_before.csv') | Where-Object {$_.Trim()})
$backgroundGraphicsPresent=$computeLines.Count -gt 0
[pscustomobject]@{background_graphics_present=$backgroundGraphicsPresent;listed_process_count=$computeLines.Count;exclusive_gpu_access_verified=$false;interpretation='WDDM can list desktop graphics processes in compute-apps; retain evidence, do not kill processes, rely on frozen variability gates'} | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $resultDir 'background_load.json') -Encoding utf8
$rows=@();$pairs=@();$position=0
foreach($config in $protocol.configs){
 $modes=if(($position%2)-eq 0){@('event','control')}else{@('control','event')}
 foreach($mode in $modes){
  $output=Join-Path $resultDir "$($config.id).$mode.json";$log=Join-Path $resultDir "$($config.id).$mode.log"
  $start=[DateTime]::UtcNow.ToString('o');$errorText=$null
  try{& (Join-Path $PSScriptRoot 'invoke.ps1') -Config $config.id -Mode $mode -Output $output -Run *> $log}catch{$errorText=$_.Exception.Message;$errorText | Add-Content -LiteralPath $log -Encoding utf8}
  $rows+=[pscustomobject]@{config=$config.id;mode=$mode;output=$output;started_utc=$start;finished_utc=[DateTime]::UtcNow.ToString('o');succeeded=($null -eq $errorText);error=$errorText;raw_exists=(Test-Path -LiteralPath $output)}
 }
 $assessment=Join-Path $resultDir "$($config.id).assessment.json"
 & 'E:/anaconda/python.exe' (Join-Path $PSScriptRoot 'assess.py') --config $config.id --event (Join-Path $resultDir "$($config.id).event.json") --control (Join-Path $resultDir "$($config.id).control.json") --output $assessment *> (Join-Path $resultDir "$($config.id).assessment.log")
 $pairs+=[pscustomobject]@{config=$config.id;path=$assessment;exit_code=$LASTEXITCODE;exists=(Test-Path -LiteralPath $assessment)}
 Write-Output "$($position+1)/$(@($protocol.configs).Count) $($config.id) assessment status=$LASTEXITCODE"
 $position++
}
& $driver $query --format=csv,noheader,nounits | Set-Content -LiteralPath (Join-Path $resultDir 'device_after.csv') -Encoding utf8
$deviceAfterExit=$LASTEXITCODE
$postError=$null;$finalIdentity=$null
try{$finalIdentity=& (Join-Path $PSScriptRoot 'invoke.ps1');$finalIdentity | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resultDir 'identity_after.json') -Encoding utf8}catch{$postError=$_.Exception.Message}
& 'E:/anaconda/python.exe' (Join-Path $PSScriptRoot 'full_raw_audit.py') $resultDir *> (Join-Path $resultDir 'full_raw_audit.log')
$auditExit=$LASTEXITCODE
[pscustomobject]@{schema='stream-probe-matrix-execution/v2';background_graphics_present=$backgroundGraphicsPresent;exclusive_gpu_access_verified=$false;runs=$rows;assessments=$pairs;device_after_exit=$deviceAfterExit;post_identity_error=$postError;identity_unchanged=($null -ne $finalIdentity -and $initialIdentity.manifest_sha256 -ceq $finalIdentity.manifest_sha256);full_raw_audit_exit=$auditExit;finished_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $resultDir 'execution.json') -Encoding utf8
& 'E:/anaconda/python.exe' (Join-Path $PSScriptRoot 'summarize_matrix.py') $resultDir
$summaryExit=$LASTEXITCODE
Write-Output $resultDir
if($summaryExit -ne 0){throw 'Matrix quality rejection or incomplete evidence retained; no coefficient generated'}
