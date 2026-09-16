param(
 [Parameter(Mandatory=$true)][string]$Config,
 [ValidateRange(1,3)][int]$Pair=1,
 [ValidateSet('direct','profile')][string]$Arm='direct',
 [string]$Output='',
 [switch]$Run,
 [switch]$RootReviewed,
 [switch]$IdleWindowConfirmed
)
$ErrorActionPreference='Stop'
$protocolPath=Join-Path $PSScriptRoot 'protocol.json'
$protocol=Get-Content -LiteralPath $protocolPath -Raw | ConvertFrom-Json
$manifestPath=Join-Path $PSScriptRoot 'build_manifest.json'
$chosen=@($protocol.configs | Where-Object {$_.id -ceq $Config})
if($chosen.Count -ne 1){throw 'Configuration is not one of the six reviewed combinations'}
if(!(Test-Path -LiteralPath $manifestPath -PathType Leaf)){
 if($Run){throw 'Probe is source preparation only: parent review and a compiled freeze are required before GPU access'}
 [pscustomobject]@{status='source_prepared_not_compiled';config=$Config;pair=$Pair;arm=$Arm;gpu_access=$false};return
}
$manifest=Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
function Assert-Frozen {
 if($manifest.schema -cne 'graph-submit-probe-build/v1' -or $manifest.status -cne 'compiled_host_tested_not_gpu_executed' -or @($manifest.files).Count -lt 50 -or $manifest.host_reference.pass -ne $true){throw 'Missing or incomplete compiled probe freeze'}
 $paths=@{}
 foreach($r in @($manifest.files)){
  if(!$r.path -or !$r.sha256 -or $r.sha256 -cnotmatch '^[a-f0-9]{64}$' -or $null -eq $r.bytes -or $r.bytes -le 0 -or !(Test-Path -LiteralPath $r.path -PathType Leaf)){throw 'Invalid required identity'}
  if($paths.ContainsKey($r.path)){throw 'Duplicate frozen identity'};$paths[$r.path]=$r
  if((Get-Item -LiteralPath $r.path).Length -ne $r.bytes -or (Get-FileHash -LiteralPath $r.path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $r.sha256){throw "Frozen file changed: $($r.path)"}
 }
 foreach($name in @('protocol.json','source_provenance.json','graph_submit_probe.cpp','math_reference.h','prepared_identity.h','frozen_module_guard.h','invoke.ps1','build_probe.py','reference_check.py','graph-submit-probe.exe','host_reference_validation.json')){
  if(!$paths.ContainsKey((Join-Path $PSScriptRoot $name))){throw "Necessary frozen input missing: $name"}
 }
 foreach($entry in @($manifest.executable,$manifest.profiler,$manifest.protocol)){
  if(!$entry.path -or $entry.sha256 -cnotmatch '^[a-f0-9]{64}$' -or $null -eq $entry.bytes -or !$paths.ContainsKey($entry.path)){throw 'Incomplete executable/profiler/protocol identity'}
  $matching=$paths[$entry.path]
  if($entry.sha256 -cne $matching.sha256 -or $entry.bytes -ne $matching.bytes){throw 'Metadata-to-frozen-file identity mismatch'}
 }
 if($manifest.executable.path -cne (Join-Path $PSScriptRoot 'graph-submit-probe.exe') -or $manifest.protocol.path -cne $protocolPath -or $manifest.profiler.path -cne $protocol.profiler.path){throw 'Unexpected executable/profiler/protocol path'}
}
Assert-Frozen
if(!$Run){[pscustomobject]@{status='identity_verified_not_executed';config=$Config;pair=$Pair;arm=$Arm;gpu_access=$false};return}
if(!$RootReviewed -or !$IdleWindowConfirmed){throw 'Parent source review and coordinated idle window are required'}
if(!$Output){throw 'Explicit new raw output path required'}
$Output=[IO.Path]::GetFullPath($Output)
if(!(Test-Path -LiteralPath (Split-Path -Parent $Output) -PathType Container)){throw 'Collector must create the intended output directory first'}
$receiptPath=$Output+'.receipt.json';$stdoutPath=$Output+'.stdout.log';$stderrPath=$Output+'.stderr.log'
$prefix=$Output+'.trace'
foreach($file in @($Output,$receiptPath,$stdoutPath,$stderrPath,($prefix+'.nsys-rep'),($prefix+'.sqlite'),($prefix+'.qdstrm'))){if(Test-Path -LiteralPath $file){throw "Refusing to overwrite prior evidence: $file"}}
function New-TextFile([string]$Path,[string]$Content){
 $stream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
 try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes($Content);$stream.Write($bytes,0,$bytes.Length);$stream.Flush()}finally{$stream.Dispose()}
}
function Save-Receipt {
 $temporary=$receiptPath+'.'+[Guid]::NewGuid().ToString('N')+'.tmp'
 New-TextFile $temporary ($script:receipt | ConvertTo-Json -Depth 20)
 [IO.File]::Replace($temporary,$receiptPath,$null)
}
function Get-GpuTelemetry {
 $tool=Get-Command 'nvidia-smi.exe' -ErrorAction Stop | Select-Object -First 1
 $lines=@(& $tool.Source '--query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used' '--format=csv,noheader,nounits' 2>&1)
 $code=$LASTEXITCODE
 if($code -ne 0){throw "Actual GPU telemetry unavailable, status $code"}
 $rows=@(($lines -join "`n") | ConvertFrom-Csv -Header index,uuid,name,driver,sm_mhz,memory_mhz,temperature_c,power_w,utilization_pct,memory_used_mib)
 $device=@($rows | Where-Object {$_.index.Trim() -ceq '0'})
 if($device.Count -ne 1){throw 'Actual device 0 fingerprint missing'}
 $d=$device[0]
 if($d.uuid.Trim() -cne $protocol.runtime.gpu_expected.uuid -or $d.name.Trim() -cne $protocol.runtime.gpu_expected.name -or $d.driver.Trim() -cne $protocol.runtime.gpu_expected.driver){throw 'Actual GPU/driver does not match fixed native runtime'}
 [pscustomobject]@{utc=[DateTime]::UtcNow.ToString('o');tool=$tool.Source;tool_sha256=(Get-FileHash -LiteralPath $tool.Source -Algorithm SHA256).Hash.ToLowerInvariant();raw=$lines;devices=$rows;clock_policy='record actual; no automatic clock/driver changes'}
}
# Each argv element is fixed by this protocol or an explicit path. Use C-runtime
# quoting for spaces/quotes; do not interpolate into cmd.exe or another shell.
function Quote-Arg([string]$Text){
 if($Text -notmatch '[\s"]' -and $Text.Length -gt 0){return $Text}
 $escaped=[regex]::Replace($Text,'(\\*)"','$1$1\"')
 $escaped=[regex]::Replace($escaped,'(\\+)$','$1$1')
 return '"'+$escaped+'"'
}
$pairId=$Config+'.pair'+$Pair
$targetArgs=@('--run','--config',$Config,'--pair-id',$pairId,'--output',$Output)
$executable=$manifest.executable.path
$launchArgs=$targetArgs
if($Arm -ceq 'profile'){$executable=$manifest.profiler.path;$launchArgs=@($protocol.profiler.options)+@("--output=$prefix",$manifest.executable.path)+$targetArgs}
$receipt=[ordered]@{schema='graph-submit-launch-receipt/v1';status='prepared';config=$Config;pair=$Pair;pair_id=$pairId;arm=$Arm;created_utc=[DateTime]::UtcNow.ToString('o');manifest_sha256=(Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant();executable=$executable;argv=$launchArgs;target_argv=$targetArgs;output=$Output;pid=$null;raw_expected=$Output;trace_expected=$(if($Arm -ceq 'profile'){$prefix+'.nsys-rep'}else{$null});gpu_before=$null;gpu_after=$null;exit_code=$null;post_identity_pass=$false;child_terminated=$false}
New-TextFile $receiptPath ($receipt | ConvertTo-Json -Depth 20)
$variables=@('PATH')+@($protocol.runtime.environment.PSObject.Properties.Name)+@($protocol.runtime.extra_clear_environment)
$previous=@{};foreach($name in $variables){$previous[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
try{
 $receipt.gpu_before=Get-GpuTelemetry
 [Environment]::SetEnvironmentVariable('PATH',"$($manifest.native_bin);E:\cuda\bin;$($previous['PATH'])",'Process')
 foreach($property in $protocol.runtime.environment.PSObject.Properties){[Environment]::SetEnvironmentVariable($property.Name,$property.Value,'Process')}
 foreach($name in $protocol.runtime.extra_clear_environment){[Environment]::SetEnvironmentVariable($name,$null,'Process')}
 $argLine=(@($launchArgs | ForEach-Object {Quote-Arg ([string]$_)}) -join ' ')
 $process=Start-Process -FilePath $executable -ArgumentList $argLine -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
 $receipt.pid=$process.Id;$receipt.status='running';$receipt.started_utc=[DateTime]::UtcNow.ToString('o');Save-Receipt
 $deadline=[DateTime]::UtcNow.AddSeconds([int]$protocol.execution.timeout_seconds_per_process)
 while(!$process.WaitForExit(1000)){
  if([DateTime]::UtcNow -gt $deadline){$receipt.status='still_running_timeout';$receipt.timeout_utc=[DateTime]::UtcNow.ToString('o');Save-Receipt;throw 'Timeout retained: child/profiler still running. Stop scheduling and coordinate with parent; do not kill.'}
 }
 $process.Refresh();$receipt.exit_code=$process.ExitCode;$receipt.completed_utc=[DateTime]::UtcNow.ToString('o');$receipt.gpu_after=Get-GpuTelemetry
 Assert-Frozen;$receipt.post_identity_pass=$true
 if($process.ExitCode -ne 0){$receipt.status='failed';Save-Receipt;throw "Probe failed $($process.ExitCode); preserve all evidence"}
 if(!(Test-Path -LiteralPath $Output -PathType Leaf)){throw 'Expected raw JSONL missing'}
 if($Arm -ceq 'profile' -and !(Test-Path -LiteralPath ($prefix+'.nsys-rep') -PathType Leaf)){throw 'Expected native trace missing'}
 $receipt.raw_sha256=(Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash.ToLowerInvariant();$receipt.status='complete_not_calibration_ready';Save-Receipt
} catch {
 if($receipt.status -cne 'still_running_timeout'){$receipt.status='failed'}
 $receipt.error=$_.Exception.Message;Save-Receipt;throw
} finally {foreach($name in $variables){[Environment]::SetEnvironmentVariable($name,$previous[$name],'Process')}}
[pscustomobject]@{status=$receipt.status;config=$Config;pair=$Pair;arm=$Arm;raw=$Output;receipt=$receiptPath;calibration_ready=$false;trace_confirmation_required=$true}
