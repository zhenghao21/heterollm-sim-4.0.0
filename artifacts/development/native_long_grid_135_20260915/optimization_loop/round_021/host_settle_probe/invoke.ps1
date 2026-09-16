# Review-gated launcher. It neither compiles nor changes clocks, affinity, priority, driver, or GPU state.
[CmdletBinding(DefaultParameterSetName='identity')]
param(
 [Parameter(ParameterSetName='identity',Mandatory=$true)][switch]$IdentityCheck,
 [Parameter(ParameterSetName='run',Mandatory=$true)][switch]$Run,
 [Parameter(ParameterSetName='run',Mandatory=$true)][ValidateSet('buffered')][string]$Arm,
 [Parameter(ParameterSetName='run',Mandatory=$true)][ValidateSet(0,1000)][int]$SettleMs,
 [Parameter(ParameterSetName='run',Mandatory=$true)][ValidateSet('direct','profile')][string]$Mode,
 [Parameter(ParameterSetName='run',Mandatory=$true)][ValidateRange(1,3)][int]$Pair,
 [Parameter(ParameterSetName='run',Mandatory=$true)][string]$Output
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$PSScriptRoot=$PSScriptRoot
$protocol=Get-Content -Raw (Join-Path $PSScriptRoot 'protocol.json') | ConvertFrom-Json
$manifestPath=Join-Path $PSScriptRoot 'build_manifest.json'
if(!(Test-Path -LiteralPath $manifestPath -PathType Leaf)){throw 'No reviewed build manifest. Compile is a separate parent-gated step.'}
$manifest=Get-Content -Raw $manifestPath | ConvertFrom-Json
if($manifest.status -cne 'compiled_host_tested_not_gpu_executed' -or $manifest.gpu_access -ne $false){throw 'Build manifest is not a clean non-GPU compile/host-test receipt'}
function Assert-Frozen {
 foreach($item in $manifest.files){
  if(!(Test-Path -LiteralPath $item.path -PathType Leaf)){throw "Frozen file missing: $($item.path)"}
  $hash=(Get-FileHash -LiteralPath $item.path -Algorithm SHA256).Hash.ToLowerInvariant()
  if($hash -cne $item.sha256){throw "Frozen file changed: $($item.path)"}
 }
}
function New-ExclusiveTextFile([string]$Path,[string]$Text){
 $stream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
 try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes($Text);$stream.Write($bytes,0,$bytes.Length);$stream.Flush($true)}finally{$stream.Dispose()}
}
function Quote-Arg([string]$Text){
 if($Text -notmatch '[\s"]' -and $Text.Length -gt 0){return $Text}
 $escaped=[regex]::Replace($Text,'(\\*)"','$1$1\"')
 $escaped=[regex]::Replace($escaped,'(\\+)$','$1$1')
 return '"'+$escaped+'"'
}
Assert-Frozen
$variables=@('PATH')+@($protocol.runtime.environment.PSObject.Properties.Name)+@($protocol.runtime.extra_clear_environment)
$previous=@{};foreach($name in $variables){$previous[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
try {
 [Environment]::SetEnvironmentVariable('PATH',"$($manifest.native_bin);E:\cuda\bin;$($previous['PATH'])",'Process')
 foreach($property in $protocol.runtime.environment.PSObject.Properties){[Environment]::SetEnvironmentVariable($property.Name,$property.Value,'Process')}
 foreach($name in $protocol.runtime.extra_clear_environment){[Environment]::SetEnvironmentVariable($name,$null,'Process')}
 if($IdentityCheck){
  $result=& $manifest.executable.path '--identity-check' 2>&1
  if($LASTEXITCODE -ne 0){throw "Identity check failed: $result"}
  $result
  return
 }
 # Root defines the formal order. This launcher only enforces the two preregistered buffered settle conditions.
 $config=$protocol.config.id
 if($config -cne 'scale_f32_e262144_g8' -or $Pair -gt [int]$protocol.execution.pairs_per_condition){throw 'Run is outside the reviewed R21 settle scope'}
 $condition=if($SettleMs -eq 0){'control'}else{'treatment'}
 if(@($protocol.conditions | Where-Object {$_.settle_ms -eq $SettleMs -and $_.id -eq $condition}).Count -ne 1){throw 'Protocol settle condition mismatch'}
 $pairId="r21_host_settle.$condition.pair$Pair"
 $targetArgs=@('--run','--config',$config,'--arm',$Arm,'--settle-ms',[string]$SettleMs,'--pair-id',$pairId,'--output',$Output)
 $prefix=[IO.Path]::Combine([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($Output)),[IO.Path]::GetFileNameWithoutExtension($Output)+'.'+$Mode)
 $receiptPath=$Output+'.launch-receipt.json'
 $stdoutPath=$Output+'.stdout.log';$stderrPath=$Output+'.stderr.log'
 $launch=@($targetArgs);$executable=$manifest.executable.path
 if($Mode -ceq 'profile'){$executable=$manifest.profiler.path;$launch=@($protocol.profiler.profile_options)+@("--output=$prefix",$manifest.executable.path)+$targetArgs}
 $receipt=[ordered]@{schema='host-settle-launch-receipt/v1';status='prepared';condition=$condition;settle_ms=$SettleMs;config=$config;arm=$Arm;pair=$Pair;mode=$Mode;created_utc=[DateTime]::UtcNow.ToString('o');manifest_sha256=(Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant();freeze=@{manifest=($manifest.files|ForEach-Object {$_.sha256});protocol_sha256=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'protocol.json') -Algorithm SHA256).Hash.ToLowerInvariant()};executable=$executable;argv=$launch;target_argv=$targetArgs;output=$Output;trace_expected=$(if($Mode -ceq 'profile'){$prefix+'.nsys-rep'}else{$null});pid=$null;exit_code=$null;post_identity_pass=$false;calibration_ready=$false}
 New-ExclusiveTextFile $receiptPath (($receipt|ConvertTo-Json -Depth 20)+"`n")
 $argLine=(@($launch|ForEach-Object {Quote-Arg ([string]$_)}) -join ' ')
 $process=Start-Process -FilePath $executable -ArgumentList $argLine -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
 $receipt.pid=$process.Id;$receipt.status='running';$receipt.started_utc=[DateTime]::UtcNow.ToString('o');Set-Content -NoNewline -Encoding utf8 $receiptPath ($receipt|ConvertTo-Json -Depth 20)
 $process.WaitForExit();$process.Refresh();$receipt.exit_code=$process.ExitCode;$receipt.completed_utc=[DateTime]::UtcNow.ToString('o')
 Assert-Frozen;$receipt.post_identity_pass=$true
 if($process.ExitCode -ne 0){$receipt.status='failed';Set-Content -NoNewline -Encoding utf8 $receiptPath ($receipt|ConvertTo-Json -Depth 20);throw "Probe failed $($process.ExitCode); retained raw/log/receipt artifacts"}
 foreach($expected in @($Output,$Output+'.receipt.json')){if(!(Test-Path -LiteralPath $expected -PathType Leaf)){throw "Expected probe artifact missing: $expected"}}
 if($Mode -ceq 'profile' -and !(Test-Path -LiteralPath ($prefix+'.nsys-rep') -PathType Leaf)){throw 'Expected Nsight trace missing'}
 $receipt.raw_sha256=(Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash.ToLowerInvariant();$receipt.probe_receipt_sha256=(Get-FileHash -LiteralPath ($Output+'.receipt.json') -Algorithm SHA256).Hash.ToLowerInvariant();$receipt.status='complete_not_calibration_ready';Set-Content -NoNewline -Encoding utf8 $receiptPath ($receipt|ConvertTo-Json -Depth 20)
 [pscustomobject]@{status=$receipt.status;condition=$condition;settle_ms=$SettleMs;arm=$Arm;pair=$Pair;mode=$Mode;raw=$Output;calibration_ready=$false;trace_confirmation_required=$true}
} finally {foreach($name in $variables){[Environment]::SetEnvironmentVariable($name,$previous[$name],'Process')}}