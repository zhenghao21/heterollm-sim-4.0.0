param(
    [Parameter(Mandatory=$true)][string]$SourceRoot,
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [Parameter(Mandatory=$true)][string]$Protocol,
    [Parameter(Mandatory=$true)][string]$Output,
    [int]$MaxSeconds=28800,
    [switch]$Resume
)
$ErrorActionPreference='Stop'
$executionRoot=(Resolve-Path -LiteralPath $SourceRoot).Path
$projectRoot=(Resolve-Path -LiteralPath $DataRoot).Path
$protocolPath=(Resolve-Path -LiteralPath $Protocol).Path
$outputPath=[IO.Path]::GetFullPath($Output)
$parent=Split-Path -Parent $outputPath
if(-not (Test-Path -LiteralPath $parent)){throw 'Output parent must already exist'}
$stamp=[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffff')
$receipt=Join-Path $parent ('hardware_control_'+$stamp+'.json')
$consoleLog=Join-Path $parent ('native_console_'+$stamp+'.log')
$state=@{started_utc=[DateTime]::UtcNow.ToString('o');source_root=$executionRoot;data_root=$projectRoot;protocol=$protocolPath;output=$outputPath;gpu_target_mhz=2400;clock_changed=$false;clock_restored=$null;power_restored=$false;status='starting';max_seconds=$MaxSeconds;console_log=$consoleLog}
$exitCode=1
function Get-ActualScheme {
    $text=(& powercfg /getactivescheme | Out-String)
    if($LASTEXITCODE -ne 0){throw 'Unable to query power scheme'}
    $id=[regex]::Match($text,'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}').Value.ToLowerInvariant()
    if(-not $id){throw 'Actual power scheme identity missing'}
    return $id
}
function Save-State {
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receipt -Encoding utf8
}
try {
    Set-Location -LiteralPath $executionRoot
    $state.administrator=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if(-not $state.administrator){throw 'Fixed-clock measurement requires the authorized administrator process'}
    $conflicting=Get-Process -Name llama-server -ErrorAction SilentlyContinue
    if($conflicting){throw 'Another native inference process is already active'}
    $state.power_before=Get-ActualScheme
    & powercfg /setactive '381b4222-f694-41f0-9685-ff5bb260df2e'
    if($LASTEXITCODE -ne 0){throw 'Unable to set the frozen power scheme'}
    $state.power_actual=Get-ActualScheme
    if($state.power_actual -ne '381b4222-f694-41f0-9685-ff5bb260df2e'){throw 'Power scheme readback mismatch'}
    $state.clock_command_output=(& 'C:\Windows\System32\nvidia-smi.exe' -lgc '2400,2400' 2>&1 | Out-String)
    $state.clock_command_exit=$LASTEXITCODE
    $state.clock_changed=($LASTEXITCODE -eq 0)
    if(-not $state.clock_changed){throw 'Driver rejected the GPU clock setting'}
    Start-Sleep -Seconds 1
    $state.clock_readback=(& 'C:\Windows\System32\nvidia-smi.exe' --query-gpu=uuid,pstate,clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader 2>&1 | Out-String)
    $state.status='measuring';Save-State
    $nativeArgs=@((Join-Path $executionRoot 'tools\native_repeatability_experiment.py'),'--data-root',$projectRoot,'--protocol',$protocolPath,'--output',$outputPath,'--max-seconds',[string]$MaxSeconds)
    if($Resume){$nativeArgs+='--resume'}
    & 'E:\anaconda\python.exe' @nativeArgs *> $consoleLog
    $exitCode=$LASTEXITCODE
    $state.native_exit=$exitCode
    $state.status='measurement_finished'
} catch {
    $state.status='failed';$state.error=$_.Exception.Message
} finally {
    if($state.clock_changed){
        $state.clock_reset_output=(& 'C:\Windows\System32\nvidia-smi.exe' -rgc 2>&1 | Out-String)
        $state.clock_restored=($LASTEXITCODE -eq 0)
    }
    if($state.power_before){
        & powercfg /setactive $state.power_before
        $state.power_restored=(($LASTEXITCODE -eq 0) -and ((Get-ActualScheme) -eq $state.power_before))
    }
    $state.measurement_ended_utc=[DateTime]::UtcNow.ToString('o');Save-State
}
# The renderer is part of the same frozen source snapshot. Never run it beside native inference.
$reporter=Join-Path $executionRoot 'tools\native_long_grid_report.py'
if((Test-Path -LiteralPath $reporter) -and (Test-Path -LiteralPath (Join-Path $outputPath 'freeze.json'))){
    & 'E:\anaconda\python.exe' $reporter --campaign (Split-Path -Parent $outputPath) --output (Join-Path $parent 'report') *> (Join-Path $parent ('report_console_'+$stamp+'.log'))
    $state.report_exit=$LASTEXITCODE
}
$state.ended_utc=[DateTime]::UtcNow.ToString('o');Save-State
exit $exitCode
