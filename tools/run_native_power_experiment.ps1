param(
    [Parameter(Mandatory=$true)][string]$Protocol,
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet('balanced','high')][string]$PowerMode='balanced',
    [int]$MaxSeconds=3600,
    [switch]$Resume
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
function Get-CurrentScheme {
    $value=& powercfg /getactivescheme
    if($LASTEXITCODE -ne 0){throw 'Cannot read power scheme'}
    $id=[regex]::Match(($value -join ' '),'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}').Value.ToLowerInvariant()
    if(-not $id){throw 'Power scheme GUID missing'}
    return $id
}
$before=Get-CurrentScheme
$target=if($PowerMode -eq 'high'){'8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'}else{'381b4222-f694-41f0-9685-ff5bb260df2e'}
$outputAbs=[IO.Path]::GetFullPath((Join-Path $root $Output))
$protocolAbs=[IO.Path]::GetFullPath((Join-Path $root $Protocol))
$receipt=$outputAbs+'.power-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffff')+'.json'
$state=@{before=$before;requested=$target;started_utc=[DateTime]::UtcNow.ToString('o');restored=$false;native_exit_code=$null}
$exitCode=1
try {
    & powercfg /setactive $target
    if($LASTEXITCODE -ne 0){throw 'Power scheme change failed'}
    $state.actual=Get-CurrentScheme
    if($state.actual -ne $target){throw 'Power scheme readback differs'}
    $argsList=@((Join-Path $root 'tools/native_repeatability_experiment.py'),'--protocol',$protocolAbs,'--output',$outputAbs,'--max-seconds',[string]$MaxSeconds)
    if($Resume){$argsList+='--resume'}
    & 'E:\anaconda\python.exe' @argsList
    $exitCode=$LASTEXITCODE
    $state.native_exit_code=$exitCode
} finally {
    & powercfg /setactive $before
    $state.restored=(($LASTEXITCODE -eq 0) -and (Get-CurrentScheme) -eq $before)
    $state.ended_utc=[DateTime]::UtcNow.ToString('o')
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receipt -Encoding utf8
    if(-not $state.restored){throw "Failed to restore original power plan: $before"}
}
exit $exitCode
