param(
    [string]$UserSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value,
    [string]$PrivateDirectory = (Join-Path $env:USERPROFILE '.heterollm-sim\remote-access'),
    [string]$TunnelCredential = (Join-Path $env:USERPROFILE '.cloudflared\42954194-8db2-47fd-97ca-84950ed672cb.json'),
    [string]$CloudflaredPath = 'C:\Program Files (x86)\cloudflared\cloudflared.exe'
)
$ErrorActionPreference = 'Stop'
trap {
    $_.Exception.Message | Set-Content -LiteralPath (Join-Path $PrivateDirectory 'autostart.install-error.log') -Encoding UTF8
    exit 1
}
$labTaskName = 'HeteroLLM Remote Lab'
$labProjectRoot = Split-Path -Parent $PSScriptRoot
$labStartupScript = Join-Path $PSScriptRoot 'start_remote_lab.ps1'
# Use Windows' own PowerShell, independent of Codex's bundled runtime.
$labPowerShell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
foreach ($labPath in @($labStartupScript, $labPowerShell, $TunnelCredential, $CloudflaredPath, (Join-Path $PrivateDirectory 'api-token.txt'))) {
    if (-not (Test-Path -LiteralPath $labPath -PathType Leaf)) { throw "Required service file is missing: $labPath" }
}
$labArguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -File "{0}" -PrivateDirectory "{1}" -TunnelCredential "{2}" -CloudflaredPath "{3}" -KeepAlive' -f $labStartupScript, $PrivateDirectory, $TunnelCredential, $CloudflaredPath
$labAction = New-ScheduledTaskAction -Execute $labPowerShell -Argument $labArguments -WorkingDirectory $labProjectRoot
$labTrigger = New-ScheduledTaskTrigger -AtStartup
$labTrigger.Delay = 'PT30S'
# No Windows password is stored; only existing local tunnel credentials are used.
$labPrincipal = New-ScheduledTaskPrincipal -UserId $UserSid -LogonType S4U -RunLevel Limited
$labSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$labTask = New-ScheduledTask -Action $labAction -Trigger $labTrigger -Principal $labPrincipal -Settings $labSettings -Description 'Start only the authenticated simulator HTTP API and dedicated HTTPS tunnel. Never resume scans or simulation jobs.'
if (Get-ScheduledTask -TaskName $labTaskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $labTaskName
    $labStopDeadline = [DateTime]::UtcNow.AddSeconds(10)
    while ((Get-ScheduledTask -TaskName $labTaskName).State -eq 'Running' -and [DateTime]::UtcNow -lt $labStopDeadline) { Start-Sleep -Milliseconds 100 }
    if ((Get-ScheduledTask -TaskName $labTaskName).State -eq 'Running') { throw 'The old service supervisor did not exit; the task was not replaced.' }
}
Register-ScheduledTask -TaskName $labTaskName -InputObject $labTask -Force | Out-Null
Start-ScheduledTask -TaskName $labTaskName
[ordered]@{ task_name=$labTaskName; user_sid=$UserSid; trigger='Boot + 30 seconds'; logon_type='S4U'; run_level='Limited'; website='https://zh.gripe/simulator/'; installed_at=[DateTimeOffset]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PrivateDirectory 'autostart.install.json') -Encoding UTF8
Write-Output 'Remote lab boot task installed and started. No scan task was registered.'
