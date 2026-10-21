param(
    [ValidateRange(1, 65535)][int]$Port = 8765,
    [string]$PrivateDirectory = (Join-Path $env:USERPROFILE '.heterollm-sim\remote-access'),
    [string]$TunnelCredential = (Join-Path $env:USERPROFILE '.cloudflared\42954194-8db2-47fd-97ca-84950ed672cb.json'),
    [string]$CloudflaredPath = 'C:\Program Files (x86)\cloudflared\cloudflared.exe',
    [switch]$KeepAlive
)
$ErrorActionPreference = 'Stop'
$labProjectRoot = Split-Path -Parent $PSScriptRoot
$labPrivateDirectory = [System.IO.Path]::GetFullPath($PrivateDirectory)
$labTokenPath = Join-Path $labPrivateDirectory 'api-token.txt'
$labStatePath = Join-Path $labPrivateDirectory 'status.json'
$labTunnelId = '42954194-8db2-47fd-97ca-84950ed672cb'
$labTunnelCredential = [System.IO.Path]::GetFullPath($TunnelCredential)
$labTunnelConfiguration = Join-Path $labPrivateDirectory 'tunnel.yml'
$labCloudflared = [System.IO.Path]::GetFullPath($CloudflaredPath)
$labPython = Join-Path $labProjectRoot '.venv\Scripts\python.exe'
$labBackendScript = Join-Path $PSScriptRoot 'serve_remote_web.py'
# AppData is virtualized inside the desktop app, but not in a boot task.
# Pin user state outside that merged view so both contexts use identical files.
$labUserStateRoot = Split-Path -Parent $labPrivateDirectory
$env:HETEROLLM_SIM_CATALOG_CACHE = Join-Path $labUserStateRoot 'model-catalog'
$env:HETEROLLM_SIM_MODEL_ARTIFACT_DIR = Join-Path $labUserStateRoot 'model-artifacts'
$env:HETEROLLM_SIM_HARDWARE_PRESET_CACHE = Join-Path $labUserStateRoot 'component-presets'
foreach ($labRequiredPath in @($labTunnelCredential, $labCloudflared, $labPython, $labBackendScript)) {
    if (-not (Test-Path -LiteralPath $labRequiredPath -PathType Leaf)) { throw "Required service file is missing: $labRequiredPath" }
}

# The boot task uses the original user at limited privilege, before logon.
# A private file lock excludes manual starts from other Windows sessions.
$labIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
New-Item -ItemType Directory -Path $labPrivateDirectory -Force | Out-Null
$labServiceLock = $null
try {
try { $labServiceLock = [System.IO.File]::Open((Join-Path $labPrivateDirectory 'service.lock'), [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None) } catch [System.IO.IOException] { }
if ($null -eq $labServiceLock) {
    if (Test-Path -LiteralPath $labStatePath) {
        $labExistingState = Get-Content -LiteralPath $labStatePath -Raw -Encoding utf8 | ConvertFrom-Json
        if ($labExistingState.port -ne $Port) { throw 'The managed service already uses a different port.' }
        $labExistingState | ConvertTo-Json
    } else { Write-Output 'The remote lab service is already starting.' }
    return
}
New-Item -ItemType Directory -Path $labPrivateDirectory -Force | Out-Null
$labAcl = [System.Security.AccessControl.DirectorySecurity]::new()
$labAcl.SetAccessRuleProtection($true, $false)
$labInheritance = [System.Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit'
$labAcl.AddAccessRule([System.Security.AccessControl.FileSystemAccessRule]::new($labIdentity, 'FullControl', $labInheritance, 'None', 'Allow'))
$labAcl.AddAccessRule([System.Security.AccessControl.FileSystemAccessRule]::new([System.Security.Principal.SecurityIdentifier]::new('S-1-5-18'), 'FullControl', $labInheritance, 'None', 'Allow'))
$labDirectoryInfo = [System.IO.DirectoryInfo]::new($labPrivateDirectory)
if ($PSVersionTable.PSEdition -eq 'Core') { [System.IO.FileSystemAclExtensions]::SetAccessControl($labDirectoryInfo, $labAcl) }
else { $labDirectoryInfo.SetAccessControl($labAcl) }
if (-not (Test-Path -LiteralPath $labTokenPath)) {
    $labRandomBytes = [byte[]]::new(32)
    $labRng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $labRng.GetBytes($labRandomBytes) } finally { $labRng.Dispose() }
    $labToken = [Convert]::ToBase64String($labRandomBytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    [System.IO.File]::WriteAllText($labTokenPath, $labToken, [System.Text.UTF8Encoding]::new($false))
    $labToken = $null
}
function Test-LabProcess([int]$ProcessId, [string]$ExpectedMarker) {
    $labObserved = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    return ($null -ne $labObserved -and $null -ne $labObserved.CommandLine -and $labObserved.CommandLine.IndexOf($ExpectedMarker, [StringComparison]::OrdinalIgnoreCase) -ge 0)
}
function Start-LabServices {
$labPreviousState = $null
if (Test-Path -LiteralPath $labStatePath) { $labPreviousState = Get-Content -LiteralPath $labStatePath -Raw -Encoding utf8 | ConvertFrom-Json }
if ($labPreviousState -and $labPreviousState.port -ne $Port -and (
    (Test-LabProcess $labPreviousState.backend_pid 'serve_remote_web.py') -or
    (Test-LabProcess $labPreviousState.tunnel_pid $labTunnelId))) {
    throw 'A managed process still uses the original port. Stop it before changing the port.'
}
$labBackendPid = 0
if ($labPreviousState -and (Test-LabProcess $labPreviousState.backend_pid 'serve_remote_web.py')) {
    if ($labPreviousState.port -ne $Port) { throw 'The managed backend already uses a different port.' }
    $labBackendPid = [int]$labPreviousState.backend_pid
} else {
    $labListener = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
    if ($labListener.Count -gt 0) {
        # The actual venv Python child may outlive its launcher.
        $labListenerPid = [int]$labListener[0].OwningProcess
        if (-not (Test-LabProcess $labListenerPid $labBackendScript) -or -not (Test-LabProcess $labListenerPid $labTokenPath)) { throw 'The API port is occupied by another process; it was not changed.' }
        $labBackendPid = $labListenerPid
    } else {
        $labBackendProcess = Start-Process -FilePath $labPython -ArgumentList @('-u', ('"' + $labBackendScript + '"'), '--token-file', ('"' + $labTokenPath + '"'), '--port', $Port) -WorkingDirectory $labProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $labPrivateDirectory 'backend.log') -RedirectStandardError (Join-Path $labPrivateDirectory 'backend.err.log') -PassThru
        $labBackendPid = $labBackendProcess.Id
    }
}
$labBackendReady = $false
$labLastHealthError = ''
$labReadyDeadline = [DateTime]::UtcNow.AddSeconds(20)
while ([DateTime]::UtcNow -lt $labReadyDeadline) {
    try {
        $labHealthText = & $labPython $labBackendScript --token-file $labTokenPath --port $Port --check-health
        $labHealthExit = $LASTEXITCODE
        $labHealth = $labHealthText | ConvertFrom-Json
        if ($labHealthExit -eq 0 -and $labHealth.ok -eq $true) { $labBackendReady = $true; break }
        $labLastHealthError = $labHealthText -join ' '
    } catch { $labLastHealthError = $_.Exception.Message }
    if (-not (Test-LabProcess $labBackendPid 'serve_remote_web.py')) { break }
    Start-Sleep -Milliseconds 250
}
if (-not $labBackendReady) { throw "The authenticated API did not become ready: $labLastHealthError Check $labPrivateDirectory\backend.err.log." }
$labActualListener = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop)
$labBackendPid = [int]$labActualListener[0].OwningProcess
if (-not (Test-LabProcess $labBackendPid $labBackendScript)) { throw 'The authenticated API listener does not belong to this service.' }
$labYaml = @"
tunnel: $labTunnelId
credentials-file: '$labTunnelCredential'
ingress:
  - hostname: sim-api.zh.gripe
    path: ^/api(/.*)?$
    service: http://127.0.0.1:$Port
  - service: http_status:404
"@
[System.IO.File]::WriteAllText($labTunnelConfiguration, $labYaml, [System.Text.UTF8Encoding]::new($false))
$labTunnelPid = 0
if ($labPreviousState -and (Test-LabProcess $labPreviousState.tunnel_pid $labTunnelId)) {
    $labTunnelPid = [int]$labPreviousState.tunnel_pid
} else {
    $labExistingTunnels = @(Get-CimInstance Win32_Process -Filter "Name = 'cloudflared.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($labTunnelId) -and $_.CommandLine.Contains($labTunnelConfiguration) })
    if ($labExistingTunnels.Count -gt 1) { throw 'Multiple matching tunnels exist; no additional tunnel was started.' }
    if ($labExistingTunnels.Count -eq 1) { $labTunnelPid = [int]$labExistingTunnels[0].ProcessId }
    else {
        $labTunnelProcess = Start-Process -FilePath $labCloudflared -ArgumentList @('tunnel', '--config', ('"' + $labTunnelConfiguration + '"'), '--no-autoupdate', 'run', $labTunnelId) -WorkingDirectory $labProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $labPrivateDirectory 'tunnel.log') -RedirectStandardError (Join-Path $labPrivateDirectory 'tunnel.err.log') -PassThru
        $labTunnelPid = $labTunnelProcess.Id
    }
}
if (-not (Test-LabProcess $labTunnelPid $labTunnelId)) { throw "The tunnel did not start. Check $labPrivateDirectory\tunnel.err.log." }
$labStatus = [ordered]@{ backend_pid=$labBackendPid; tunnel_pid=$labTunnelPid; supervisor_pid=$(if ($KeepAlive) { $PID } else { $null }); port=$Port; backend_url='https://sim-api.zh.gripe'; website='https://zh.gripe/simulator/'; token_file=$labTokenPath; checked_at=[DateTimeOffset]::UtcNow.ToString('o') }
[System.IO.File]::WriteAllText($labStatePath, ($labStatus | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
return $labStatus
}

$labLastServicePids = ''
do {
    try {
        $labStatus = Start-LabServices
        if (-not $KeepAlive) { $labStatus | ConvertTo-Json }
        else {
            $labServicePids = '{0}/{1}' -f $labStatus.backend_pid, $labStatus.tunnel_pid
            if ($labServicePids -ne $labLastServicePids) {
                Add-Content -LiteralPath (Join-Path $labPrivateDirectory 'autostart.log') -Encoding UTF8 -Value ('{0} Ready: API {1}, tunnel {2}; no simulation was started.' -f [DateTimeOffset]::UtcNow.ToString('o'), $labStatus.backend_pid, $labStatus.tunnel_pid)
                $labLastServicePids = $labServicePids
            }
        }
    } catch {
        if (-not $KeepAlive) { throw }
        Add-Content -LiteralPath (Join-Path $labPrivateDirectory 'autostart.log') -Encoding UTF8 -Value ('{0} Service retry: {1}' -f [DateTimeOffset]::UtcNow.ToString('o'), $_.Exception.Message)
    }
    if ($KeepAlive) { Start-Sleep -Seconds 15 }
} while ($KeepAlive)
} finally {
    if ($null -ne $labServiceLock) { $labServiceLock.Dispose() }
}
