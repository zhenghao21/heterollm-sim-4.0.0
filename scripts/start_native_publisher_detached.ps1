# usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_native_publisher_detached.ps1
# Starts the native comparison publisher (15-minute uploads) through WMI, so it is a child of the WMI service rather
# than of the Claude desktop app and keeps running when the app restarts. It does not survive a reboot: run it again.
# The Netlify CLI login was made inside the packaged Claude app, whose %APPDATA% writes land in the package's private
# Roaming folder. Only APPDATA is pointed there so the CLI finds that login; no token is copied or passed.
$root = Split-Path -Parent $PSScriptRoot
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*publish_exploring_hbf_native*' }
if ($running) { Write-Output ("publisher already running: {0}" -f ($running.ProcessId -join ',')); exit 0 }
$login = Get-ChildItem (Join-Path $env:LOCALAPPDATA 'Packages') -Directory -Filter 'Claude_*' -ErrorAction SilentlyContinue |
    ForEach-Object { Join-Path $_.FullName 'LocalCache\Roaming' } |
    Where-Object { Test-Path (Join-Path $_ 'netlify\Config\config.json') } | Select-Object -First 1
if (-not $login) { throw 'No Netlify CLI login found in the Claude app package; run netlify login inside the app first.' }
$python = Join-Path $root '.venv\Scripts\python.exe'
$script = Join-Path $root 'scripts\publish_exploring_hbf_native.py'
$log = Join-Path $root 'tmp\native_comparison_publisher.log'
$command = "cmd.exe /d /c set `"APPDATA=$login`"&& `"$python`" `"$script`" --interval 900 >> `"$log`" 2>&1"
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $command; CurrentDirectory = $root; ProcessStartupInformation = $startup }
if ($result.ReturnValue -ne 0) { throw "Win32_Process.Create failed with $($result.ReturnValue)" }
Write-Output ("started pid {0}; status in tmp\native_comparison_publisher.json" -f $result.ProcessId)
