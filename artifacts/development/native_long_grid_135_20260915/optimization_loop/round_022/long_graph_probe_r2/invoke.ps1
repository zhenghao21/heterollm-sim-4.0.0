# Identity-only helper; timed execution belongs exclusively to the frozen collector.
param([Parameter(Mandatory=$true)][switch]$IdentityCheck)
$ErrorActionPreference='Stop'
$m=Get-Content -Raw (Join-Path $PSScriptRoot 'build_manifest.json')|ConvertFrom-Json
$priorPath=$env:PATH
try {$env:PATH="$($m.native_bin);E:\cuda\bin;$priorPath"; & $m.executable.path --identity-check; if($LASTEXITCODE -ne 0){throw 'Identity failed'}} finally {$env:PATH=$priorPath}
