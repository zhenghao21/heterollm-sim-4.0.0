param()
$ErrorActionPreference = 'Stop'
$toolRoot = 'F:\codex_project\37_LLMsim\tools\nsys_cli\target-windows-x64'
$inventoryPath = Join-Path $PSScriptRoot 'tool_inventory.json'
$signaturePath = Join-Path $PSScriptRoot 'tool_signatures.json'
if ((Test-Path -LiteralPath $inventoryPath) -or (Test-Path -LiteralPath $signaturePath)) { throw 'Tool identity already recorded; do not overwrite.' }
$files = @(Get-ChildItem -LiteralPath $toolRoot -File -Recurse | Where-Object { $_.Extension -ne '.pyc' -and $_.FullName -notmatch '[\\/]__pycache__[\\/]' } | Sort-Object FullName)
if ($files.Count -eq 0) { throw 'Empty Nsight tool tree' }
$rows = @($files | ForEach-Object { [ordered]@{ path=$_.FullName; bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() } })
$binaryRows = @($files | Where-Object { $_.Extension -in @('.exe','.dll','.pyd') } | ForEach-Object {
    $sig = Get-AuthenticodeSignature -LiteralPath $_.FullName
    [ordered]@{ path=$_.FullName; status=[string]$sig.Status; signer_subject=if($sig.SignerCertificate){$sig.SignerCertificate.Subject}else{$null}; signer_thumbprint=if($sig.SignerCertificate){$sig.SignerCertificate.Thumbprint}else{$null}; file_version=$_.VersionInfo.FileVersion }
})
$criticalNames = @('nsys.exe','cupti64_134.dll','ToolsInjection64.dll','nsight-sys-service.exe')
foreach ($name in $criticalNames) {
    $expectedPath = Join-Path $toolRoot $name
    $row = @($binaryRows | Where-Object { $_.path -eq $expectedPath })
    if ($row.Count -ne 1 -or $row[0].status -ne 'Valid' -or $row[0].signer_subject -notmatch 'NVIDIA Corporation') { throw "Critical NVIDIA tool signature invalid: $name" }
}
$inventory = [ordered]@{schema='nsys-bundled-tool-inventory/v1';created_utc=[DateTime]::UtcNow.ToString('o');root=$toolRoot;files=$rows;excluded='Generated __pycache__ and .pyc only';runtime_os_dependency_closure_complete=$false}
$signatures = [ordered]@{schema='nsys-authenticode-signature-evidence/v1';created_utc=[DateTime]::UtcNow.ToString('o');critical_nvidia_binary_signatures_valid=$true;critical_names=$criticalNames;all_bundled_binary_signatures=$binaryRows;loaded_modules_during_capture_not_yet_observed=$true}
[IO.File]::WriteAllText($inventoryPath, ($inventory | ConvertTo-Json -Depth 8) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText($signaturePath, ($signatures | ConvertTo-Json -Depth 8) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
[PSCustomObject]@{file_count=$rows.Count;binary_signature_count=$binaryRows.Count;critical_valid=$true;gpu_workload_executed=$false} | ConvertTo-Json
