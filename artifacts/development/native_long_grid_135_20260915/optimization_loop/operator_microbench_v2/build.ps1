param([string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path)
$ErrorActionPreference = 'Stop'
$nativeRoot = Join-Path $ProjectRoot 'source/llama.cpp-semantic'
$nativeBuild = Join-Path $nativeRoot 'build-semantic-direct'
$includeDir = Join-Path $nativeRoot 'ggml/include'
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$compiler = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe'
$sourceFile = Join-Path $PSScriptRoot 'generic_gemm_microbench.cpp'
$exe = Join-Path $PSScriptRoot 'generic-gemm-microbench.exe'
$obj = Join-Path $PSScriptRoot 'generic-gemm-microbench.obj'
$libRoot = Join-Path $nativeBuild 'ggml/src'
$cpuLibRoot = Join-Path $ProjectRoot 'source/llama.cpp-native-thread-control/build-native-thread-control/ggml/src'
$selectedNativeBin = Join-Path $ProjectRoot 'source/llama.cpp-native-thread-control/build-native-thread-control/bin'
$cudaLibRoot = Join-Path $libRoot 'ggml-cuda'
$attempt = @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter 'compile-*.log' -File).Count + 1
if ($attempt -gt 2) { throw 'At most two build attempts in this directory; inspect failure with parent before continuing.' }
foreach ($inputFile in @($vcvars, $compiler, $sourceFile, (Join-Path $libRoot 'ggml-base.lib'), (Join-Path $cpuLibRoot 'ggml-cpu.lib'), (Join-Path $cudaLibRoot 'ggml-cuda.lib'))) {
    if (!(Test-Path -LiteralPath $inputFile -PathType Leaf)) { throw "Missing build input: $inputFile" }
}
# Compiler writes only within this independent directory; existing native build stays read-only.
$commandFile = Join-Path $PSScriptRoot 'compile.cmd'
$commandText = @"
@echo off
call "$vcvars" >nul
if errorlevel 1 exit /b %errorlevel%
"$compiler" /nologo /std:c++17 /EHsc /O2 /MD /utf-8 /DGGML_SHARED /DGGML_BACKEND_SHARED /I"$includeDir" /I"E:\cuda\include" "$sourceFile" /Fe:"$exe" /Fo:"$obj" /link /LIBPATH:"$cpuLibRoot" /LIBPATH:"$libRoot" /LIBPATH:"$cudaLibRoot" /LIBPATH:"E:\cuda\lib\x64" ggml-base.lib ggml-cpu.lib ggml-cuda.lib bcrypt.lib cudart.lib
exit /b %errorlevel%
"@
[IO.File]::WriteAllText($commandFile, $commandText, [Text.UTF8Encoding]::new($false))
$logPath = Join-Path $PSScriptRoot ('compile-' + $attempt + '.log')
$oldPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$compilerOutput = & $env:ComSpec /d /c "`"$commandFile`"" 2>&1
$compilerExit = $LASTEXITCODE
$ErrorActionPreference = $oldPreference
[IO.File]::WriteAllText($logPath, ($compilerOutput | Out-String), [Text.UTF8Encoding]::new($false))
$compilerOutput | Write-Output
if ($compilerExit -ne 0) { throw "Compile attempt $attempt failed with exit code $compilerExit; see $logPath" }
function EvidenceRef([string]$Path) {
    $item = Get-Item -LiteralPath $Path
    return @{path=$item.FullName; bytes=$item.Length; sha256=(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$inputs = @($sourceFile, $PSCommandPath, (Join-Path $PSScriptRoot 'invoke.ps1'), $commandFile,
    (Join-Path $nativeBuild 'CMakeCache.txt'), (Join-Path $libRoot 'ggml-base.lib'), (Join-Path $cpuLibRoot 'ggml-cpu.lib'),
    (Join-Path $cudaLibRoot 'ggml-cuda.lib'))
$inputs += (Get-ChildItem -LiteralPath $includeDir -Filter '*.h' -File).FullName
$inputs += @('ggml/src/ggml.c','ggml/src/ggml-backend.cpp','ggml/src/ggml-cuda/ggml-cuda.cu') | ForEach-Object { Join-Path $nativeRoot $_ }
$dlls = @('ggml-base.dll','ggml-cpu.dll','ggml-cuda.dll','ggml.dll') | ForEach-Object { Join-Path $selectedNativeBin $_ }
$manifest = @{
    schema='generic-gemm-build/v2'; built_utc=[DateTime]::UtcNow.ToString('o'); compiler=(EvidenceRef $compiler);
    executable=(EvidenceRef $exe); compile_log=(EvidenceRef $logPath); compile_attempt=$attempt;
    flags=@('/std:c++17','/EHsc','/O2','/MD','/utf-8','/DGGML_SHARED','/DGGML_BACKEND_SHARED');
    inputs=@($inputs | ForEach-Object { EvidenceRef $_ }); dependencies=@($dlls | ForEach-Object { EvidenceRef $_ });
    source_root=$nativeRoot; native_bin=$selectedNativeBin;
    native_source_commit=(& git -C $nativeRoot rev-parse HEAD);
    source_binding='current local locked build assets; loaded DLL identity must match at execution';
    native_binary_modified=$false; llm_actual_inputs=@();
    timing_contract='ggml_backend_graph_compute includes internal synchronize; no separately added launch/sync';
    cache_policy='explicit_hot_or_untimed_eviction; GPU eviction >=4x measured L2, not a guarantee of DRAM transactions'; hbm_bandwidth_validation=$false;
    test_status='compiled_not_computed'
}
$manifestPath = Join-Path $PSScriptRoot 'build_manifest.json'
[IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
Write-Output ("Built independent executable: " + $exe)
Write-Output ("Manifest: " + $manifestPath)
