@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\dumpbin.exe" /nologo /dependents F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_028\host_submission_probe\host-decoder-tests.0002.exe
