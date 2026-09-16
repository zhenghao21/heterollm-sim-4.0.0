@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
E:\cuda\bin\cuobjdump.exe --extract-elf all F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\development\native_long_grid_135_20260915\optimization_loop\round_026\mmvq_wrapper_launch_match\build_attempt.0001\mmvq-wrapper-launch-match.exe
