@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Keep the CMD launcher self-contained in the extracted package.  The
rem PowerShell entry point is renamed to Start-Simulator.ps1 by the packager.
set "LAUNCHER=%~dp0Start-Simulator.ps1"
if not exist "%LAUNCHER%" set "LAUNCHER=%~dp0start_offline.ps1"
if not exist "%LAUNCHER%" (
    echo HeteroLLM Simulator launcher not found:
    echo "%LAUNCHER%"
    echo.
    pause
    endlocal & exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER%" %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo HeteroLLM Simulator failed to start. Exit code: %EXITCODE%
    echo Check the message above, then press any key to close this window.
    pause
)
endlocal & exit /b %EXITCODE%
