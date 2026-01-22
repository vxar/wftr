@echo off
REM View bot logs in real-time (Windows)

setlocal

set "SCRIPT_DIR=%~dp0"
set "LOG_DIR=%SCRIPT_DIR%logs"

if not exist "%LOG_DIR%" (
    echo Log directory not found: %LOG_DIR%
    exit /b 1
)

REM Find the most recent log file
for /f "delims=" %%I in ('dir /b /o-d "%LOG_DIR%\bot_*.log" 2^>NUL') do (
    set "LATEST_LOG=%LOG_DIR%\%%I"
    goto :found
)

echo No log files found in %LOG_DIR%
exit /b 1

:found
echo Viewing log file: %LATEST_LOG%
echo Press Ctrl+C to exit
echo.
echo ========================================
echo.

REM Use PowerShell to tail the log file
powershell -Command "Get-Content '%LATEST_LOG%' -Wait -Tail 50"

endlocal
