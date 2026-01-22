@echo off
REM Stop the background trading bot (Windows)

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PID_FILE=%SCRIPT_DIR%bot.pid"

if not exist "%PID_FILE%" (
    echo Bot is not running (PID file not found)
    exit /b 1
)

set /p PID=<"%PID_FILE%"

tasklist /FI "PID eq %PID%" 2>NUL | find /I /N "%PID%">NUL
if "!ERRORLEVEL!"=="1" (
    echo Bot is not running (process %PID% not found)
    del "%PID_FILE%" 2>NUL
    exit /b 1
)

echo Stopping bot (PID: %PID%)...
taskkill /PID %PID% /F >NUL 2>&1

timeout /t 2 /nobreak >NUL

tasklist /FI "PID eq %PID%" 2>NUL | find /I /N "%PID%">NUL
if "!ERRORLEVEL!"=="1" (
    echo Bot stopped successfully
    del "%PID_FILE%" 2>NUL
) else (
    echo Warning: Bot process may still be running. Try stopping manually.
    echo   taskkill /PID %PID% /F
)

endlocal
