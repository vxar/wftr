@echo off
REM Enhanced Autonomous Trading Bot - Background Runner (Windows)
REM Runs the bot in the background with logs redirected to file

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Log directory
set "LOG_DIR=%SCRIPT_DIR%logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Log file with timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "dt=%%I"
set "TIMESTAMP=%dt:~0,8%_%dt:~8,6%"
set "LOG_FILE=%LOG_DIR%\bot_%TIMESTAMP%.log"
set "PID_FILE=%SCRIPT_DIR%bot.pid"

REM Check if bot is already running
if exist "%PID_FILE%" (
    set /p OLD_PID=<"%PID_FILE%"
    tasklist /FI "PID eq !OLD_PID!" 2>NUL | find /I /N "!OLD_PID!">NUL
    if "!ERRORLEVEL!"=="0" (
        echo Bot is already running with PID !OLD_PID!
        echo To stop it, run: stop_bot.bat
        exit /b 1
    ) else (
        REM PID file exists but process is not running, remove stale PID file
        del "%PID_FILE%" 2>NUL
    )
)

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Error: Virtual environment not found. Please create it first.
    exit /b 1
)

echo ========================================
echo   Enhanced Autonomous Trading Bot
echo   Background Mode
echo ========================================
echo.
echo Starting bot in background...
echo Log file: %LOG_FILE%
echo PID file: %PID_FILE%
echo.

REM Start bot in background using PowerShell (captures PID reliably)
powershell -Command "$proc = Start-Process -FilePath 'python' -ArgumentList 'run_dashboard_with_bot.py' -RedirectStandardOutput '%LOG_FILE%' -RedirectStandardError '%LOG_FILE%' -PassThru -WindowStyle Hidden; $proc.Id | Out-File -FilePath '%PID_FILE%' -Encoding ASCII -NoNewline; Write-Host ('Bot started with PID: ' + $proc.Id)"

REM Wait a moment for process to start
timeout /t 2 /nobreak >NUL

REM Read PID from file
set "BOT_PID="
if exist "%PID_FILE%" (
    set /p BOT_PID=<"%PID_FILE%"
)

REM Verify process is running
if defined BOT_PID (
    tasklist /FI "PID eq %BOT_PID%" 2>NUL | find /I "%BOT_PID%" >NUL
    if "!ERRORLEVEL!"=="0" (
        echo.
        echo To view logs in real-time:
        echo   powershell Get-Content "%LOG_FILE%" -Wait -Tail 50
        echo.
        echo To stop the bot:
        echo   stop_bot.bat
        echo.
        echo Bot is running in background. You can close this window.
    ) else (
        echo Warning: Bot process %BOT_PID% not found after start.
        echo Check the log file for errors: %LOG_FILE%
        echo.
        echo Last few lines of log:
        powershell Get-Content "%LOG_FILE%" -Tail 10
        del "%PID_FILE%" 2>NUL
    )
) else (
    echo Warning: Could not read bot PID from file.
    echo Check the log file for errors: %LOG_FILE%
    echo.
    echo Last few lines of log:
    powershell Get-Content "%LOG_FILE%" -Tail 10
    echo.
    echo Recent Python processes:
    powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,StartTime | Sort-Object StartTime -Descending | Select-Object -First 3 | Format-Table"
)

endlocal
