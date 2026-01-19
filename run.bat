@echo off
echo Activating Python virtual environment...
call venv\Scripts\activate.bat

echo Starting Enhanced Autonomous Trading Bot...
echo.
echo ========================================
echo   Enhanced Autonomous Trading Bot
echo ========================================
echo.

cd /d "%~dp0"

REM Run the enhanced bot
python run_dashboard_with_bot.py
if %ERRORLEVEL% EQU 0 goto end

REM Fallback to simple bot
echo Trying simple bot...
python simple_bot.py

:end
echo.
echo Bot stopped. Press any key to exit...
pause > nul
