@echo off
echo ========================================
echo Trading Bot with Auto-Scheduler
echo ========================================
echo.
echo This will start the trading bot with automatic scheduling:
echo - Automatically starts trading at 4:00 AM ET on weekdays
echo - Automatically stops trading at 8:00 PM ET on weekdays  
echo - Sleep mode from 8:00 PM to 4:00 AM
echo - Sleep mode on weekends
echo.
echo Press Ctrl+C to stop the bot at any time
echo.
pause

venv\Scripts\python.exe run_scheduled_bot.py

pause
