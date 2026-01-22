# Running Bot in Background

The bot can be run in the background to reduce console output overhead and improve performance. Logs are redirected to files instead of the console.

## Available Scripts

### For Linux/Mac/Git Bash:
- `run_background.sh` - Start bot in background
- `stop_bot.sh` - Stop the background bot
- `view_logs.sh` - View logs in real-time

### For Windows:
- `run_background.bat` - Start bot in background (Command Prompt)
- `run_background.ps1` - Start bot in background (PowerShell - recommended)
- `stop_bot.bat` - Stop the background bot (Command Prompt)
- `stop_bot.ps1` - Stop the background bot (PowerShell)
- `view_logs.bat` - View logs in real-time (Command Prompt)
- `view_logs.ps1` - View logs in real-time (PowerShell)

## Usage

### Starting the Bot

**Linux/Mac/Git Bash:**
```bash
./run_background.sh
```

**Windows (PowerShell - Recommended):**
```powershell
.\run_background.ps1
```

**Windows (Command Prompt):**
```cmd
run_background.bat
```

The bot will start in the background and logs will be written to:
- `logs/bot_YYYYMMDD_HHMMSS.log`

### Stopping the Bot

**Linux/Mac/Git Bash:**
```bash
./stop_bot.sh
```

**Windows (PowerShell):**
```powershell
.\stop_bot.ps1
```

**Windows (Command Prompt):**
```cmd
stop_bot.bat
```

### Viewing Logs

**Linux/Mac/Git Bash:**
```bash
./view_logs.sh
```

**Windows (PowerShell):**
```powershell
.\view_logs.ps1
```

**Windows (Command Prompt):**
```cmd
view_logs.bat
```

Or manually view the latest log file:
```bash
# Linux/Mac
tail -f logs/bot_*.log

# Windows PowerShell
Get-Content logs\bot_*.log -Wait -Tail 50
```

## Benefits of Background Running

1. **Performance**: No console output overhead - significantly faster
2. **Memory**: Reduced memory usage from console buffering
3. **Resource Usage**: Lower CPU usage without console rendering
4. **Persistence**: Bot continues running even if terminal is closed (on Linux/Mac)
5. **Logging**: All logs saved to files for later analysis

## Log Files

- Logs are stored in the `logs/` directory
- Each run creates a new log file with timestamp: `bot_YYYYMMDD_HHMMSS.log`
- Logs include both stdout and stderr
- Log files can grow large over time - consider log rotation

## Checking Bot Status

**Linux/Mac:**
```bash
# Check if bot is running
ps aux | grep python | grep run_dashboard_with_bot

# Or check PID file
cat bot.pid
ps -p $(cat bot.pid)
```

**Windows (PowerShell):**
```powershell
# Check if bot is running
Get-Process python | Where-Object {$_.CommandLine -like "*run_dashboard_with_bot*"}

# Or check job status (if using PowerShell script)
Get-Job
```

## Troubleshooting

### "bad interpreter: No such file or directory" or "^M" error
This means the shell script has Windows line endings (CRLF) instead of Unix line endings (LF).

**Fix in WSL/Ubuntu:**
```bash
# Option 1: Use the fix script
./fix_line_endings.sh

# Option 2: Manual fix
sed -i 's/\r$//' run_background.sh stop_bot.sh view_logs.sh
chmod +x run_background.sh stop_bot.sh view_logs.sh

# Option 3: If dos2unix is installed
dos2unix run_background.sh stop_bot.sh view_logs.sh
chmod +x run_background.sh stop_bot.sh view_logs.sh
```

### Bot won't start
- Check if virtual environment is activated
- Verify Python is installed and accessible
- Check log file for error messages

### Bot stops immediately
- Check the log file for errors
- Verify all dependencies are installed
- Check if port 5000 (dashboard) is already in use

### Can't stop the bot
- On Linux/Mac: `kill -9 $(cat bot.pid)`
- On Windows: Use Task Manager to find and kill python.exe processes

### Logs not appearing
- Verify `logs/` directory exists and is writable
- Check disk space
- Verify bot is actually running

## Notes

- On Windows, batch files run in a minimized window. PowerShell scripts use background jobs.
- The bot.pid file stores the process ID for stopping the bot
- Logs are appended, so multiple runs will create separate log files
- For production, consider using a process manager like systemd (Linux) or a Windows service
