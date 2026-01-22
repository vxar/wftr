# Enhanced Autonomous Trading Bot - Background Runner (PowerShell)
# Runs the bot in the background with logs redirected to file

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Log directory
$LogDir = Join-Path $ScriptDir "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# Log file with timestamp
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "bot_$Timestamp.log"
$PidFile = Join-Path $ScriptDir "bot.pid"

# Check if bot is already running
if (Test-Path $PidFile) {
    $OldPid = Get-Content $PidFile
    $Process = Get-Process -Id $OldPid -ErrorAction SilentlyContinue
    if ($Process) {
        Write-Host "Bot is already running with PID $OldPid"
        Write-Host "To stop it, run: .\stop_bot.ps1"
        exit 1
    } else {
        # PID file exists but process is not running, remove stale PID file
        Remove-Item $PidFile -Force
    }
}

# Find virtual environment
$VenvPath = Join-Path $ScriptDir "venv"
$PythonExe = $null

if (Test-Path (Join-Path $VenvPath "Scripts\python.exe")) {
    $PythonExe = Join-Path $VenvPath "Scripts\python.exe"
} elseif (Test-Path (Join-Path $VenvPath "bin\python")) {
    $PythonExe = Join-Path $VenvPath "bin\python"
} else {
    # Fallback to system Python
    $PythonExe = "python"
    Write-Host "Warning: Using system Python (venv not found)"
}

Write-Host "========================================"
Write-Host "  Enhanced Autonomous Trading Bot"
Write-Host "  Background Mode"
Write-Host "========================================"
Write-Host ""
Write-Host "Starting bot in background..."
Write-Host "Log file: $LogFile"
Write-Host "PID file: $PidFile"
Write-Host ""

# Create a wrapper script that handles fallback
$WrapperScript = Join-Path $ScriptDir "run_bot_wrapper.ps1"
$PythonExeEscaped = $PythonExe -replace "'", "''"
$LogFileEscaped = $LogFile -replace "'", "''"
@"
# Wrapper script for bot execution
Set-Location '$ScriptDir'
& '$PythonExeEscaped' run_dashboard_with_bot.py *>> '$LogFileEscaped' 2>&1
if (`$LASTEXITCODE -ne 0) {
    `$Timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    '[' + `$Timestamp + '] Enhanced bot failed, trying simple bot...' | Out-File -FilePath '$LogFileEscaped' -Append
    & '$PythonExeEscaped' simple_bot.py *>> '$LogFileEscaped' 2>&1
}
"@ | Out-File -FilePath $WrapperScript -Encoding UTF8

# Start bot process in background (truly detached)
$ProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
$ProcessInfo.FileName = "powershell.exe"
$ProcessInfo.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$WrapperScript`""
$ProcessInfo.WorkingDirectory = $ScriptDir
$ProcessInfo.UseShellExecute = $false
$ProcessInfo.CreateNoWindow = $true
$ProcessInfo.RedirectStandardOutput = $false
$ProcessInfo.RedirectStandardError = $false

$Process = [System.Diagnostics.Process]::Start($ProcessInfo)

# Save PID
$Process.Id | Out-File -FilePath $PidFile

Write-Host "Bot started with PID: $($Process.Id)"
Write-Host ""
Write-Host "To view logs in real-time:"
Write-Host "  .\view_logs.ps1"
Write-Host ""
Write-Host "To stop the bot:"
Write-Host "  .\stop_bot.ps1"
Write-Host ""
Write-Host "To check if bot is running:"
Write-Host "  Get-Process -Id $($Process.Id) -ErrorAction SilentlyContinue"
Write-Host ""

# Wait a moment to check if bot started successfully
Start-Sleep -Seconds 2

$RunningProcess = Get-Process -Id $Process.Id -ErrorAction SilentlyContinue
if ($RunningProcess) {
    Write-Host "Bot is running successfully!"
    Write-Host "Background process started. You can close this terminal."
} else {
    Write-Host "Warning: Bot process may have exited immediately."
    Write-Host "Check the log file for errors: $LogFile"
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Remove-Item $WrapperScript -Force -ErrorAction SilentlyContinue
    exit 1
}
