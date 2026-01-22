# Stop the background trading bot (PowerShell)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path $ScriptDir "bot.pid"

if (-not (Test-Path $PidFile)) {
    Write-Host "Bot is not running (PID file not found)"
    exit 1
}

$Pid = Get-Content $PidFile
$Process = Get-Process -Id $Pid -ErrorAction SilentlyContinue

if (-not $Process) {
    Write-Host "Bot is not running (process $Pid not found)"
    Remove-Item $PidFile -Force
    exit 1
}

Write-Host "Stopping bot (PID: $Pid)..."
Stop-Process -Id $Pid -Force

# Wait a moment
Start-Sleep -Seconds 1

$Process = Get-Process -Id $Pid -ErrorAction SilentlyContinue
if (-not $Process) {
    Write-Host "Bot stopped successfully"
    Remove-Item $PidFile -Force
} else {
    Write-Host "Warning: Bot process may still be running."
}
