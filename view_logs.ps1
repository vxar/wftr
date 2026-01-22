# View bot logs in real-time (PowerShell)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptDir "logs"

if (-not (Test-Path $LogDir)) {
    Write-Host "Log directory not found: $LogDir"
    exit 1
}

# Find the most recent log file
$LatestLog = Get-ChildItem -Path $LogDir -Filter "bot_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $LatestLog) {
    Write-Host "No log files found in $LogDir"
    exit 1
}

Write-Host "Viewing log file: $($LatestLog.FullName)"
Write-Host "Press Ctrl+C to exit"
Write-Host ""
Write-Host "========================================"
Write-Host ""

# Tail the log file
Get-Content -Path $LatestLog.FullName -Wait -Tail 50
