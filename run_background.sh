#!/bin/bash

# Enhanced Autonomous Trading Bot - Background Runner
# Runs the bot in the background with logs redirected to file

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Log directory
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/bot_${TIMESTAMP}.log"
PID_FILE="$SCRIPT_DIR/bot.pid"

# Check if bot is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Bot is already running with PID $OLD_PID"
        echo "To stop it, run: ./stop_bot.sh"
        exit 1
    else
        # PID file exists but process is not running, remove stale PID file
        rm -f "$PID_FILE"
    fi
fi

# Activate virtual environment
VENV_ACTIVATED=false

# Check if already in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Already in virtual environment: $VIRTUAL_ENV"
    VENV_ACTIVATED=true
# Try Linux-style venv first
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    VENV_ACTIVATED=true
# Try Windows-style venv (for WSL/Git Bash)
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
    VENV_ACTIVATED=true
# Try Windows-style venv with .bat extension (WSL)
elif [ -f "venv/Scripts/activate.bat" ]; then
    # In WSL, we can't directly source .bat files, but we can use the Python directly
    # Set PATH to include venv Scripts
    export PATH="$(pwd)/venv/Scripts:$PATH"
    export VIRTUAL_ENV="$(pwd)/venv"
    VENV_ACTIVATED=true
    echo "Note: Using Windows venv in WSL (Python from venv/Scripts)"
# Check if venv directory exists but activate script is missing
elif [ -d "venv" ]; then
    echo "Warning: venv directory exists but activate script not found."
    echo "Attempting to use Python from venv directly..."
    # Try to find Python in venv
    if [ -f "venv/bin/python" ]; then
        export PATH="$(pwd)/venv/bin:$PATH"
        export VIRTUAL_ENV="$(pwd)/venv"
        VENV_ACTIVATED=true
    elif [ -f "venv/Scripts/python.exe" ]; then
        export PATH="$(pwd)/venv/Scripts:$PATH"
        export VIRTUAL_ENV="$(pwd)/venv"
        VENV_ACTIVATED=true
    fi
fi

if [ "$VENV_ACTIVATED" = false ]; then
    echo "Error: Virtual environment not found or cannot be activated."
    echo "Looking for:"
    echo "  - venv/bin/activate (Linux/Mac)"
    echo "  - venv/Scripts/activate (Windows)"
    echo ""
    echo "If you're in WSL with a Windows venv, you may need to:"
    echo "  1. Create a Linux venv: python3 -m venv venv"
    echo "  2. Or activate the venv before running this script"
    exit 1
fi

echo "========================================"
echo "  Enhanced Autonomous Trading Bot"
echo "  Background Mode"
echo "========================================"
echo ""
echo "Starting bot in background..."
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"
echo ""

# Function to run bot
run_bot() {
    # Determine Python executable
    PYTHON_CMD="python"
    if command -v python3 > /dev/null 2>&1; then
        PYTHON_CMD="python3"
    fi
    
    # If using Windows venv in WSL, try to use the Windows Python
    if [ -f "venv/Scripts/python.exe" ]; then
        PYTHON_CMD="venv/Scripts/python.exe"
    elif [ -f "venv/bin/python" ]; then
        PYTHON_CMD="venv/bin/python"
    fi
    
    # Set encoding to UTF-8 to avoid Unicode errors
    export PYTHONIOENCODING=utf-8
    
    # Try enhanced bot first
    $PYTHON_CMD run_dashboard_with_bot.py >> "$LOG_FILE" 2>&1
    ENHANCED_EXIT_CODE=$?
    
    # If enhanced bot fails, try simple bot
    if [ $ENHANCED_EXIT_CODE -ne 0 ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Enhanced bot failed (exit code: $ENHANCED_EXIT_CODE), trying simple bot..." >> "$LOG_FILE"
        $PYTHON_CMD simple_bot.py >> "$LOG_FILE" 2>&1
    fi
}

# Run bot in background
run_bot &
BOT_PID=$!

# Save PID
echo $BOT_PID > "$PID_FILE"

echo "Bot started with PID: $BOT_PID"
echo ""
echo "To view logs in real-time:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop the bot:"
echo "  ./stop_bot.sh"
echo ""
echo "To check if bot is running:"
echo "  ps -p $BOT_PID"
echo ""

# Wait a moment to check if bot started successfully
sleep 2

if ps -p $BOT_PID > /dev/null 2>&1; then
    echo "Bot is running successfully!"
    echo "Background process started. You can close this terminal."
else
    echo "Warning: Bot process may have exited immediately."
    echo "Check the log file for errors: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
