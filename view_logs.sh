#!/bin/bash

# View bot logs in real-time

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$SCRIPT_DIR/logs"

if [ ! -d "$LOG_DIR" ]; then
    echo "Log directory not found: $LOG_DIR"
    exit 1
fi

# Find the most recent log file
LATEST_LOG=$(ls -t "$LOG_DIR"/bot_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No log files found in $LOG_DIR"
    exit 1
fi

echo "Viewing log file: $LATEST_LOG"
echo "Press Ctrl+C to exit"
echo ""
echo "========================================"
echo ""

# Tail the log file
tail -f "$LATEST_LOG"
