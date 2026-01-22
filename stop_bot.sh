#!/bin/bash

# Stop the background trading bot

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PID_FILE="$SCRIPT_DIR/bot.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Bot is not running (PID file not found)"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Bot is not running (process $PID not found)"
    rm -f "$PID_FILE"
    exit 1
fi

echo "Stopping bot (PID: $PID)..."
kill "$PID"

# Wait for process to stop
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "Bot stopped successfully"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# If still running, force kill
if ps -p "$PID" > /dev/null 2>&1; then
    echo "Force killing bot..."
    kill -9 "$PID"
    sleep 1
    rm -f "$PID_FILE"
    echo "Bot force stopped"
else
    echo "Bot stopped successfully"
    rm -f "$PID_FILE"
fi
