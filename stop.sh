#!/bin/bash
# Stop the CoPE-A-9B inference service

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f cope.pid ]; then
    echo "No PID file found. Service may not be running."
    exit 0
fi

PID=$(cat cope.pid)

if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping CoPE-A-9B service (PID: $PID)..."
    kill "$PID"
    
    # Wait for process to terminate
    for i in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    
    # Force kill if still running
    if kill -0 "$PID" 2>/dev/null; then
        echo "Process did not stop gracefully, forcing..."
        kill -9 "$PID"
    fi
    
    rm -f cope.pid
    echo "Service stopped."
else
    echo "Process $PID not running. Cleaning up PID file."
    rm -f cope.pid
fi
