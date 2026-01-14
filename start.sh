#!/bin/bash
# Start the CoPE-A-9B inference service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source conda
CONDA_BASE="/opt/conda"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cope

# Check if already running
if [ -f cope.pid ]; then
    PID=$(cat cope.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo "Service already running (PID: $PID)"
        exit 1
    else
        rm -f cope.pid
    fi
fi

echo "Starting CoPE-A-9B inference service..."
echo "Logs will be written to: $SCRIPT_DIR/cope.log"

# Start the service in the background
nohup python cope_service.py > cope.log 2>&1 &
PID=$!
echo $PID > cope.pid

echo "Service started with PID: $PID"
echo "Waiting for model to load (this may take 1-2 minutes)..."
echo ""
echo "To check status:  curl http://localhost:8000/health"
echo "To view logs:     tail -f $SCRIPT_DIR/cope.log"
echo "To stop:          ./stop.sh"
