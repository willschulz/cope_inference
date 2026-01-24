#!/bin/bash
# Install CoPE vLLM as a systemd service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/cope-vllm.service"
SERVICE_NAME="cope-vllm"

echo "Installing CoPE-A-9B vLLM as a systemd service..."

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo ./install_service.sh"
    exit 1
fi

# Stop existing manual service if running
if [ -f "$SCRIPT_DIR/cope.pid" ]; then
    PID=$(cat "$SCRIPT_DIR/cope.pid")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping existing manual service (PID: $PID)..."
        kill "$PID" 2>/dev/null || true
        sleep 3
        kill -9 "$PID" 2>/dev/null || true
    fi
    rm -f "$SCRIPT_DIR/cope.pid"
fi

# Copy service file to systemd directory
cp "$SERVICE_FILE" /etc/systemd/system/

# Reload systemd daemon
systemctl daemon-reload

# Enable service to start on boot
systemctl enable "$SERVICE_NAME"

echo ""
echo "Service installed successfully!"
echo ""
echo "Commands:"
echo "  Start:   sudo systemctl start $SERVICE_NAME"
echo "  Stop:    sudo systemctl stop $SERVICE_NAME"
echo "  Restart: sudo systemctl restart $SERVICE_NAME"
echo "  Status:  sudo systemctl status $SERVICE_NAME"
echo "  Logs:    sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "The service will start automatically on boot."
echo ""
read -p "Start the service now? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    systemctl start "$SERVICE_NAME"
    echo ""
    echo "Service started. Checking status..."
    sleep 2
    systemctl status "$SERVICE_NAME" --no-pager || true
fi
