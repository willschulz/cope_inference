#!/bin/bash
# Uninstall CoPE vLLM systemd service

set -e

SERVICE_NAME="cope-vllm"

echo "Uninstalling CoPE-A-9B vLLM systemd service..."

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo ./uninstall_service.sh"
    exit 1
fi

# Stop service if running
if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    echo "Stopping service..."
    systemctl stop "$SERVICE_NAME"
fi

# Disable service
if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
    echo "Disabling service..."
    systemctl disable "$SERVICE_NAME"
fi

# Remove service file
if [ -f "/etc/systemd/system/$SERVICE_NAME.service" ]; then
    echo "Removing service file..."
    rm -f "/etc/systemd/system/$SERVICE_NAME.service"
fi

# Reload systemd daemon
systemctl daemon-reload

echo ""
echo "Service uninstalled successfully!"
echo "You can now use ./start_vllm.sh and ./stop.sh for manual control."
