#!/usr/bin/env bash
# ==============================================================================
# GraphRAG Web Application Launcher for WSL AlmaLinux 9
# Runs the Flask Web UI on port 5000
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "[!] Virtual environment 'venv' not found. Please run ./1_setup_env.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

# Ensure containers are up
./2_start_containers.sh

# Detect WSL host IP
WSL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")

echo ""
echo "============================================================"
echo " Starting GraphRAG Flask Web Application"
echo "============================================================"
echo "  Localhost URL: http://localhost:5000"
echo "  WSL IP URL:    http://${WSL_IP}:5000"
echo "  Health Check:  http://${WSL_IP}:5000/health"
echo "  Logs:          logs/app.log"
echo "  Press Ctrl+C to stop the server."
echo "============================================================"
echo ""

mkdir -p logs
exec python app.py
