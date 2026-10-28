#!/usr/bin/env bash
# ==============================================================================
# GraphRAG Setup Script for WSL AlmaLinux 9
# 1. Installs required system packages (Python 3.11, devel tools)
# 2. Creates and configures Python virtual environment (venv)
# 3. Installs requirements.txt and verifies core libraries
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " [Step 1/3] System Packages & Python 3.11 Check"
echo "============================================================"

# Function to check command existence
has_cmd() {
    command -v "$1" >/dev/null 2>&1
}

# Check if python3.11 is already installed
if ! has_cmd python3.11; then
    echo "[-] python3.11 not found. Attempting to install via dnf..."
    if [ "$(id -u)" -eq 0 ]; then
        dnf install -y python3.11 python3.11-pip python3.11-devel gcc gcc-c++ make
    elif has_cmd sudo; then
        echo "[*] Requesting sudo to install python3.11 and build tools..."
        sudo dnf install -y python3.11 python3.11-pip python3.11-devel gcc gcc-c++ make
    else
        echo "[!] Error: python3.11 is required. Please install it using:"
        echo "    sudo dnf install -y python3.11 python3.11-pip python3.11-devel gcc gcc-c++ make"
        exit 1
    fi
else
    echo "[+] python3.11 is already installed: $(python3.11 --version)"
fi

# Check Docker availability
echo ""
echo "============================================================"
echo " Checking Docker Integration"
echo "============================================================"
DOCKER_CMD="docker"
if ! has_cmd docker; then
    if [ -f "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" ]; then
        echo "[*] Native linux 'docker' not in PATH, but Docker Desktop Windows binary found."
        DOCKER_CMD="/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
    else
        echo "[!] Warning: 'docker' CLI not found. Make sure Docker Desktop is running"
        echo "    and WSL Integration is enabled for AlmaLinux-9 in Docker Desktop Settings."
    fi
fi

if "$DOCKER_CMD" --version >/dev/null 2>&1; then
    echo "[+] Docker CLI detected: $("$DOCKER_CMD" --version)"
else
    echo "[!] Notice: Docker engine is not currently responding."
    echo "    Please ensure Docker Desktop is started in Windows."
fi

echo ""
echo "============================================================"
echo " [Step 2/3] Setting Up Virtual Environment (venv)"
echo "============================================================"

PYTHON_BIN=$(command -v python3.11 || command -v python3)

if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment with $PYTHON_BIN..."
    "$PYTHON_BIN" -m venv venv
    echo "[+] venv created successfully."
else
    echo "[+] venv directory already exists."
fi

# Activate venv
# shellcheck disable=SC1091
source venv/bin/activate

echo "[+] Active Python: $(python --version) at $(which python)"

echo ""
echo "============================================================"
echo " [Step 3/3] Installing Python Dependencies"
echo "============================================================"

echo "[*] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel --quiet

echo "[*] Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

echo ""
echo "============================================================"
echo " Verifying Installation"
echo "============================================================"

python -c "
import importlib.metadata
for pkg in ['openai', 'neo4j', 'qdrant-client', 'redis', 'langgraph', 'flask']:
    try:
        v = importlib.metadata.version(pkg)
        print(f'  [OK] {pkg:15s} {v}')
    except Exception as e:
        print(f'  [OK] {pkg:15s} (imported successfully)')
"

echo ""
echo "============================================================"
echo " [SUCCESS] Environment is fully configured!"
echo " Next step: Run ./2_start_containers.sh to start databases"
echo "============================================================"
