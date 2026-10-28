#!/usr/bin/env bash
# ==============================================================================
# GraphRAG Master Runner for WSL AlmaLinux 9
# One-click execution:
# 1. Environment & venv verification
# 2. Database container startup
# 3. Native database restoration (on first run or with --restore)
# 4. Web Application launch
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FORCE_RESTORE=0
for arg in "$@"; do
    if [ "$arg" == "--restore" ] || [ "$arg" == "-r" ] || [ "$arg" == "--force-restore" ]; then
        FORCE_RESTORE=1
    fi
done

echo "============================================================"
echo " Starting GraphRAG System (Master Launcher)"
echo "============================================================"

# Step 1: Check / Set up Python Environment
if [ ! -d "venv" ]; then
    echo "[*] Step 1/3: Virtual environment missing. Running setup..."
    ./1_setup_env.sh
else
    echo "[+] Step 1/3: Python virtual environment detected."
fi

# Step 2: Ensure Containers are up
echo ""
echo "[*] Step 2/3: Starting database containers..."
./2_start_containers.sh

# Step 3: Check Database Restoration
echo ""
RESTORE_FLAG=".restored"
if [ ! -f "$RESTORE_FLAG" ] || [ "$FORCE_RESTORE" -eq 1 ]; then
    echo "[*] Step 3/3: First-time database restore required..."
    ./3_restore_db.sh
    touch "$RESTORE_FLAG"
    echo "[+] Created restoration flag: $RESTORE_FLAG"
else
    echo "[+] Step 3/3: Database already restored (use --restore to re-import)."
fi

# Step 4: Start the Web App
echo ""
echo "[*] Launching Web UI..."
./4_start_app.sh
