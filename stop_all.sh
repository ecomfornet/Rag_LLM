#!/usr/bin/env bash
# ==============================================================================
# GraphRAG Service Shutdown Script
# Stops Flask server and halts all Docker containers cleanly
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Stopping GraphRAG Services"
echo "============================================================"

# Stop Flask app if running in background
pkill -f "python.*app.py" 2>/dev/null && echo "[+] Stopped running Flask app process." || echo "[*] No background Flask app was running."

# Resolve Docker command
DOCKER_BIN="docker"
if ! command -v docker >/dev/null 2>&1; then
    if [ -f "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" ]; then
        DOCKER_BIN="/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
    fi
fi

if "$DOCKER_BIN" ps >/dev/null 2>&1; then
    echo "[*] Stopping database containers (preserving volume data)..."
    if "$DOCKER_BIN" compose version >/dev/null 2>&1; then
        "$DOCKER_BIN" compose stop
    elif command -v docker-compose >/dev/null 2>&1; then
        docker-compose stop
    else
        "$DOCKER_BIN" stop graphrag_neo4j graphrag_qdrant graphrag_redis 2>/dev/null || true
    fi
    echo "[+] Database containers stopped."
else
    echo "[*] Docker daemon not reachable or already stopped."
fi

echo "============================================================"
echo " All services stopped cleanly."
echo "============================================================"
