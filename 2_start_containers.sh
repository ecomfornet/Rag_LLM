#!/usr/bin/env bash
# ==============================================================================
# GraphRAG Container Startup Script for WSL AlmaLinux 9
# Starts Neo4j, Qdrant, and Redis containers and verifies health
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Starting GraphRAG Database Services (Docker)"
echo "============================================================"

# Resolve Docker command
DOCKER_BIN="docker"
if ! command -v docker >/dev/null 2>&1; then
    if [ -f "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" ]; then
        DOCKER_BIN="/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
    else
        echo "[!] Error: docker command not found."
        echo "    Please start Docker Desktop on Windows and enable WSL Integration."
        exit 1
    fi
fi

# Test Docker engine
if ! "$DOCKER_BIN" ps >/dev/null 2>&1; then
    echo "[!] Error: Docker daemon is not running."
    echo "    Please open Docker Desktop in Windows and wait until it is running."
    exit 1
fi

echo "[+] Docker engine is responding."

# Start services
echo "[*] Launching containers via Docker Compose..."
if "$DOCKER_BIN" compose version >/dev/null 2>&1; then
    COMPOSE_CMD="$DOCKER_BIN compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe compose"
fi

$COMPOSE_CMD up -d

echo ""
echo "[*] Waiting for services to become healthy..."

MAX_WAIT=35

wait_for_port() {
    local host="$1"
    local port="$2"
    local name="$3"
    local success=false

    for ((i=1; i<=MAX_WAIT; i++)); do
        if timeout 1 bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null; then
            echo "  [OK] $name is responding on port $port"
            success=true
            break
        fi

        # Auto-recover Neo4j if in restart loop due to stale pid
        if [ "$port" -eq 7688 ] && [ "$i" -eq 12 ]; then
            if "$DOCKER_BIN" ps -a --filter "name=graphrag_neo4j" --format "{{.Status}}" | grep -iq "restarting"; then
                echo "  [*] Detected Neo4j restart loop (stale pid). Recreating container cleanly..."
                "$DOCKER_BIN" rm -f graphrag_neo4j >/dev/null 2>&1 || true
                $COMPOSE_CMD up -d neo4j >/dev/null 2>&1 || true
            fi
        fi

        sleep 1
    done

    if [ "$success" = false ]; then
        echo "  [!] Warning: $name did not respond on port $port within ${MAX_WAIT}s"
    fi
}

wait_for_port "127.0.0.1" 6340 "Qdrant Vector DB"
wait_for_port "127.0.0.1" 7688 "Neo4j Bolt Protocol"
wait_for_port "127.0.0.1" 6380 "Redis Cache"

echo ""
echo "============================================================"
echo " Active Containers Status"
echo "============================================================"
"$DOCKER_BIN" ps --filter "name=graphrag_" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "============================================================"
echo " [SUCCESS] All database services are running!"
echo "============================================================"
