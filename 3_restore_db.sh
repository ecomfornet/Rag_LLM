#!/usr/bin/env bash
# ==============================================================================
# GraphRAG Database Restoration Script for WSL AlmaLinux 9
# Restores Qdrant snapshot, Neo4j database dump, and Redis cached keys
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Restoring GraphRAG Databases from Backup"
echo "============================================================"

# Check venv
if [ ! -d "venv" ]; then
    echo "[!] Virtual environment 'venv' not found. Running ./1_setup_env.sh first..."
    ./1_setup_env.sh
fi

# shellcheck disable=SC1091
source venv/bin/activate

# Ensure docker CLI is in PATH for python's subprocess calls
if ! command -v docker >/dev/null 2>&1; then
    if [ -f "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" ]; then
        echo "[*] Adding Docker Desktop bin path to PATH..."
        export PATH="/mnt/c/Program Files/Docker/Docker/resources/bin:$PATH"
    fi
fi

# Check required backup files
echo "[*] Checking backup files in db/..."
MISSING=0
for f in "db/legal_docs_graphrag.snapshot" "db/neo4j_database.dump" "db/redis_export.json"; do
    if [ ! -f "$f" ]; then
        echo "  [!] Missing backup file: $f"
        MISSING=1
    else
        echo "  [OK] Found $f ($(du -h "$f" | cut -f1))"
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo "[!] Error: Missing database backup files. Please ensure the backup archive is extracted."
    exit 1
fi

# Ensure containers are up
echo ""
echo "[*] Ensuring database containers are running..."
./2_start_containers.sh

# Run native import
echo ""
echo "============================================================"
echo " Executing import_databases.py"
echo "============================================================"
python import_databases.py --input ./

echo ""
echo "============================================================"
echo " Verifying Data Storage & Node Counts"
echo "============================================================"
python check_storage.py

echo ""
echo "============================================================"
echo " Running System Diagnostic Tests"
echo "============================================================"
python test_system.py

echo ""
echo "============================================================"
echo " [SUCCESS] Database restoration and verification complete!"
echo " Start the Web UI by running: ./4_start_app.sh"
echo "============================================================"
