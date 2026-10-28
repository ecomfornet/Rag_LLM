#!/usr/bin/env bash
# ==============================================================================
#  GraphRAG Full Project Backup Script
#  Project: /home/llm.kytisa.com/v2 (GraphRAG Legal Documents)
#
#  Backs up:
#    1. Source code + config (.py, .md, .txt, .yml, .env, docs)
#    2. data/ directory   (cleaned files, chunks, embeddings, bm25_cache)
#    3. Neo4j             -> native dump       (db/neo4j_database.dump)
#    4. Qdrant            -> native snapshot   (db/legal_docs_graphrag.snapshot)
#    5. Redis             -> BGSAVE dump + redis_export.json
#    6. Manifest (BACKUP_MANIFEST.json), SHA256SUMS, final .tar.gz
#
#  Output layout matches: python import_databases.py --input ./
#
#  Usage:
#    ./backup_graphrag.sh                  # full backup (stops Neo4j briefly)
#    ./backup_graphrag.sh --no-tar         # skip final tar.gz (keep staging dir)
#    ./backup_graphrag.sh --keep-staging   # tar.gz AND keep staging dir
#    ./backup_graphrag.sh --skip-neo4j     # skip Neo4j dump (no container stop)
#    ./backup_graphrag.sh --keep 5         # keep last 5 archives (default: 3)
# ==============================================================================
set -uo pipefail

# --------------------------------- Config ------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
BACKUP_BASE="$PROJECT_ROOT/backups"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$BACKUP_BASE/graphrag_backup_${TIMESTAMP}"
ARCHIVE="$BACKUP_BASE/graphrag_backup_${TIMESTAMP}.tar.gz"

# Services (defaults match docker-compose.yml)
NEO4J_CONTAINER="graphrag_neo4j"
QDRANT_CONTAINER="graphrag_qdrant"
REDIS_CONTAINER="graphrag_redis"
NEO4J_IMAGE="neo4j:5.14-community"
NEO4J_DATABASE="neo4j"
QDRANT_HOST="localhost"
QDRANT_PORT="6340"
QDRANT_COLLECTION="legal_docs_graphrag"
REDIS_HOST="localhost"
REDIS_PORT="6380"
REDIS_DB="0"
REDIS_PASSWORD=""
KEEP_LAST=3

MAKE_TAR=1
KEEP_STAGING=0
SKIP_NEO4J=0

# Overrides from .env (read-only, no secrets printed)
if [ -f "$PROJECT_ROOT/.env" ]; then
    v="$(grep -E '^QDRANT_PORT=' "$PROJECT_ROOT/.env" | cut -d= -f2- | tr -d '[:space:]')"
    [ -n "$v" ] && QDRANT_PORT="$v"
    v="$(grep -E '^QDRANT_HOST=' "$PROJECT_ROOT/.env" | cut -d= -f2- | tr -d '[:space:]')"
    [ -n "$v" ] && QDRANT_HOST="$v"
    v="$(grep -E '^REDIS_HOST=' "$PROJECT_ROOT/.env" | cut -d= -f2- | tr -d '[:space:]')"
    [ -n "$v" ] && REDIS_HOST="$v"
    v="$(grep -E '^REDIS_PORT=' "$PROJECT_ROOT/.env" | cut -d= -f2- | tr -d '[:space:]')"
    [ -n "$v" ] && REDIS_PORT="$v"
    v="$(grep -E '^REDIS_DB=' "$PROJECT_ROOT/.env" | cut -d= -f2- | tr -d '[:space:]')"
    [ -n "$v" ] && REDIS_DB="$v"
    v="$(grep -E '^REDIS_PASSWORD=' "$PROJECT_ROOT/.env" | cut -d= -f2- | tr -d '[:space:]')"
    [ -n "$v" ] && REDIS_PASSWORD="$v"
fi

# Python (prefer project venv)
if [ -x "$PROJECT_ROOT/venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
else
    PYTHON_BIN="$(command -v python3 || true)"
fi

# --------------------------------- Logging -----------------------------------
LOG_FILE="$PROJECT_ROOT/logs/backup_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"
log()  { local m; m="$(date '+%Y-%m-%d %H:%M:%S') [INFO]  $*"; echo "$m"; echo "$m" >>"$LOG_FILE"; }
warn() { local m; m="$(date '+%Y-%m-%d %H:%M:%S') [WARN]  $*"; echo "$m"; echo "$m" >>"$LOG_FILE"; }
err()  { local m; m="$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $*"; echo "$m" >&2; echo "$m" >>"$LOG_FILE"; }

# --------------------------------- Args --------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --no-tar)       MAKE_TAR=0; KEEP_STAGING=1 ;;
        --keep-staging) KEEP_STAGING=1 ;;
        --skip-neo4j)   SKIP_NEO4J=1 ;;
        --keep)         KEEP_LAST="${2:-3}"; shift ;;
        -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)              err "Unknown option: $1"; exit 2 ;;
    esac
    shift
done

# ----------------------------- Prerequisites ---------------------------------
log "=============================================================="
log " GraphRAG FULL PROJECT BACKUP started"
log " Project root : $PROJECT_ROOT"
log " Backup dir   : $BACKUP_DIR"
log "=============================================================="

command -v docker >/dev/null 2>&1 || { err "docker not found in PATH"; exit 1; }
command -v curl    >/dev/null 2>&1 || { err "curl not found in PATH"; exit 1; }

for c in "$NEO4J_CONTAINER" "$QDRANT_CONTAINER" "$REDIS_CONTAINER"; do
    docker ps --format '{{.Names}}' | grep -qx "$c" \
        || warn "Container not running: $c"
done

mkdir -p "$BACKUP_DIR/db"
cd "$PROJECT_ROOT"

# ----------------------------- 1. Source files -------------------------------
log "Step 1/6: Copying source code and configuration..."
log "  (excluding: venv, __pycache__, logs, backups, exports, db/, *.log, nohup.out)"

rsync -a \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude 'logs/' \
    --exclude 'backups/' \
    --exclude 'exports/' \
    --exclude 'db/' \
    --exclude '*.log' \
    --exclude 'nohup.out' \
    --exclude '.DS_Store' \
    ./ "$BACKUP_DIR/" >>"$LOG_FILE" 2>&1 \
    && log "  [OK] Source files + .env copied" \
    || err "  rsync of source files FAILED"

# .env contains secrets - restrict permissions inside the backup
chmod 600 "$BACKUP_DIR/.env" 2>/dev/null || true

# ----------------------------- 2. Data directory -----------------------------
log "Step 2/6: Copying data/ directory (may take a while)..."
if [ -d "$PROJECT_ROOT/data" ]; then
    rsync -a --info=progress2 "$PROJECT_ROOT/data/" "$BACKUP_DIR/data/" >>"$LOG_FILE" 2>&1 \
        && log "  [OK] data/ copied: $(du -sh "$BACKUP_DIR/data" | cut -f1)" \
        || err "  rsync of data/ FAILED"
else
    warn "  data/ directory not found - skipped"
fi

# ----------------------------- 3. Neo4j native dump --------------------------
dump_neo4j() {
    log "Step 3/6: Neo4j native dump (brief container stop required)..."

    if ! docker ps --format '{{.Names}}' | grep -qx "$NEO4J_CONTAINER"; then
        warn "  $NEO4J_CONTAINER is not running - attempting dump anyway"
    fi

    log "  Stopping container $NEO4J_CONTAINER..."
    docker stop "$NEO4J_CONTAINER" >>"$LOG_FILE" 2>&1

    log "  Running ephemeral neo4j-admin dump..."
    docker run --rm --volumes-from "$NEO4J_CONTAINER" "$NEO4J_IMAGE" \
        neo4j-admin database dump "$NEO4J_DATABASE" \
        --to-path=/data --overwrite-destination=true >>"$LOG_FILE" 2>&1
    dump_ok=$?

    log "  Copying dump out of container..."
    docker cp "$NEO4J_CONTAINER:/data/$NEO4J_DATABASE.dump" \
        "$BACKUP_DIR/db/neo4j_database.dump" >>"$LOG_FILE" 2>&1 || true

    log "  Cleaning dump file inside volume (best effort)..."
    docker run --rm --volumes-from "$NEO4J_CONTAINER" "$NEO4J_IMAGE" \
        rm -f "/data/$NEO4J_DATABASE.dump" >>"$LOG_FILE" 2>&1 || true

    log "  Restarting container $NEO4J_CONTAINER..."
    docker start "$NEO4J_CONTAINER" >>"$LOG_FILE" 2>&1

    if [ "$dump_ok" -eq 0 ] && [ -s "$BACKUP_DIR/db/neo4j_database.dump" ]; then
        log "  [OK] Neo4j dump saved: db/neo4j_database.dump ($(du -h "$BACKUP_DIR/db/neo4j_database.dump" | cut -f1))"
    else
        err "  Neo4j dump FAILED - check $LOG_FILE"
    fi

    log "  Waiting for Bolt to come back online (max 120s)..."
    waited=0
    until docker exec "$NEO4J_CONTAINER" cypher-shell -u neo4j -p graphrag123 "RETURN 1;" >/dev/null 2>&1 \
       || docker exec "$NEO4J_CONTAINER" cypher-shell "RETURN 1;" >/dev/null 2>&1; do
        waited=$((waited + 5))
        if [ "$waited" -ge 120 ]; then
            warn "  Neo4j not ready after ${waited}s (it may still be starting)"
            break
        fi
        sleep 5
    done
    log "  [OK] Neo4j restarted (waited ${waited}s)"
}

if [ "$SKIP_NEO4J" -eq 0 ]; then
    dump_neo4j
else
    log "Step 3/6: Neo4j dump skipped (--skip-neo4j)"
fi

# ----------------------------- 4. Qdrant native snapshot ---------------------
dump_qdrant() {
    log "Step 4/6: Qdrant native snapshot via REST API..."

    local base_url="http://${QDRANT_HOST}:${QDRANT_PORT}"
    local resp snap_name

    resp="$(curl -sf -X POST "${base_url}/collections/${QDRANT_COLLECTION}/snapshots?wait=true" -m 900)" || {
        err "  Could not reach Qdrant at ${base_url} - snapshot failed"
        return 1
    }

    snap_name="$(printf '%s' "$resp" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["name"])' 2>/dev/null \
        || printf '%s' "$resp" | sed -n 's/.*"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1)"

    [ -n "$snap_name" ] || { err "  Could not parse snapshot name from: $resp"; return 1; }
    log "  Snapshot created on server: $snap_name"

    log "  Downloading snapshot..."
    curl -sf "${base_url}/collections/${QDRANT_COLLECTION}/snapshots/${snap_name}" \
        -m 1800 -o "$BACKUP_DIR/db/${QDRANT_COLLECTION}.snapshot" || {
        err "  Snapshot download failed"
        return 1
    }

    log "  [OK] Qdrant snapshot saved: db/${QDRANT_COLLECTION}.snapshot ($(du -h "$BACKUP_DIR/db/${QDRANT_COLLECTION}.snapshot" | cut -f1))"

    log "  Deleting snapshot on server (best effort)..."
    curl -sf -X DELETE "${base_url}/collections/${QDRANT_COLLECTION}/snapshots/${snap_name}" >/dev/null 2>&1 || true
}

dump_qdrant || warn "Qdrant snapshot FAILED (continuing with the rest)"

# ----------------------------- 5. Redis backup -------------------------------
dump_redis() {
    log "Step 5/6: Redis backup..."

    if docker ps --format '{{.Names}}' | grep -qx "$REDIS_CONTAINER"; then
        log "  Triggering BGSAVE..."
        docker exec "$REDIS_CONTAINER" redis-cli BGSAVE >>"$LOG_FILE" 2>&1 || true
        sleep 3
        if docker cp "$REDIS_CONTAINER:/data/dump.rdb" "$BACKUP_DIR/db/redis_dump.rdb" >>"$LOG_FILE" 2>&1; then
            log "  [OK] Native redis_dump.rdb saved"
        else
            warn "  Could not copy dump.rdb"
        fi
    else
        warn "  $REDIS_CONTAINER not running - skipping native dump"
    fi

    # redis_export.json (same format as export_project.py -> import_databases.py compatible)
    if [ -n "$PYTHON_BIN" ]; then
        log "  Generating redis_export.json..."
        export REDIS_HOST REDIS_PORT REDIS_DB REDIS_PASSWORD
        if "$PYTHON_BIN" - "$BACKUP_DIR/db/redis_export.json" <<'PYEOF' >>"$LOG_FILE" 2>&1; then
import os, sys, json, redis

out_path = sys.argv[1]
client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6380")),
    db=int(os.getenv("REDIS_DB", "0")),
    password=os.getenv("REDIS_PASSWORD") or None,
)
try:
    keys = client.keys("*")
    data = []
    for k in keys:
        t = client.type(k).decode("utf-8")
        val = None
        try:
            if t == "string":
                v = client.get(k)
                val = v.decode("utf-8") if v is not None else None
            elif t == "hash":
                val = {fk.decode(): fv.decode() for fk, fv in client.hgetall(k).items()}
        except Exception:
            pass
        data.append({"key": k.decode("utf-8"), "type": t, "value": val})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Redis: {len(data)} keys exported to JSON")
finally:
    client.close()
PYEOF
            log "  [OK] redis_export.json generated"
        else
            warn "  redis_export.json generation failed"
        fi
    else
        warn "  No python available - skipping redis_export.json"
    fi
}

dump_redis

# ----------------------------- 6. Manifest + archive -------------------------
log "Step 6/6: Manifest, checksums and final archive..."

( cd "$BACKUP_DIR" && find . -type f -print0 | sort -z | xargs -0 sha256sum ) \
    > "$BACKUP_DIR/SHA256SUMS" 2>>"$LOG_FILE" \
    && log "  [OK] SHA256SUMS written"

{
    echo "{"
    echo "  \"backup_name\": \"graphrag_backup_${TIMESTAMP}\","
    echo "  \"timestamp\": \"${TIMESTAMP}\","
    echo "  \"project_root\": \"${PROJECT_ROOT}\","
    echo "  \"created_by\": \"backup_graphrag.sh\","
    echo "  \"neo4j_dump\": \"db/neo4j_database.dump\","
    echo "  \"qdrant_snapshot\": \"db/${QDRANT_COLLECTION}.snapshot\","
    echo "  \"redis_files\": [\"db/redis_dump.rdb\", \"db/redis_export.json\"],"
    echo "  \"restore\": \"python import_databases.py --input ./\""
    echo "}"
} > "$BACKUP_DIR/BACKUP_MANIFEST.json"
log "  [OK] BACKUP_MANIFEST.json written"

if [ "$MAKE_TAR" -eq 1 ]; then
    log "  Creating final archive (may take several minutes)..."
    if tar -czf "$ARCHIVE" -C "$BACKUP_BASE" "graphrag_backup_${TIMESTAMP}" 2>>"$LOG_FILE"; then
        log "  [OK] Archive: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
        if [ "$KEEP_STAGING" -eq 0 ]; then
            rm -rf "$BACKUP_DIR"
            log "  [OK] Staging directory removed"
        fi
    else
        err "  tar archive FAILED - staging kept: $BACKUP_DIR"
    fi
else
    log "  Archive skipped (--no-tar)"
fi

# ----------------------------- Rotation --------------------------------------
log "Rotation: keeping last $KEEP_LAST backups..."
if [ "$MAKE_TAR" -eq 1 ]; then
    count=0
    while IFS= read -r old; do
        count=$((count + 1))
        if [ "$count" -gt "$KEEP_LAST" ]; then
            rm -f "$old"
            log "  Deleted old backup: $old"
        fi
    done < <(ls -1t "$BACKUP_BASE"/graphrag_backup_*.tar.gz 2>/dev/null)
fi
if [ "$MAKE_TAR" -eq 0 ]; then
    count=0
    while IFS= read -r old; do
        count=$((count + 1))
        if [ "$count" -gt "$KEEP_LAST" ]; then
            rm -rf "$old"
            log "  Deleted old backup dir: $old"
        fi
    done < <(ls -1dt "$BACKUP_BASE"/graphrag_backup_*_[0-9]* 2>/dev/null)
fi

# ----------------------------- Summary ---------------------------------------
FINAL_ITEM="$ARCHIVE"
[ "$MAKE_TAR" -eq 0 ] && FINAL_ITEM="$BACKUP_DIR"

log "=============================================================="
log " BACKUP COMPLETE"
log " Output      : $FINAL_ITEM"
log " Size        : $(du -sh "$FINAL_ITEM" 2>/dev/null | cut -f1 || echo 'n/a')"
log " Log file    : $LOG_FILE"
log " Restore     : extract, then: python import_databases.py --input ./"
log "=============================================================="
