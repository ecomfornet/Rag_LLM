"""
Complete Project Import Script for GraphRAG Legal Documents
Supports native DB restores (Neo4j .dump, Qdrant .snapshot) and JSON fallbacks
"""
import os
import sys
import json
import time
import logging
import argparse
import subprocess
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("project-import")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# DB Settings (matching config.py)
QDRANT_HOST       = os.getenv("QDRANT_HOST",     "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", "6340"))
QDRANT_COLLECTION = "legal_docs_graphrag"
QDRANT_USE_HTTPS  = os.getenv("QDRANT_USE_HTTPS", "false").lower() == "true"
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY",  None)

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7688")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag123")
NEO4J_DATABASE = "neo4j"
NEO4J_CONTAINER = "graphrag_neo4j"

def _get_qdrant_url():
    scheme = "https" if QDRANT_USE_HTTPS else "http"
    return f"{scheme}://{QDRANT_HOST}:{QDRANT_PORT}"

def _get_qdrant_headers():
    h = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        h["api-key"] = QDRANT_API_KEY
    return h

# --- Import Functions ---

def import_qdrant_native(input_dir: Path):
    """Restore Qdrant from native .snapshot file."""
    log.info("Restoring Qdrant from native snapshot...")
    snap_path = list(input_dir.glob("db/*.snapshot")) or list(input_dir.glob("db/qdrant/*.snapshot")) or list(input_dir.glob("qdrant/*.snapshot"))
    if not snap_path:
        log.warning("  [!] No .snapshot file found for Qdrant.")
        return False
    
    snap_path = snap_path[0]
    base_url = _get_qdrant_url()
    headers = _get_qdrant_headers()
    
    log.info(f"  Uploading snapshot: {snap_path.name}...")
    
    # 1. Delete existing collection if any
    requests.delete(f"{base_url}/collections/{QDRANT_COLLECTION}", headers=headers)
    
    # 2. Upload snapshot
    with open(snap_path, "rb") as f:
        upload_headers = {k: v for k, v in headers.items() if k != "Content-Type"}
        resp = requests.post(
            f"{base_url}/collections/{QDRANT_COLLECTION}/snapshots/upload?wait=true",
            headers=upload_headers,
            files={"snapshot": (snap_path.name, f, "application/octet-stream")},
            timeout=600
        )
    
    if resp.status_code == 200:
        log.info("  [OK] Qdrant snapshot restored successfully.")
        return True
    else:
        log.error(f"  [X] Qdrant restoration failed: {resp.text}")
        return False

def import_neo4j_native(input_dir: Path):
    """Restore Neo4j from native .dump file."""
    log.info("Restoring Neo4j from native dump...")
    dump_path = input_dir / "db" / "neo4j_database.dump"
    if not dump_path.exists():
        dump_path = input_dir / "db" / "neo4j" / "neo4j_database.dump"
        
    if not dump_path.exists():
        log.warning("  [!] No native .dump file found for Neo4j.")
        return False
        
    try:
        # 1. Stop the container to allow offline load in Community Edition
        log.info(f"  Stopping container {NEO4J_CONTAINER} for offline restore...")
        subprocess.run(["docker", "stop", NEO4J_CONTAINER], check=True, capture_output=True)
        
        # 2. Copy dump to a temp volume location (if using a persistent volume, we copy to the host volume or use container mounting)
        # Wait, the easiest way is to start an ephemeral container with the same volume, copy into it, then load.
        # But we can also just `docker cp` to a temporary dummy container bridging the volume, or directly to the stopped container.
        # Actually `docker cp` works on STOPPED containers!
        log.info(f"  Copying dump to stopped container {NEO4J_CONTAINER}...")
        
        # Create a tiny ephemeral container just to mount the volume and receive the file
        subprocess.run(["docker", "run", "--rm", "-d", "--name", "tmp_neo4j_loader", "--volumes-from", NEO4J_CONTAINER, "alpine", "sleep", "300"], capture_output=True)
        try:
            subprocess.run(["docker", "exec", "tmp_neo4j_loader", "mkdir", "-p", "/data/dumps"], check=True)
            subprocess.run(["docker", "cp", str(dump_path), "tmp_neo4j_loader:/data/dumps/neo4j.dump"], check=True)
            
            # 3. Restore using neo4j-admin load using the ephemeral container with Neo4j image
            log.info("  Standard restoration via neo4j-admin database load...")
            load_cmd = [
                "docker", "run", "--rm", "--volumes-from", NEO4J_CONTAINER, "neo4j:5.14-community",
                "neo4j-admin", "database", "load", NEO4J_DATABASE, "--from-path=/data/dumps", "--overwrite-destination=true"
            ]
            res = subprocess.run(load_cmd, capture_output=True, text=True)
            
            if res.returncode != 0:
                log.error(f"  [X] Neo4j Load failed: {res.stderr}")
                return False
        finally:
            log.info("  Cleaning up temporary loader container...")
            subprocess.run(["docker", "stop", "tmp_neo4j_loader"], capture_output=True)
            
        # 4. Start database again
        log.info(f"  Starting container {NEO4J_CONTAINER} again...")
        subprocess.run(["docker", "start", NEO4J_CONTAINER], check=True, capture_output=True)
        
        log.info("  [OK] Neo4j database restored and started.")
        return True
        
    except Exception as e:
        log.error(f"  [X] Neo4j restoration failed: {e}")
        subprocess.run(["docker", "start", NEO4J_CONTAINER], capture_output=True)
        return False

def import_redis(input_dir: Path):
    """Restore Redis from JSON export."""
    log.info("Restoring Redis keys from JSON...")
    json_path = input_dir / "db" / "redis_export.json"
    if not json_path.exists():
        json_path = input_dir / "db" / "redis" / "redis_export.json"
        
    if not json_path.exists():
        log.warning("  [!] No Redis JSON export found.")
        return False
        
    import redis
    client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=os.getenv("REDIS_PORT", 6380), db=int(os.getenv("REDIS_DB", 0)))
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for entry in data:
            k, t, v = entry["key"], entry["type"], entry["value"]
            if t == "string": client.set(k, v)
            elif t == "hash": client.hset(k, mapping=v)
            
        log.info(f"  [OK] {len(data)} Redis keys restored.")
        return True
    except Exception as e:
        log.error(f"  [X] Redis restoration failed: {e}")
        return False
    finally:
        client.close()

def main():
    parser = argparse.ArgumentParser(description="Import GraphRAG migration package")
    parser.add_argument("--input", required=True, help="Path to the extracted export folder")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    if not input_dir.exists():
        log.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
        
    print("\n" + "="*70)
    print("[START] GRAPHRAG NATIVE RESTORE")
    print("="*70)
    
    # 1. Qdrant
    import_qdrant_native(input_dir)
    
    # 2. Neo4j
    import_neo4j_native(input_dir)
    
    # 3. Redis
    import_redis(input_dir)
    
    print("\n" + "="*70)
    print("[OK] RESTORATION PROCESS FINISHED")
    print("Verify your databases using check_status.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
