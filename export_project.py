"""
Complete Project Export Script for GraphRAG Legal Documents
Exports all project files, data directory, and native DB dumps (Qdrant snapshots, Neo4j dumps)
"""
import os
import sys
import shutil
import subprocess
import json
import time
import logging
import requests
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("project-export")

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

REDIS_HOST     = os.getenv("REDIS_HOST",     "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", "6380"))
REDIS_DB       = int(os.getenv("REDIS_DB",   "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

def _get_qdrant_url():
    scheme = "https" if QDRANT_USE_HTTPS else "http"
    return f"{scheme}://{QDRANT_HOST}:{QDRANT_PORT}"

def _get_qdrant_headers():
    h = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        h["api-key"] = QDRANT_API_KEY
    return h

# --- Export Functions ---

def export_qdrant_native(output_dir: Path):
    """Trigger Qdrant snapshot and download it."""
    log.info("Creating native Qdrant snapshot...")
    db_dir = output_dir / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = _get_qdrant_url()
    headers = _get_qdrant_headers()
    
    # Trigger snapshot
    resp = requests.post(f"{base_url}/collections/{QDRANT_COLLECTION}/snapshots", headers=headers, timeout=120)
    resp.raise_for_status()
    snap_info = resp.json()["result"]
    snap_name = snap_info["name"]
    log.info(f"  [OK] Snapshot created on server: {snap_name}")
    
    # Download snapshot
    log.info(f"  Downloading snapshot {snap_name}...")
    dl_url = f"{base_url}/collections/{QDRANT_COLLECTION}/snapshots/{snap_name}"
    dl_resp = requests.get(dl_url, headers=headers, timeout=600, stream=True)
    dl_resp.raise_for_status()
    
    snap_path = db_dir / f"{QDRANT_COLLECTION}.snapshot"
    with open(snap_path, "wb") as f:
        for chunk in dl_resp.iter_content(chunk_size=8192):
            f.write(chunk)
            
    log.info(f"  [OK] Native Qdrant snapshot saved: {snap_path}")
    return str(snap_path)

def export_neo4j_native(output_dir: Path):
    """Create native Neo4j dump using neo4j-admin via offline container (Community Edition compatible)."""
    log.info("Creating native Neo4j database dump...")
    db_dir = output_dir / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    
    dump_path = db_dir / "neo4j_database.dump"
    
    try:
        # 1. Stop the container
        log.info(f"  Stopping container {NEO4J_CONTAINER} for offline dump...")
        subprocess.run(["docker", "stop", NEO4J_CONTAINER], check=True, capture_output=True)
        
        # 2. Run ephemeral dump container
        log.info(f"  Executing neo4j-admin dump via ephemeral container...")
        dump_cmd = [
            "docker", "run", "--rm", 
            "--volumes-from", NEO4J_CONTAINER, 
            "neo4j:5.14-community",
            "neo4j-admin", "database", "dump", NEO4J_DATABASE, 
            "--to-path=/data", "--overwrite-destination=true"
        ]
        res = subprocess.run(dump_cmd, capture_output=True, text=True)
        
        # 3. Start the original container back up
        log.info(f"  Restarting container {NEO4J_CONTAINER}...")
        subprocess.run(["docker", "start", NEO4J_CONTAINER], check=True, capture_output=True)

        if res.returncode != 0:
            log.error(f"  [X] Neo4j admin dump failed: {res.stderr}")
            return None
        
        # 4. Copy from container to host
        log.info("  Copying dump file from container...")
        subprocess.run(["docker", "cp", f"{NEO4J_CONTAINER}:/data/{NEO4J_DATABASE}.dump", str(dump_path)], check=True, capture_output=True)
        
        log.info(f"  [OK] Native Neo4j dump saved: {dump_path}")
        return str(dump_path)
        
    except Exception as e:
        log.error(f"  [X] Neo4j native export failed: {e}")
        # Make sure container is restarted even if something fails
        subprocess.run(["docker", "start", NEO4J_CONTAINER], capture_output=True)
        return None

def export_redis(output_dir: Path):
    """Export Redis keys to JSON (simplest for small/medium DBs)."""
    log.info("Exporting Redis database...")
    db_dir = output_dir / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    
    import redis
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASSWORD)
    
    try:
        keys = client.keys("*")
        data = []
        for k in keys:
            t = client.type(k).decode("utf-8")
            val = None
            try:
                if t == "string": val = client.get(k).decode("utf-8")
                elif t == "hash": val = {fk.decode(): fv.decode() for fk, fv in client.hgetall(k).items()}
                # Add other types if necessary
            except:
                pass # skip binary or incompatible
                
            data.append({"key": k.decode("utf-8"), "type": t, "value": val})
            
        with open(db_dir / "redis_export.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        log.info(f"  [OK] Redis: {len(data)} keys exported to JSON.")
        return len(data)
    finally:
        client.close()

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    export_name = f"graphrag_full_migration_{timestamp}"
    export_dir = Path("exports") / export_name
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GRAPHRAG FULL PROJECT EXPORT (NATIVE DUMPS)")
    print("="*70)
    
    stats = {}
    
    # 1. Project Files (Source Code)
    log.info("Step 1: Copying project files...")
    root_files = [f for f in os.listdir(".") if os.path.isfile(f)]
    for f in root_files:
        if f.endswith(".py") or f in ["requirements.txt", "docker-compose.yml", ".env.example", ".gitignore", "README.md", "QUICKSTART.md", "start.md", "setup.md", "EXPORT_IMPORT_GUIDE.md"]:
            shutil.copy2(f, export_dir / f)
            print(f"  [OK] {f}")
            
    # 2. Data Directory (Cleaned files, chunks, etc.)
    log.info("Step 2: Copying data directory...")
    data_src = Path("data")
    if data_src.exists():
        shutil.copytree(data_src, export_dir / "data", dirs_exist_ok=True)
        print("  [OK] data/ directory copied")
        
    # 3. Database Exports (Native)
    log.info("Step 3: Creating native database snapshots/dumps...")
    
    # Qdrant Native Snapshot
    try:
        stats["qdrant_snapshot"] = export_qdrant_native(export_dir)
    except Exception as e:
        log.error(f"  [X] Qdrant Snapshot failed: {e}")
        
    # Neo4j Native Dump
    try:
        stats["neo4j_dump"] = export_neo4j_native(export_dir)
    except Exception as e:
        log.error(f"  [X] Neo4j Dump failed: {e}")
        
    # Redis Export
    try:
        stats["redis_keys"] = export_redis(export_dir)
    except Exception as e:
        log.error(f"  [X] Redis export failed: {e}")
        
    # 4. Create Manifest
    log.info("Step 4: Creating export manifest...")
    manifest = {
        "timestamp": timestamp,
        "project": "GraphRAG Legal Documents",
        "export_mode": "native_snapshots",
        "stats": stats,
        "environment": {
            "qdrant_collection": QDRANT_COLLECTION,
            "neo4j_db": NEO4J_DATABASE
        }
    }
    with open(export_dir / "EXPORT_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    # 5. Installation/Restore Guide
    log.info("Step 5: Creating installation guide...")
    guide = f"""# GraphRAG Project Migration Guide

This package contains a full snapshot of the GraphRAG project, including source code, processed data, and native database dumps.

## Restoration Steps

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   # Update .env with your keys
   ```

2. **Start Infrastructure**:
   ```bash
   docker-compose up -d
   ```

3. **Restore Databases**:
   The easiest way is to use the updated `import_databases.py`:
   ```bash
   python import_databases.py --input ./
   ```
   
   This script will:
   - Upload the `{QDRANT_COLLECTION}.snapshot` to your new Qdrant instance.
   - Use `neo4j-admin database load` to restore the Neo4j dump.

4. **Verify & Run**:
   - Check status: `python check_status.py`
   - Start UI: `streamlit run app.py`
"""
    with open(export_dir / "INSTALLATION.md", "w", encoding="utf-8") as f:
        f.write(guide)
        
    # 6. Final Zip Archive
    log.info("Step 6: Creating final ZIP archive...")
    zip_path = Path("exports") / f"{export_name}.zip"
    shutil.make_archive(str(export_dir.parent / export_name), 'zip', export_dir)
    
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print(f"Package: {zip_path.resolve()}")
    print(f"Dumps: db/ subdirectory")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
