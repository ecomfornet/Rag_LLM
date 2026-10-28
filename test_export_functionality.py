import os
import sys
import shutil
import json
import logging
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test-export")

def run_test():
    log.info("Starting Export Functionality Test...")
    
    # 1. Clean previous exports
    export_base = Path("exports")
    if export_base.exists():
        log.info("Cleaning up previous exports (if possible)...")
        try:
            shutil.rmtree(export_base)
        except Exception as e:
            log.warning(f"  ⚠ Could not fully clean exports/ directory: {e}")
    
    # 2. Run the export script
    log.info("Executing export_project.py...")
    result = subprocess.run([sys.executable, "export_project.py"], capture_output=True, text=True)
    
    if result.returncode != 0:
        log.error("Export script failed!")
        print(result.stdout)
        print(result.stderr)
        return False
        
    print(result.stdout)
    
    # 3. Verify output
    export_dirs = list(export_base.glob("graphrag_full_migration_*"))
    if not export_dirs:
        log.error("No export directory created!")
        return False
        
    export_dir = export_dirs[0]
    log.info(f"Verifying export in: {export_dir}")
    
    expected_files = [
        "EXPORT_MANIFEST.json",
        "INSTALLATION.md",
        "requirements.txt",
        "docker-compose.yml",
        "db/legal_docs_graphrag.snapshot",
        "db/neo4j_database.dump",
        "db/redis_export.json"
    ]
    
    missing = []
    for f in expected_files:
        p = export_dir / f
        if not p.exists():
            missing.append(f)
        else:
            size = p.stat().st_size
            log.info(f"  ✓ {f} exists ({size} bytes)")
            if size == 0:
                log.warning(f"  ⚠ {f} is empty!")
                
    if missing:
        log.error(f"Missing files: {missing}")
        return False
        
    # Check Manifest
    with open(export_dir / "EXPORT_MANIFEST.json", "r") as f:
        manifest = json.load(f)
        log.info("Manifest content verified.")
        
    # Check ZIP
    zip_files = list(export_base.glob("graphrag_full_migration_*.zip"))
    if not zip_files:
        log.error("Final ZIP archive not found!")
        return False
    log.info(f"  ✓ ZIP archive created: {zip_files[0].name}")
    
    log.info("=== TEST SUCCESSFUL ===")
    return True

if __name__ == "__main__":
    if run_test():
        sys.exit(0)
    else:
        sys.exit(1)
