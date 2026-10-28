import subprocess
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("test-neo4j-export")

NEO4J_CONTAINER = "graphrag_neo4j"
NEO4J_DATABASE = "neo4j"

def run_cmd(cmd):
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        log.info("Success!")
        if result.stdout:
            print(result.stdout)
    else:
        log.error(f"Failed with return code {result.returncode}")
        if result.stderr:
            log.error(f"Error: {result.stderr}")
    return result

def main():
    log.info("--- Neo4j Export Diagnostic Test ---")
    
    # 1. Check if docker is available
    log.info("Checking docker availability...")
    run_cmd(["docker", "--version"])
    
    # 2. Check if container is running
    log.info(f"Checking if container {NEO4J_CONTAINER} is running...")
    run_cmd(["docker", "ps", "--filter", f"name={NEO4J_CONTAINER}"])
    
    # 3. Try to run neo4j-admin inside the container
    log.info("Checking neo4j-admin version inside container...")
    run_cmd(["docker", "exec", NEO4J_CONTAINER, "neo4j-admin", "--version"])
    
    # 4. Try to create /dumps directory
    log.info("Attempting to create /dumps directory inside container...")
    run_cmd(["docker", "exec", NEO4J_CONTAINER, "mkdir", "-p", "/dumps"])
    
    # 5. Check database status (to see if it's online)
    log.info("Checking database status...")
    run_cmd(["docker", "exec", NEO4J_CONTAINER, "cypher-shell", "-u", "neo4j", "-p", "graphrag123", "SHOW DATABASES;"])

    # 6. Try the dump command that export_project.py uses
    log.info("Attempting the dump command (CAUTION: might fail if DB is online)...")
    dump_cmd = [
        "docker", "exec", NEO4J_CONTAINER, 
        "neo4j-admin", "database", "dump", NEO4J_DATABASE, "--to-path=/dumps", "--overwrite-destination=true"
    ]
    run_cmd(dump_cmd)

if __name__ == "__main__":
    main()
