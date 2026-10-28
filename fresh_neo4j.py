"""
Fresh Neo4j Database Setup Script
This script manages Neo4j Docker container lifecycle:
- Stops and removes existing Neo4j containers
- Removes all Neo4j volumes (data cleanup)
- Starts a fresh Neo4j instance with configured credentials
"""

import subprocess
import time
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent))
from config import NEO4J_USER, NEO4J_PASSWORD, NEO4J_URI

def run_command(cmd, check=True, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result.stdout.strip() if capture_output else None
    except subprocess.CalledProcessError as e:
        if check:
            print(f"❌ Command failed: {cmd}")
            print(f"Error: {e.stderr}")
        return None

def stop_neo4j_containers():
    """Stop all running Neo4j containers."""
    print("[STOP] Stopping Neo4j containers...")

    # Always force-remove the container by its fixed name first,
    # in case it exists in any state (running, stopped, exited).
    container_name = "graphrag_neo4j"
    check = run_command(
        f'docker ps -a --filter "name=^{container_name}$" --format "{{{{.Names}}}}"',
        check=False
    )
    if check and container_name in check:
        print(f"   Force-removing existing container: {container_name}")
        run_command(f"docker rm -f {container_name}", check=False)
        print("[OK] Existing container removed")
    
    # Also sweep any other containers that were run from the neo4j image
    containers = run_command(
        'docker ps -a --filter "ancestor=neo4j" --format "{{.Names}}"',
        check=False
    )
    
    if containers:
        for container in containers.split('\n'):
            if container.strip():
                print(f"   Stopping: {container}")
                run_command(f"docker stop {container}", check=False)
                run_command(f"docker rm {container}", check=False)
        print("[OK] Containers stopped and removed")
    else:
        print("   No additional Neo4j containers found")

def remove_neo4j_volumes():
    """Remove all Neo4j Docker volumes."""
    print("[DELETE] Removing Neo4j volumes...")
    
    # Get all volumes with 'neo4j' in the name
    volumes = run_command(
        'docker volume ls --filter "name=neo4j" --format "{{.Name}}"',
        check=False
    )
    
    if volumes:
        volume_list = [v.strip() for v in volumes.split('\n') if v.strip()]
        for volume in volume_list:
            print(f"   Removing: {volume}")
            run_command(f"docker volume rm {volume}", check=False)
        print("[OK] Volumes removed")
    else:
        print("   No Neo4j volumes found")

def start_fresh_neo4j():
    """Start a fresh Neo4j container with configured credentials."""
    print("[START] Starting fresh Neo4j instance...")
    
    # Extract port from URI (default 7687)
    # URI format: bolt://host:port  — split carefully to avoid grabbing host part
    port = "7687"
    try:
        port = NEO4J_URI.split(":")[-1].split("/")[0] or port
    except (IndexError, ValueError):
        pass
    
    container_name = "graphrag_neo4j"
    
    # Docker run command
    cmd = f"""docker run -d \
        --name {container_name} \
        -p 7474:7474 \
        -p {port}:{port} \
        -e NEO4J_AUTH={NEO4J_USER}/{NEO4J_PASSWORD} \
        -v graphrag_neo4j_data:/data \
        -v graphrag_neo4j_logs:/logs \
        -v graphrag_neo4j_import:/import \
        -v graphrag_neo4j_plugins:/plugins \
        neo4j:5.14-community"""
    
    result = run_command(cmd.replace('\n', ' ').replace('\\', ''))
    
    if result:
        print(f"[OK] Neo4j container started: {container_name}")
        print(f"   Container ID: {result[:12]}")
        return True
    else:
        print("[ERROR] Failed to start Neo4j container")
        return False

def wait_for_neo4j(max_wait=60):
    """Wait for Neo4j to be ready."""
    print("[WAIT] Waiting for Neo4j to be ready...")
    
    from neo4j import GraphDatabase
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            print("[OK] Neo4j is ready!")
            return True
        except Exception:
            time.sleep(2)
            print("   Still waiting...")
    
    print("[ERROR] Neo4j did not become ready in time")
    return False

def main():
    """Main execution flow."""
    print("=" * 60)
    print("Fresh Neo4j Database Setup")
    print("=" * 60)
    print()
    
    print(f"Configuration:")
    print(f"  URI: {NEO4J_URI}")
    print(f"  User: {NEO4J_USER}")
    print(f"  Password: {'*' * len(NEO4J_PASSWORD)}")
    print()
    
    # Step 1: Stop existing containers
    stop_neo4j_containers()
    print()
    
    # Step 2: Remove volumes
    remove_neo4j_volumes()
    print()
    
    # Step 3: Start fresh instance
    if not start_fresh_neo4j():
        sys.exit(1)
    print()
    
    # Step 4: Wait for Neo4j to be ready
    if wait_for_neo4j():
        print()
        print("=" * 60)
        print("[SUCCESS] Fresh Neo4j database is ready")
        print("=" * 60)
        print()
        print("Access Neo4j Browser at: http://localhost:7474")
        print(f"Username: {NEO4J_USER}")
        print(f"Password: {'*' * len(NEO4J_PASSWORD)}  (see .env file)")
        print()
        print("You can now run your data_cleaning_1.py script!")
    else:
        print()
        print("=" * 60)
        print("[WARNING] Neo4j started but not yet ready")
        print("=" * 60)
        print("Please wait a few more seconds and try connecting")
        sys.exit(1)

if __name__ == "__main__":
    main()
