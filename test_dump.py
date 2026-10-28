import subprocess
import os

NEO4J_CONTAINER = "graphrag_neo4j"
NEO4J_DATABASE = "neo4j"
DUMP_OUTPUT = "test_dump.dump"

print("1. Stopping container...")
subprocess.run(["docker", "stop", NEO4J_CONTAINER])

try:
    print("2. Running dump container...")
    # Using the same Neo4j community image
    dump_cmd = [
        "docker", "run", "--rm", 
        "--volumes-from", NEO4J_CONTAINER, 
        "neo4j:5.14-community",
        "neo4j-admin", "database", "dump", NEO4J_DATABASE, 
        "--to-path=/data", "--overwrite-destination=true"
    ]
    res = subprocess.run(dump_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Dump failed! Error: {res.stderr}")
    else:
        print("Dump succeeded in container!")

    print("3. Copying dump from container...")
    cp_cmd = ["docker", "cp", f"{NEO4J_CONTAINER}:/data/{NEO4J_DATABASE}.dump", DUMP_OUTPUT]
    subprocess.run(cp_cmd, check=True)
    print("Dump Copied Out!")
finally:
    print("4. Restarting container...")
    subprocess.run(["docker", "start", NEO4J_CONTAINER])
