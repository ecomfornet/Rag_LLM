import sys
from pathlib import Path
from neo4j import GraphDatabase
import json
from datetime import datetime

# Add project root to sys.path
project_root = Path(r"c:\Users\mouni\Documents\Adobe\Final\almalinux\graphrag_export\exports\graphrag_full_migration")
sys.path.append(str(project_root))

import config

def serialize_neo4j(obj):
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return str(obj)

def inspect_properties():
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    with driver.session() as session:
        print("--- Inspecting Document Node Properties ---")
        doc_res = session.run("""
            MATCH (d:Document) 
            WHERE d.title CONTAINS '173/2021' 
            RETURN properties(d) as props LIMIT 1
        """)
        doc = doc_res.single()
        if doc:
            props = {k: serialize_neo4j(v) for k, v in doc['props'].items()}
            print(f"Document Properties: {json.dumps(props, indent=2)}")
            
            print("\n--- Inspecting Chunk Node Properties ---")
            chunk_res = session.run("""
                MATCH (d:Document)<-[:BELONGS_TO]-(c:Chunk)
                WHERE d.title CONTAINS '173/2021'
                RETURN properties(c) as props LIMIT 1
            """)
            chunk = chunk_res.single()
            if chunk:
                c_props = {k: serialize_neo4j(v) for k, v in chunk['props'].items()}
                print(f"Chunk Properties: {json.dumps(c_props, indent=2)}")
            else:
                print("No chunks found for this document.")
        else:
            print("Document not found.")
    driver.close()

if __name__ == "__main__":
    inspect_properties()
