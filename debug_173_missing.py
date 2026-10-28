import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from neo4j import GraphDatabase
import json

# Add project root to sys.path
project_root = Path(r"c:\Users\mouni\Documents\Adobe\Final\almalinux\graphrag_export\exports\graphrag_full_migration")
sys.path.append(str(project_root))

import config

def debug_specific_doc():
    # 1. Search in Qdrant
    q_client = QdrantClient(
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT,
        https=config.QDRANT_USE_HTTPS,
        api_key=config.QDRANT_API_KEY
    )
    
    print("--- Searching Qdrant for '173/2021' ---")
    results = q_client.scroll(
        collection_name=config.QDRANT_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.reference",
                    match=models.MatchValue(value="N° 173")
                )
            ]
        ),
        limit=5
    )
    
    if results[0]:
        print(f"Found {len(results[0])} matches in Qdrant.")
        for p in results[0]:
            print(json.dumps(p.payload, indent=2))
    else:
        print("No exact reference match for 'N° 173' in Qdrant.")
        # Try metadata title search
        print("\n--- Searching Qdrant for '173' in title ---")
        results_title = q_client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.doc_title",
                        match=models.MatchText(text="173")
                    )
                ]
            ),
            limit=5
        )
        for p in results_title[0]:
            print(json.dumps(p.payload, indent=2))

    # 2. Search in Neo4j
    print("\n--- Searching Neo4j for '173/2021' ---")
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run("MATCH (d:Document) WHERE d.reference CONTAINS '173' OR d.title CONTAINS '173' RETURN d.title, d.reference, d.doc_id LIMIT 5")
        records = list(result)
        print(f"Found {len(records)} matches in Neo4j.")
        for r in records:
            print(f"Title: {r['d.title']}, Ref: {r['d.reference']}, ID: {r['d.doc_id']}")
    driver.close()

if __name__ == "__main__":
    debug_specific_doc()
