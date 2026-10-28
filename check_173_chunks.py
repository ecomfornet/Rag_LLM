import sys
from pathlib import Path
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json

# Add project root to sys.path
project_root = Path(r"c:\Users\mouni\Documents\Adobe\Final\almalinux\graphrag_export\exports\graphrag_full_migration")
sys.path.append(str(project_root))

import config

def check_chunks():
    # 1. Search Neo4j for doc and its chunks
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    with driver.session() as session:
        print("--- Checking Neo4j for Document and Chunks ---")
        result = session.run("""
            MATCH (d:Document) 
            WHERE d.title CONTAINS '173/2021' 
            OPTIONAL MATCH (d)<-[:BELONGS_TO]-(c:Chunk)
            RETURN d.title, d.doc_id, count(c) as chunk_count, collect(c.chunk_id)[..3] as sample_chunks
        """)
        record = result.single()
        if record:
            print(f"Document: {record['d.title']}")
            print(f"Doc ID: {record['d.doc_id']}")
            print(f"Chunk Count: {record['chunk_count']}")
            print(f"Sample Chunks: {record['sample_chunks']}")
            
            doc_id = record['d.doc_id']
            
            # 2. Check Qdrant for these chunks
            if doc_id:
                print(f"\n--- Checking Qdrant for Doc ID: {doc_id} ---")
                q_client = QdrantClient(
                    host=config.QDRANT_HOST,
                    port=config.QDRANT_PORT,
                    https=config.QDRANT_USE_HTTPS,
                    api_key=config.QDRANT_API_KEY
                )
                q_res = q_client.scroll(
                    collection_name=config.QDRANT_COLLECTION,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=doc_id)
                            )
                        ]
                    ),
                    limit=5
                )
                if q_res[0]:
                    print(f"Found {len(q_res[0])} chunks in Qdrant for doc_id {doc_id}")
                else:
                    print(f"No chunks found in Qdrant for doc_id {doc_id}")
            else:
                print("No Doc ID found, cannot check Qdrant easily.")
        else:
            print("Document '173/2021' not found in Neo4j.")
    driver.close()

if __name__ == "__main__":
    check_chunks()
