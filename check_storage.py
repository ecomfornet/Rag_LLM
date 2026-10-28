"""
Storage Diagnostic Script
Checks where and what data is stored in the GraphRAG system
"""
import logging
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
import config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_neo4j_storage():
    """Check Neo4j database storage and data"""
    print("\n" + "="*70)
    print("NEO4J DATABASE CHECK")
    print("="*70)
    
    try:
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        
        with driver.session(database=config.NEO4J_DATABASE) as session:
            # Get data directory path
            print("\n[*] Neo4j Storage Location:")
            try:
                result = session.run("""
                    CALL dbms.listConfig() 
                    YIELD name, value 
                    WHERE name = 'dbms.directories.data' 
                    RETURN value
                """)
                record = result.single()
                if record:
                    print(f"   Data Directory: {record['value']}")
                else:
                    print("   [!] Could not retrieve data directory path")
            except Exception as e:
                print(f"   [!] Could not query config (may need admin privileges): {e}")
            
            # Get database statistics
            print("\n[+] Database Statistics:")
            
            # Total documents
            result = session.run("MATCH (d:Document) RETURN count(d) as count")
            doc_count = result.single()['count']
            print(f"   Documents: {doc_count}")
            
            # Total chunks
            result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
            chunk_count = result.single()['count']
            print(f"   Chunks: {chunk_count}")
            
            # Total relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()['count']
            print(f"   Relationships: {rel_count}")
            
            # Relationship breakdown
            print("\n[+] Relationship Types:")
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
            """)
            for record in result:
                print(f"   {record['rel_type']}: {record['count']}")
            
            # Sample documents
            if doc_count > 0:
                print("\n[+] Sample Documents (first 5):")
                result = session.run("""
                    MATCH (d:Document)
                    RETURN d.id as id, d.title as title, d.type as type
                    LIMIT 5
                """)
                for i, record in enumerate(result, 1):
                    title = record['title'] if record['title'] else "No title"
                    print(f"   {i}. [{record['type']}] {title[:60]}...")
                    print(f"      ID: {record['id']}")
            
            # Database size estimate
            print("\n[+] Database Size:")
            print(f"   Total Nodes: {doc_count + chunk_count}")
            print(f"   Total Relationships: {rel_count}")
        
        driver.close()
        print("\n[OK] Neo4j connection successful")
        
    except Exception as e:
        print(f"\n[ERROR] Neo4j Error: {e}")
        print(f"   Connection: {config.NEO4J_URI}")
        print(f"   Database: {config.NEO4J_DATABASE}")

def check_qdrant_storage():
    """Check Qdrant vector database storage"""
    print("\n" + "="*70)
    print("QDRANT VECTOR DATABASE CHECK")
    print("="*70)
    
    try:
        client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            https=config.QDRANT_USE_HTTPS,
            api_key=config.QDRANT_API_KEY
        )
        
        print(f"\n[*] Qdrant Storage:")
        print(f"   Host: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
        print(f"   Collection: {config.QDRANT_COLLECTION}")
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        print(f"\n[+] Collections Found: {len(collection_names)}")
        for col_name in collection_names:
            print(f"   - {col_name}")
        
        # Get collection info if it exists
        if config.QDRANT_COLLECTION in collection_names:
            collection_info = client.get_collection(config.QDRANT_COLLECTION)
            print(f"\n[+] Collection '{config.QDRANT_COLLECTION}' Details:")
            print(f"   Vectors Count: {collection_info.points_count}")
            # Compatible with both old qdrant-client (vectors.size) and new (vectors as dict)
            vectors_cfg = collection_info.config.params.vectors
            if hasattr(vectors_cfg, 'size'):
                vec_size = vectors_cfg.size
                vec_dist = vectors_cfg.distance
            else:
                first = list(vectors_cfg.values())[0]
                vec_size, vec_dist = first.size, first.distance
            print(f"   Vector Dimension: {vec_size}")
            print(f"   Distance Metric: {vec_dist}")
            
            # Sample a few points
            if collection_info.points_count > 0:
                print(f"\n[+] Sample Vectors (first 3):")
                scroll_result = client.scroll(
                    collection_name=config.QDRANT_COLLECTION,
                    limit=3,
                    with_payload=True,
                    with_vectors=False
                )
                for i, point in enumerate(scroll_result[0], 1):
                    payload = point.payload
                    print(f"   {i}. ID: {point.id}")
                    if 'doc_id' in payload:
                        print(f"      Doc ID: {payload['doc_id']}")
                    if 'chunk_id' in payload:
                        print(f"      Chunk ID: {payload['chunk_id']}")
                    if 'content' in payload:
                        content_preview = payload['content'][:80]
                        print(f"      Content: {content_preview}...")
        else:
            print(f"\n[!] Collection '{config.QDRANT_COLLECTION}' not found!")
            print("   The collection may not have been created yet.")
        
        print("\n[OK] Qdrant connection successful")
        
    except Exception as e:
        print(f"\n[ERROR] Qdrant Error: {e}")
        print(f"   Host: {config.QDRANT_HOST}:{config.QDRANT_PORT}")


def check_local_files():
    """Check local file storage"""
    print("\n" + "="*70)
    print("LOCAL FILE STORAGE CHECK")
    print("="*70)
    
    print(f"\n[*] Project Directory: {config.BASE_DIR}")
    print(f"   Data Directory: {config.DATA_DIR}")
    print(f"   Logs Directory: {config.LOGS_DIR}")
    
    # Check data directory contents
    if config.DATA_DIR.exists():
        data_files = list(config.DATA_DIR.glob("*"))
        print(f"\n[+] Files in Data Directory: {len(data_files)}")
        for file in data_files[:10]:  # Show first 10
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   - {file.name} ({size_mb:.2f} MB)")
        if len(data_files) > 10:
            print(f"   ... and {len(data_files) - 10} more files")
    else:
        print("\n[!] Data directory does not exist yet")
    
    # Check logs
    if config.LOGS_DIR.exists():
        log_files = list(config.LOGS_DIR.glob("*.log"))
        print(f"\n[+] Log Files: {len(log_files)}")
        for log_file in log_files:
            size_kb = log_file.stat().st_size / 1024
            print(f"   - {log_file.name} ({size_kb:.2f} KB)")


def main():
    """Run all storage checks"""
    print("\n" + "="*70)
    print("GRAPHRAG STORAGE DIAGNOSTIC")
    print("="*70)
    
    check_neo4j_storage()
    check_qdrant_storage()
    check_local_files()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\n[*] Summary:")
    print("   - Neo4j stores graph data (documents, chunks, relationships)")
    print("   - Qdrant stores vector embeddings for semantic search")
    print("   - Local files store raw data and logs")
    print("\n")


if __name__ == "__main__":
    main()
