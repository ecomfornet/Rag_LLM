import sys
from pathlib import Path
from qdrant_client import QdrantClient
import json

# Add project root to sys.path
project_root = Path(r"c:\Users\mouni\Documents\Adobe\Final\almalinux\graphrag_export\exports\graphrag_full_migration")
sys.path.append(str(project_root))

import config

def dump_payloads():
    client = QdrantClient(
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT,
        https=config.QDRANT_USE_HTTPS,
        api_key=config.QDRANT_API_KEY
    )
    
    print(f"Connecting to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}...")
    
    # Scroll some points
    scroll_result = client.scroll(
        collection_name=config.QDRANT_COLLECTION,
        limit=5,
        with_payload=True
    )
    
    points = scroll_result[0]
    print(f"Found {len(points)} points.")
    
    for i, point in enumerate(points, 1):
        print(f"\n--- Point {i} (ID: {point.id}) ---")
        print(json.dumps(point.payload, indent=2))

if __name__ == "__main__":
    dump_payloads()
