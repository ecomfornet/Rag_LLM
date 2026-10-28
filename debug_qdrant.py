
from qdrant_client import QdrantClient
import config

try:
    client = QdrantClient(host="localhost", port=config.QDRANT_PORT)
    attrs = dir(client)
    print("Client attributes count:", len(attrs))
    
    if 'search' in attrs:
        print("✓ Search method exists!")
    else:
        print("✗ Search method MISSING!")
        # Print what methods might be available
        print("Possible alternatives:", [a for a in attrs if 'search' in a or 'query' in a])
        
except Exception as e:
    print(f"Error: {e}")
