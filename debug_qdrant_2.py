
from qdrant_client import QdrantClient
import config

try:
    client = QdrantClient(host="localhost", port=config.QDRANT_PORT)
    attrs = dir(client)
    # Print only non-magic methods
    methods = [a for a in attrs if not a.startswith('_')]
    print(f"Available methods: {methods}")
    
except Exception as e:
    print(f"Error: {e}")
