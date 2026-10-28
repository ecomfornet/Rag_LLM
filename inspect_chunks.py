import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import config

load_dotenv()

client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, api_key=config.QDRANT_API_KEY)

url = "https://kytisa.com/cour-dappel-de-kankan-chambre-du-premier-president-arret-n-004-du-11-janvier-2017/"

results = client.scroll(
    collection_name=config.QDRANT_COLLECTION,
    scroll_filter=Filter(
        should=[
            FieldCondition(key="metadata.source_url", match=MatchValue(value=url)),
            FieldCondition(key="source_url", match=MatchValue(value=url))
        ]
    ),
    limit=2,
    with_payload=True
)

for p in results[0]:
    print(f"--- Chunk ID: {p.id} ---")
    payload = p.payload
    print(f"Content length: {len(payload.get('content', ''))}")
    print(f"Content preview: {payload.get('content', '')[:100]}...")
    metadata = payload.get('metadata', {})
    print(f"Title: {metadata.get('doc_title') or payload.get('doc_title')}")
    print(f"URL: {metadata.get('source_url') or payload.get('source_url')}")
    print(json.dumps(metadata, indent=2))
