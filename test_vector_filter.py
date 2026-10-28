from retrieval_4 import HybridRetriever
from qdrant_client.models import Filter, FieldCondition, MatchValue
import json

retriever = HybridRetriever()

# Simulate what analyze_query extracts
query = "Arrêt N° 004 du 11 Janvier 2017"
filters = {"reference": "N° 004", "year": 2017}

print(f"\n--- Testing vector_search with filters: {filters} ---")
results = retriever.vector_search(query, top_k=5, filters=filters)

for res in results:
    title = res['metadata'].get('doc_title', 'No Title')
    ref = res['metadata'].get('reference', '')
    yr = res['metadata'].get('year', '')
    print(f"[{res['score']:.4f}] {title} (Ref: {ref}, Year: {yr})")
