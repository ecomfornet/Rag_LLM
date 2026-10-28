from retrieval_4 import HybridRetriever
import json
import logging

# Enable logger for debug
logging.basicConfig(level=logging.DEBUG)

retriever = HybridRetriever()

query = "Arrêt N° 004 du 11 Janvier 2017"

print(f"\n--- Testing retrieve pipeline ---")
results = retriever.retrieve(query, top_k=5)

for i, res in enumerate(results):
    title = res['metadata'].get('doc_title', 'No Title')
    ref = res['metadata'].get('reference', '')
    yr = res['metadata'].get('year', '')
    print(f"{i+1}. [{res.get('score', 0):.4f} | RRF: {res.get('rrf_score', 0):.4f}] {title} (Ref: {ref}, Year: {yr})")
