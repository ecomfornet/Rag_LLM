from retrieval_4 import HybridRetriever
import json

retriever = HybridRetriever()
query = "Arrêt N° 004 du 11 Janvier 2017"

print(f"\n--- Full Retrieval for: '{query}' ---")
results = retriever.retrieve(query, top_k=5)

for res in results:
    title = res['metadata'].get('doc_title', 'No Title')
    ref = res['metadata'].get('reference', '')
    yr = res['metadata'].get('year', '')
    print(f"[{res['score']:.4f} | RRF: {res['rrf_score']:.4f}] {title} (Ref: {ref}, Year: {yr})")
