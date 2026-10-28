from retrieval_4 import HybridRetriever
import json

retriever = HybridRetriever()
query = "Arrêt N° 004 du 11 Janvier 2017"

print("--- BM25 Search ---")
bm25_results = retriever.bm25_search(query, top_k=5)
for res in bm25_results:
    print(f"[{res['score']:.4f}] {res['metadata'].get('doc_title', 'No Title')}")

print("\n--- Vector Search ---")
vector_results = retriever.vector_search(query, top_k=5)
for res in vector_results:
    print(f"[{res['score']:.4f}] {res['metadata'].get('doc_title', 'No Title')}")
