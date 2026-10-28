import sys
import logging
import time

sys.stdout.reconfigure(encoding='utf-8')

import config
from retrieval_4 import HybridRetriever

logging.basicConfig(level=logging.INFO)

query = "Quel est le taux d'imposition sur les plus-values de cession d'actions selon le CGI 2024"

print(f"Testing retrieval for query: {query}")

# Overriding config values to maximize retrieval coverage
config.DENSE_TOP_K = 20
config.FINAL_TOP_K = 15
config.BM25_TOP_K = 15
config.RERANKING_ENABLED = False # BYPASS LLM RERANKING to see raw RRF output

retriever = HybridRetriever()

try:
    chunks, context = retriever.retrieve_with_context(query, top_k=15)
    
    print(f"\nRetrieved {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        m = chunk['metadata']
        score = chunk.get('rrf_score', chunk.get('score', 0))
        title = m.get('doc_title', 'N/A')
        print(f"{i}. [{chunk['source']}] (Score: {score:.4f}) {title}")
        print(f"   Reference: {m.get('reference', 'N/A')}, Year: {m.get('year', 'N/A')}")
        print(f"   Doc ID: {chunk['doc_id']}\n")
        
    print("\n\nTesting exact matches in Qdrant:")
    # Direct Qdrant search
    raw_results = retriever.qdrant.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=retriever.embed_query(query),
        limit=20
    ).points
    
    print(f"Direct Qdrant Top 20 results:")
    for i, r in enumerate(raw_results, 1):
        m = r.payload['metadata']
        title = m.get('doc_title', 'N/A')
        print(f"{i}. (Score: {r.score:.4f}) {title}")
        
except Exception as e:
    print(f"Error: {e}")
finally:
    retriever.close()
