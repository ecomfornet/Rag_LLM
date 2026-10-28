import logging
import asyncio
from retrieval_4 import HybridRetriever

logging.basicConfig(level=logging.INFO)

async def debug_convention():
    retriever = HybridRetriever()
    query = "Quelle est la classification des travailleurs dans la Convention Collective des Mines, Carrières et Industries Assimilées"
    
    print("--- 1. Query Analysis ---")
    analysis = retriever.analyze_query(query)
    print("Analysis:", analysis)
    
    print("\n--- 2. Vector Search ---")
    vector_results = retriever.vector_search(
        query, 
        top_k=20, 
        filters=analysis.get('metadata_filters')
    )
    for i, res in enumerate(vector_results[:10]):
        # Extract title supporting both flat and nested metadata
        title = res.get('doc_title') or res.get('title') or res.get('metadata', {}).get('doc_title', 'N/A')
        url = res.get('source_url') or res.get('metadata', {}).get('source_url', 'N/A')
        print(f"[{i+1}] Score: {res.get('score', 0):.4f} | Title: {title[:60]} | URL: {url}")

    print("\n--- 3. Full Retrieve (Top 10 before Reranking) ---")
    # We'll just run the normal retrieve and inspect the final chunks
    chunks = retriever.retrieve(query)
    for i, chunk in enumerate(chunks):
        title = chunk.get('doc_title') or chunk.get('title') or chunk.get('metadata', {}).get('doc_title', 'N/A')
        url = chunk.get('source_url') or chunk.get('metadata', {}).get('source_url', 'N/A')
        print(f"Final [{i+1}] Score: {chunk.get('score', chunk.get('rrf_score', 0)):.4f} | Title: {title[:60]}... | URL: {url}")
        
if __name__ == "__main__":
    asyncio.run(debug_convention())
