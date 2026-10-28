import time
import logging
from typing import List, Dict
import statistics
from main import GraphRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEST_QUERIES = [
    "Quelles sont les obligations d'une SARL ?",
    "Comment créer une société anonyme ?",
    "Quel est le taux de l'impôt sur les sociétés en Guinée ?",
    "Quelles sont les sanctions pénales pour abus de biens sociaux ?",
    "Quelle est la durée du mandat des administrateurs ?"
]

# Warmup query to pre-load connections and avoid biasing the first result
WARMUP_QUERY = "test connexion"

def run_benchmark():
    logger.info("Initializing GraphRAG Pipeline for Benchmarking...")
    pipeline = GraphRAGPipeline()
    
    try:
        # ── Warmup: pre-load connections so the first test query is not biased ──
        logger.info("Running warmup query (not timed)...")
        try:
            pipeline.query(WARMUP_QUERY)
        except Exception:
            pass  # Warmup failure is non-fatal
        
        results = []
        total_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"STARTING BENCHMARK WITH {len(TEST_QUERIES)} QUERIES")
        print(f"{'='*60}\n")
        
        for i, query in enumerate(TEST_QUERIES):
            logger.info(f"Running Query {i+1}/{len(TEST_QUERIES)}: {query}")
            print(f"Query {i+1}: {query}")
            
            start_time = time.time()
            try:
                response = pipeline.query(query)
                end_time = time.time()
                latency = end_time - start_time
                
                response_length = len(response.get('answer', ''))
                # Fixed: use 'sources' (not 'context' which never existed in the response dict)
                source_count = len(response.get('sources', []))
                has_citations = source_count > 0
                
                result = {
                    'query': query,
                    'latency': latency,
                    'status': 'SUCCESS',
                    'response_length': response_length,
                    'citations': has_citations,
                    'source_count': source_count
                }
                print(f"  -> Time: {latency:.2f}s | Length: {response_length} chars | Sources: {source_count}")
                
            except Exception as e:
                end_time = time.time()
                latency = end_time - start_time
                logger.error(f"Query failed: {e}")
                result = {
                    'query': query,
                    'latency': latency,
                    'status': 'FAILED',
                    'error': str(e)
                }
                print(f"  -> FAILED: {str(e)}")
            
            results.append(result)
            print("-" * 40)
        
        total_duration = time.time() - total_start_time
        
        # Calculate metrics
        successful_latencies = [r['latency'] for r in results if r['status'] == 'SUCCESS']
        avg_latency    = statistics.mean(successful_latencies)   if successful_latencies else 0
        median_latency = statistics.median(successful_latencies) if successful_latencies else 0
        p95_latency    = sorted(successful_latencies)[int(len(successful_latencies) * 0.95)] \
                         if len(successful_latencies) >= 2 else (successful_latencies[0] if successful_latencies else 0)
        success_rate   = len(successful_latencies) / len(TEST_QUERIES) * 100
        
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Total Duration:    {total_duration:.2f}s")
        print(f"Average Latency:   {avg_latency:.2f}s")
        print(f"Median Latency:    {median_latency:.2f}s")
        print(f"P95 Latency:       {p95_latency:.2f}s")
        print(f"Success Rate:      {success_rate:.0f}%")
        print("\nDetailed Breakdown:")
        for r in results:
            status_text = "[OK]" if r['status'] == 'SUCCESS' else "[FAIL]"
            print(f"{status_text} {r['latency']:.2f}s - {r['query']}")
        print(f"{'='*60}\n")
    
    finally:
        # Always close Neo4j and Qdrant connections, even if a query failed
        pipeline.close()
        logger.info("Pipeline closed.")

if __name__ == "__main__":
    run_benchmark()
