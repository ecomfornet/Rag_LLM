import sys
from pathlib import Path
import logging

# Add project root to sys.path
project_root = Path(r"c:\Users\mouni\Documents\Adobe\Final\almalinux\graphrag_export\exports\graphrag_full_migration")
sys.path.append(str(project_root))

from agents_langgraph import AgenticPipeline
from retrieval_4 import HybridRetriever
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify():
    # Initialize pipeline (it creates its own retriever)
    pipeline = AgenticPipeline()
    
    # Test queries
    test_queries = [
        "Loi L/2019/0027/AN", # Specific doc title
        "What happened in 2020?", # Temporal signal
        "Explique-moi les droits des agents de l'état" # General query
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Testing Query: {query}")
        print(f"{'='*50}")
        
        try:
            result = pipeline.query(query, use_cache=False)
            
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            trace = result.get("agent_trace", [])
            
            print(f"Sources found: {len(sources)}")
            for i, src in enumerate(sources[:3], 1):
                title = src.get("metadata", {}).get("doc_title", "N/A")
                year = src.get("metadata", {}).get("year", "N/A")
                print(f"  {i}. {title} ({year})")
                
            if not sources:
                print("FAILED: No sources returned!")
            else:
                print("SUCCESS: Sources returned.")
                
            # Check if TemporalAnalystAgent was triggered h
            temporal_trace = next((t for t in trace if "TemporalAnalystAgent" in t["agent"]), None)
            if temporal_trace:
                print(f"Temporal Trace: {temporal_trace['status']}")
                
        except Exception as e:
            print(f"ERROR during query: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    verify()
