import sys
from pathlib import Path
import json
import logging

# Add project root to sys.path
project_root = Path(r"c:\Users\mouni\Documents\Adobe\Final\almalinux\graphrag_export\exports\graphrag_full_migration")
sys.path.append(str(project_root))

import config
from retrieval_4 import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_filtering():
    retriever = HybridRetriever()
    
    # Simulate a query that might cause issues
    query = "la jurisprudence de la CCJA sur l'arbitrage en 2021"
    print(f"\nAnalyzing query: {query}")
    
    plan = retriever.analyze_query(query)
    print(f"Plan generated: {json.dumps(plan, indent=2)}")
    
    metadata_filters = plan.get("metadata_filters", {})
    print(f"Metadata filters: {metadata_filters}")
    
    # Run a vector search to get raw results
    print("\nRunning vector search...")
    raw_results = retriever.vector_search(query, top_k=20)
    print(f"Raw results found: {len(raw_results)}")
    
    # Replicate the _is_title_match logic
    import re
    def _is_title_match(query_title, doc_title):
        stopwords = {"les", "des", "sur", "pour", "dans", "aux", "par", "une", "est", "qui", "que", "avec"}
        def get_words(text):
            text = re.sub(r'[^\w\s\']', ' ', text.lower())
            return [w for w in text.split() if len(w) > 2 and w not in stopwords]
        
        q_words = get_words(query_title)
        if not q_words:
            return True
            
        d_words = set(get_words(doc_title))
        
        if len(q_words) > 0 and q_words[0] not in d_words:
            return False
        if len(q_words) > 1 and q_words[1] not in d_words:
            return False
            
        matches = sum(1 for w in q_words if w in d_words)
        return (matches / len(q_words)) >= 0.4

    # Replicate the filtering loop
    filtered_results = []
    reasons = []
    
    for res in raw_results:
        match = True
        fail_reason = None
        
        if metadata_filters:
            for k, v in metadata_filters.items():
                if not v or v == {} or v == []:
                    continue
                    
                val_flat = res.get(k, '')
                val_nested = res.get('metadata', {}).get(k, '')
                val = str(val_nested or val_flat).strip()
                
                if k == 'doc_title':
                    q_vals = v if isinstance(v, list) else [v]
                    matched_any = False
                    for qv in q_vals:
                        if qv and _is_title_match(str(qv).strip(), val):
                            matched_any = True
                            break
                    if not matched_any:
                        match = False
                        fail_reason = f"doc_title mismatch: '{val}' does not match any of {q_vals}"
                        break
                else:
                    query_val = str(v).strip()
                    if val.lower() != query_val.lower():
                        match = False
                        fail_reason = f"field '{k}' mismatch: '{val.lower()}' != '{query_val.lower()}'"
                        break
        
        if match:
            filtered_results.append(res)
        else:
            reasons.append(fail_reason)
            
    print(f"\nFiltered results: {len(filtered_results)} / {len(raw_results)}")
    if reasons:
        print("\nFilter failure reasons (unique):")
        for r in set(reasons):
            print(f"  - {r}")
            
    # Sample a successful result if any
    if filtered_results:
        print("\nSample success:")
        print(f"  Title: {filtered_results[0].get('metadata', {}).get('doc_title')}")
        print(f"  Year: {filtered_results[0].get('metadata', {}).get('year')}")
    else:
        print("\nNo results survived the filter!")
        if raw_results:
            print("\nSample raw result qualities:")
            for i, r in enumerate(raw_results[:3]):
                m = r.get('metadata', {})
                print(f"  {i+1}. Title: {m.get('doc_title')}")
                print(f"     Year: {m.get('year')}, Reference: {m.get('reference')}")

if __name__ == "__main__":
    test_filtering()
