"""Debug: trace what VectorSpecialistAgent puts into vector_results."""
import sys, logging, time
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

import config
from retrieval_4 import HybridRetriever

r = HybridRetriever()

query = "Quelles etaient les dispositions du Code Minier anterieur a 2011 avec la loi en Guinee"

# Simulate VectorSpecialistAgent exactly
# 1. Query expansion
from openai import OpenAI
client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=30.0)
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"Generate 2 alternative search queries for: {query}\nReturn only the queries, one per line."}],
    temperature=0.7, max_tokens=150,
)
lines = [l.strip() for l in resp.choices[0].message.content.strip().splitlines() if l.strip()]
queries = [query] + lines[:2]
print(f"Queries: {queries}")

# 2. Dense search for each query
result_lists = []
for q in queries:
    dense = r.vector_search(q, top_k=config.DENSE_TOP_K)
    result_lists.append(dense)

# 3. BM25
if config.BM25_ENABLED:
    bm25 = r.bm25_search(query, top_k=config.BM25_TOP_K)
    if bm25:
        result_lists.append(bm25)

# 4. Graph doc search (simulating agent)
graph_doc_ids = r.graph_expansion(list({d['doc_id'] for lst in result_lists for d in lst[:4]}))
seed_doc_ids = {d['doc_id'] for lst in result_lists for d in lst}
graph_only = [d for d in graph_doc_ids if d not in seed_doc_ids]
print(f"\nGraph expansion: {len(graph_doc_ids)} total, {len(graph_only)} graph-only")

if graph_only and config.GRAPH_EXPANSION_ENABLED:
    graph_res = r.search_expanded_documents(query, graph_only, top_k=8)
    print(f"Graph search results: {len(graph_res)}")
    for g in graph_res:
        print(f"  chunk_id={g['chunk_id']}  doc_id={g['doc_id']}  source={g['source']}")
    if graph_res:
        result_lists.append(graph_res)

# Count by source
all_results = [r2 for lst in result_lists for r2 in lst]
src_counts = {}
for r2 in all_results:
    s = r2.get('source', '?')
    src_counts[s] = src_counts.get(s, 0) + 1
print(f"\nAll raw results: {len(all_results)}, source breakdown: {src_counts}")
print(f"Result lists: {len(result_lists)} (sizes: {[len(l) for l in result_lists]})")

# 5. Dedup
seen = {}
unique = []
for r2 in all_results:
    cid = r2['chunk_id']
    if cid not in seen:
        seen[cid] = len(unique)
        unique.append(r2)
    else:
        existing = unique[seen[cid]]
        if r2.get('source') == 'graph_expansion' and existing.get('source') != 'graph_expansion':
            existing['source'] = 'graph_expansion'

dedup_src = {}
for u in unique:
    s = u.get('source', '?')
    dedup_src[s] = dedup_src.get(s, 0) + 1
print(f"\nAfter dedup: {len(unique)}, source breakdown: {dedup_src}")

# 6. RRF
fused = r.reciprocal_rank_fusion([unique])
fused_src = {}
for f in fused:
    s = f.get('source', '?')
    fused_src[s] = fused_src.get(s, 0) + 1
print(f"After RRF: {len(fused)}, source breakdown: {fused_src}")

# 7. Rerank
reranked = r.rerank_with_llm(query, fused, top_k=5)
rr_src = {}
for rr in reranked:
    s = rr.get('source', '?')
    rr_src[s] = rr_src.get(s, 0) + 1
print(f"After rerank top 5: {len(reranked)}, source breakdown: {rr_src}")
for i, rr in enumerate(reranked):
    print(f"  [{i+1}] doc_id={rr['doc_id']} source={rr['source']} rrf={rr.get('rrf_score',0):.4f}")

r.close()
