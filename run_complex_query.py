"""Run a complex query and show response, sources, and agent trace."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# User query: arrêt CCJA 173/2021 — tests title/reference search + exact sources
query = (
    "Dans l'arrêt CCJA N°173/2021, pourquoi la Cour a considéré qu'il fallait "
    "renvoyer les parties à l'arbitrage?"
)

print("=" * 70)
print("COMPLEX QUERY TEST")
print("=" * 70)
print("Query:", query)
print()

if config.LANGGRAPH_ENABLED:
    from agents_langgraph import AgenticPipeline
    pipeline = AgenticPipeline()
    print("[*] Using AgenticPipeline (LangGraph)")
else:
    from main import GraphRAGPipeline
    pipeline = GraphRAGPipeline()
    print("[*] Using GraphRAGPipeline (classic)")

print()
response = pipeline.query(query)
pipeline.close()

# Answer
print("=" * 70)
print("ANSWER")
print("=" * 70)
answer = response.get("answer", "")
print(answer[:2000])
if len(answer) > 2000:
    print("\n... [truncated]")
print()

# Sources
print("=" * 70)
print("SOURCES")
print("=" * 70)
sources = response.get("sources", [])
for i, s in enumerate(sources[:10], 1):
    title = s.get("title") or s.get("doc_title") or "?"
    ref = s.get("reference", "?")
    score = s.get("score") or s.get("rrf_score", "?")
    src_type = s.get("source", "?")
    print(f"  {i}. {str(title)[:55]}...")
    print(f"     Ref: {ref} | Score: {score} | Type: {src_type}")
print(f"\n  Total: {len(sources)} sources")
print()

# Metadata
print("=" * 70)
print("METADATA")
print("=" * 70)
meta = response.get("metadata", {})
lat = meta.get("latency_ms", {})
total_ms = lat.get("total") if isinstance(lat, dict) else lat
print(f"  Chunks: {meta.get('num_chunks', '?')}")
print(f"  Latency: {total_ms} ms")
print(f"  Model: {meta.get('model', '?')}")
print(f"  Graph: {meta.get('graph_enabled', '?')}")
if response.get("agent_trace"):
    print("\n  Agent trace:")
    for t in response["agent_trace"]:
        print(f"    - {t.get('agent')}: {t.get('latency_ms')} ms | {t.get('notes', '')}")
