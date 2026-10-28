"""
GraphRAG Performance & Quality Benchmark
Runs the same query N times with cache cleared between each run.
"""
import time, json, sys, os

# Force UTF-8 output on Windows to avoid cp1252 emoji errors
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import GraphRAGPipeline

QUESTION = (
    "Quelle est la classification des travailleurs dans la Convention "
    "Collective des Mines, Carrières et Industries Assimilées ?"
)
RUNS = 1   # number of fresh (no-cache) runs

# ── helpers ──────────────────────────────────────────────────────────────────

def divider(title=""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)

def print_sources(sources):
    for i, s in enumerate(sources, 1):
        title = s.get("title") or s.get("doc_title") or "Sans titre"
        score = s.get("rrf_score") or s.get("score") or 0
        via   = s.get("source", "?")
        ref   = s.get("reference", "")
        year  = s.get("year", "")
        print(f"  {i:>2}. [{via:>16}]  score={score:.4f}  {title}")
        if ref:  print(f"       Réf:  {ref}")
        if year: print(f"       Année: {year}")
        preview = (s.get("content_preview") or "")[:180]
        if preview:
            print(f"       ↳ {preview}…")

# ── main ─────────────────────────────────────────────────────────────────────

print("\n🚀  GraphRAG Benchmark — initialising pipeline …")
pipeline = GraphRAGPipeline()

results = []

for run in range(1, RUNS + 1):
    divider(f"RUN {run}/{RUNS}  —  clearing all caches …")
    pipeline.cache.clear_all()
    time.sleep(0.3)          # tiny pause so Redis flush commits

    t0 = time.time()
    resp = pipeline.query(QUESTION, use_cache=False)
    elapsed = time.time() - t0

    meta = resp["metadata"]
    lat  = meta["latency_ms"]

    results.append({
        "run":          run,
        "total_ms":     lat["total"],
        "retrieval_ms": lat["retrieval"],
        "gen_ms":       lat["generation"],
        "chunks":       meta["num_chunks"],
        "tokens":       meta["usage"]["total_tokens"],
        "cost_usd":     meta["cost_usd"],
        "graph":        meta["graph_enabled"],
        "answer":       resp["answer"],
        "sources":      resp["sources"],
    })

    print(f"\n📄  ANSWER (run {run}):")
    print("-" * 70)
    print(resp["answer"])
    print("-" * 70)

    print(f"\n📚  SOURCES ({len(resp['sources'])} chunks):")
    print_sources(resp["sources"])

    print(f"\n📊  METRICS:")
    print(f"   Total latency  : {lat['total']:.0f} ms")
    print(f"   Retrieval      : {lat['retrieval']:.0f} ms")
    print(f"   Generation     : {lat['generation']:.0f} ms")
    print(f"   Chunks         : {meta['num_chunks']}")
    print(f"   Tokens         : {meta['usage']['total_tokens']}")
    print(f"   Cost           : ${meta['cost_usd']:.6f}")
    print(f"   Graph enabled  : {meta['graph_enabled']}")


# ── SUMMARY ──────────────────────────────────────────────────────────────────
divider("BENCHMARK SUMMARY")

def avg(key):
    return sum(r[key] for r in results) / len(results)

print(f"\n  Question : {QUESTION[:80]}…")
print(f"  Runs     : {RUNS}")
print()
print(f"  {'Metric':<22} {'Run 1':>10} {'Run 2':>10} {'Run 3':>10} {'AVG':>10}")
print(f"  {'-'*64}")

for label, key in [
    ("Total (ms)",     "total_ms"),
    ("Retrieval (ms)", "retrieval_ms"),
    ("Generation (ms)","gen_ms"),
    ("Chunks",         "chunks"),
    ("Tokens",         "tokens"),
    ("Cost (USD)",     "cost_usd"),
]:
    vals = [r[key] for r in results]
    row  = f"  {label:<22}"
    for v in vals:
        row += f" {v:>10.2f}" if isinstance(v, float) else f" {v:>10}"
    row += f" {avg(key):>10.2f}"
    print(row)

print()
# Answer consistency
unique_answers = set(r["answer"][:120] for r in results)
consistency = "✅ Consistent" if len(unique_answers) == 1 else f"⚠️  {len(unique_answers)} different answers"
print(f"  Answer consistency : {consistency}")
print(f"  Graph expansion    : {'✅ Enabled' if results[0]['graph'] else '❌ Disabled'}")

pipeline.close()
print("\n✅  Benchmark complete.\n")
