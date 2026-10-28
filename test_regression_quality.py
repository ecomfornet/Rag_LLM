"""
Regression Quality + Latency Test
=================================
Goal: ensure recent optimizations did NOT downgrade RAG quality.

What it checks:
- Answer is non-empty
- At least 1 source returned
- For exact-reference queries (e.g. N°173/2021), at least one top source title/reference contains that ref
- Prints latency and agent trace timings (agentic pipeline)

Run:
  python test_regression_quality.py
  python test_regression_quality.py --multi   (adds more slow queries)
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


REF_RE = re.compile(r"\b(\d{2,4}/\d{4})\b")


QUERIES = [
    # Exact reference: should lock onto the right decision by title/reference
    "Dans l'arrêt CCJA N°173/2021, pourquoi la Cour a considéré qu'il fallait renvoyer les parties à l'arbitrage ?",
    # General legal: should still work
    "Quel est le taux d'imposition sur les plus-values de cession d'actions selon le CGI 2024 ?",
    "Quelles sont les formalités de création d'une SARL en Guinée ?",
]

QUERIES_MULTI = [
    "Selon l'arrêt CCJA N°018/2003, quelle a été la décision prise par la CCJA et sur quel fondement ?",
    "Compare les formalités de création d'une SARL selon l'OHADA et le droit guinéen. Quels sont les délais et les textes applicables en 2024 ?",
]


def _pick_pipeline():
    if config.LANGGRAPH_ENABLED:
        from agents_langgraph import AgenticPipeline
        return AgenticPipeline(), "agentic"
    from main import GraphRAGPipeline
    return GraphRAGPipeline(), "classic"


def _extract_ref(query: str) -> str | None:
    m = REF_RE.search(query.lower())
    return m.group(1) if m else None


def _source_text(s: dict) -> str:
    title = str(s.get("title") or s.get("doc_title") or "")
    ref = str(s.get("reference") or "")
    return (title + " " + ref).lower()


def run_one(pipeline, query: str) -> tuple[bool, list[str]]:
    issues: list[str] = []

    t0 = time.time()
    resp = pipeline.query(query)
    elapsed_ms = (time.time() - t0) * 1000.0

    answer = (resp.get("answer") or "").strip()
    sources = resp.get("sources") or []
    meta = resp.get("metadata") or {}
    agent_trace = resp.get("agent_trace") or []

    if len(answer) < 80:
        issues.append(f"answer_too_short(len={len(answer)})")
    if len(sources) < 1:
        issues.append("no_sources")

    # Exact reference queries must have an exact matching source near the top
    ref = _extract_ref(query)
    if ref:
        top = sources[:5]
        if not any(ref in _source_text(s) for s in top):
            issues.append(f"missing_exact_ref_in_top_sources(ref={ref})")

    # Print a compact report
    latency = meta.get("latency_ms", {})
    total_ms = latency.get("total") if isinstance(latency, dict) else latency
    print("\n" + "=" * 70)
    print("QUERY:", query)
    print("- elapsed_ms (wall):", round(elapsed_ms, 1))
    print("- latency_ms (reported):", total_ms)
    print("- sources:", len(sources))
    print("- answer_len:", len(answer))
    if sources:
        print("Top sources:")
        for i, s in enumerate(top if ref else sources[:3], 1):
            print(f"  {i}. {str(s.get('title') or s.get('doc_title') or '?')[:70]} | ref={s.get('reference','')}")
    if agent_trace:
        print("Agent trace:")
        for step in agent_trace:
            print(f"  - {step.get('agent')}: {step.get('latency_ms')} ms")

    ok = len(issues) == 0
    if not ok:
        print("ISSUES:", issues)
    else:
        print("OK")
    return ok, issues


def main():
    queries = list(QUERIES)
    if "--multi" in sys.argv:
        queries += QUERIES_MULTI

    pipeline, mode = _pick_pipeline()
    print("=" * 70)
    print("REGRESSION TEST")
    print("=" * 70)
    print("mode:", mode)
    print("queries:", len(queries))

    failures: list[tuple[str, list[str]]] = []
    try:
        for q in queries:
            ok, issues = run_one(pipeline, q)
            if not ok:
                failures.append((q, issues))
    finally:
        if hasattr(pipeline, "close"):
            pipeline.close()

    print("\n" + "=" * 70)
    if not failures:
        print("RESULT: PASS")
        return 0
    print("RESULT: FAIL")
    for q, iss in failures:
        print("-", q)
        print("  ", iss)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

