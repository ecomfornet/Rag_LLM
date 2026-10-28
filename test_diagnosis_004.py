# -*- coding: utf-8 -*-
"""
Diagnostic Test: Arret N 004 du 11 Janvier 2017
================================================
Tests the specific query 3 times (with cache clear between runs),
inspecting router complexity, chunks retrieved, answer quality.
Run: python test_diagnosis_004.py  (app.py must be running on port 5000)
"""
import sys
import os
import time
import re
import io

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests

BASE = "http://localhost:5000"
TIMEOUT = 120

QUERY = (
    "Selon l'arret de la Cour d'Appel de Kankan, Chambre du Premier President, "
    "Arret N 004 du 11 Janvier 2017, quels sont les elements retenus pour constater "
    "le droit de propriete"
)

QUALITY_KEYWORDS = [
    "titre foncier", "donation", "decret", "article", "inscription",
    "jugement", "tribunal", "propriete",
]

def clear_cache():
    try:
        r = requests.post(f"{BASE}/api/clear", timeout=10)
        if r.status_code == 200:
            print("  [OK] Cache cleared")
        else:
            print(f"  [WARN] Cache clear returned {r.status_code}: {r.text[:80]}")
    except Exception as e:
        print(f"  [WARN] Cache clear failed: {e}")

def check_health():
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def keyword_hit(text, kw):
    """Check keyword (accent-insensitive)."""
    import unicodedata
    def norm(s):
        return unicodedata.normalize('NFD', s.lower()).encode('ascii', 'ignore').decode()
    return norm(kw) in norm(text)

def run_query(run_num: int, clear: bool = True):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"RUN #{run_num}")
    print(sep)

    if clear:
        clear_cache()

    t0 = time.time()
    try:
        # Use original French query in the actual API call
        real_query = (
            "Selon l'arr\u00eat de la Cour d'Appel de Kankan, Chambre du Premier Pr\u00e9sident, "
            "Arr\u00eat N\u00b0 004 du 11 Janvier 2017, quels sont les \u00e9l\u00e9ments retenus pour constater "
            "le droit de propri\u00e9t\u00e9"
        )
        resp = requests.post(
            f"{BASE}/api/query",
            json={"query": real_query},
            timeout=TIMEOUT,
        )
        elapsed = time.time() - t0
        data = resp.json()
    except requests.exceptions.Timeout:
        print("  [FAIL] TIMEOUT")
        return None
    except Exception as e:
        print(f"  [FAIL] ERROR: {e}")
        return None

    if resp.status_code != 200:
        print(f"  [FAIL] HTTP {resp.status_code}: {data.get('error', 'Unknown')}")
        return None

    if data.get("error"):
        print(f"  [FAIL] API error: {data['error']}")
        return None

    answer  = data.get("answer", "")
    sources = data.get("sources", [])
    meta    = data.get("metadata", {})
    cached  = data.get("cached", False)
    trace   = data.get("agent_trace", [])

    # --- Router diagnosis ---
    router_step = next((s for s in trace if s["agent"] == "RouterAgent"), None)
    fast_step   = next((s for s in trace if s["agent"] == "SimpleFastPath"), None)

    route_taken = "SimpleFastPath [WEAK]" if fast_step else "Complex/Full Pipeline"
    router_notes = router_step["notes"] if router_step else "N/A"

    print(f"\n  ROUTING")
    print(f"    Route taken   : {route_taken}")
    print(f"    Router notes  : {router_notes}")
    print(f"    Cached        : {cached}")
    print(f"    Elapsed       : {elapsed:.1f}s")

    # --- Retrieval diagnosis ---
    num_chunks = meta.get("num_chunks", len(sources))
    print(f"\n  RETRIEVAL")
    print(f"    Chunks retrieved : {num_chunks}")
    print(f"    Sources shown    : {len(sources)}")
    for i, src in enumerate(sources[:6], 1):
        title = (src.get("title") or src.get("doc_title") or "?")[:55]
        score = src.get("rrf_score") or src.get("score") or 0
        src_type = src.get("source", "?")
        print(f"    [{i}] {title} | score={score:.4f} | {src_type}")

    # --- Answer quality checks ---
    found_kw = [kw for kw in QUALITY_KEYWORDS if keyword_hit(answer, kw)]
    missing_kw = [kw for kw in QUALITY_KEYWORDS if not keyword_hit(answer, kw)]

    confidence_m = re.search(r"CONFIANCE.*?([0-9]\.[0-9]+)", answer)
    confidence = float(confidence_m.group(1)) if confidence_m else None

    has_titre  = keyword_hit(answer, "titre foncier")
    has_don    = keyword_hit(answer, "donation")
    has_decret = keyword_hit(answer, "d/94") or keyword_hit(answer, "decret")
    has_arts   = re.search(r"article\s+\d+", answer, re.IGNORECASE) is not None

    print(f"\n  ANSWER QUALITY")
    print(f"    Length            : {len(answer)} chars")
    print(f"    Confidence        : {confidence}")
    print(f"    Keywords found    : {len(found_kw)}/{len(QUALITY_KEYWORDS)} -> {found_kw}")
    print(f"    Keywords MISSING  : {missing_kw}")
    print(f"    Has titre foncier : {has_titre}")
    print(f"    Has donation      : {has_don}")
    print(f"    Has decret ref    : {has_decret}")
    print(f"    Has article ref   : {has_arts}")

    # LLM says "not in context"?
    vague_phrases = [
        "ne sont pas explicitement", "ne figure pas", "non disponible",
        "pas dans le contexte", "limitee a ce que", "limite a ce que"
    ]
    vague_hits = [p for p in vague_phrases if p in answer.lower().replace('\u00e9','e').replace('\u00e8','e').replace('\u00ea','e')]
    if vague_hits:
        print(f"    [!] LLM VAGUE PHRASES: {vague_hits}")

    # --- Answer preview ---
    preview = answer[:500].replace("\n", " ")
    print(f"\n  ANSWER PREVIEW:")
    print(f"    {preview}...")

    # --- Agent trace ---
    print(f"\n  AGENT TRACE:")
    for step in trace:
        note = (step.get('notes',''))[:80]
        print(f"    [{step['latency_ms']:.0f}ms] {step['agent']}: {note}")

    # Score
    quality_score = len(found_kw) / len(QUALITY_KEYWORDS)
    verdict = "[GOOD]" if (quality_score >= 0.5 and len(answer) > 500 and not vague_hits) else "[WEAK]"
    print(f"\n  VERDICT: {verdict} (quality={quality_score:.0%}, route={route_taken})")

    return {
        "run": run_num,
        "route": route_taken,
        "cached": cached,
        "num_chunks": num_chunks,
        "answer_len": len(answer),
        "confidence": confidence,
        "quality_score": quality_score,
        "verdict": verdict,
        "router_notes": router_notes,
        "vague": bool(vague_hits),
        "missing_kw": missing_kw,
    }


def main():
    sep = "=" * 60
    print(f"{sep}")
    print("DIAGNOSTIC: Arret N 004 - Response Consistency")
    print(f"{sep}")

    if not check_health():
        print("[FAIL] App not running. Start with: python app.py")
        return 1

    results = []

    # Run 1: fresh (cache cleared)
    results.append(run_query(1, clear=True))
    time.sleep(1)

    # Run 2: WITHOUT clearing cache (tests cache hit behavior)
    print("\n--- Run 2: WITH cache (testing cache hit) ---")
    results.append(run_query(2, clear=False))
    time.sleep(1)

    # Run 3: clear cache and run again (tests reproducibility)
    print("\n--- Run 3: FRESH (cache cleared again) ---")
    results.append(run_query(3, clear=True))

    # Summary
    print(f"\n\n{sep}")
    print("SUMMARY")
    print(sep)
    valid = [r for r in results if r is not None]
    for r in valid:
        print(
            f"  Run #{r['run']}: {r['verdict']} | route={r['route'][:25]} | "
            f"chunks={r['num_chunks']} | len={r['answer_len']} | "
            f"conf={r['confidence']} | cached={r['cached']} | vague={r['vague']}"
        )

    # Detect problems
    print(f"\nDIAGNOSIS:")
    routes = list({r["route"] for r in valid})
    if "SimpleFastPath [WEAK]" in routes:
        print("  [ROOT CAUSE 1] SimpleFastPath used - skips BM25, graph, rerank!")
    else:
        print("  [ OK ] Always uses full pipeline")

    cached_runs = [r for r in valid if r["cached"]]
    weak_cached = [r for r in cached_runs if r["verdict"] == "[WEAK]"]
    if weak_cached:
        print("  [ROOT CAUSE 2] Cache hit serving WEAK answer!")
    else:
        print("  [ OK ] No bad cache hits detected")

    all_missing = {}
    for r in valid:
        for kw in r.get("missing_kw", []):
            all_missing[kw] = all_missing.get(kw, 0) + 1
    consistent_missing = [kw for kw, cnt in all_missing.items() if cnt >= 2]
    if consistent_missing:
        print(f"  [ROOT CAUSE 3] Consistently missing context keywords: {consistent_missing}")
        print("    => These terms are likely NOT in the retrieved chunks (retrieval gap)")
    else:
        print("  [ OK ] No consistent retrieval gaps found")

    verdicts = [r["verdict"] for r in valid]
    if "[WEAK]" in verdicts:
        print(f"\n  ACTION NEEDED: {verdicts.count('[WEAK]')} weak run(s) out of {len(valid)}")
    else:
        print(f"\n  ALL RUNS GOOD!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
