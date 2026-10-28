"""
App Query Tests
===============
Multiple queries against the running Flask app.
Run: python app.py  (in another terminal)
Then: python test_app_queries.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

BASE = "http://localhost:5000"
TIMEOUT = 90

# French legal queries covering simple, complex, and multi-doc cases
QUERIES = [
    "Quel est le taux d'imposition sur les plus-values selon le CGI?",
    "Arrêt N° 004 du 11 Janvier 2017",
    "Quelles sont les formalités de création d'une SARL en Guinée?",
    "Définition de la société anonyme OHADA",
    "Abrogation décret fiscal 2024",
]


def run_queries():
    print("=" * 60)
    print("APP QUERY TESTS")
    print("=" * 60)
    print("Base URL:", BASE)
    print()

    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        if r.status_code != 200:
            print("App unhealthy. Start with: python app.py")
            return 1
    except requests.exceptions.ConnectionError:
        print("App not running. Start with: python app.py")
        return 1

    passed = 0
    failed = 0

    for i, query in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] {query[:60]}...")
        try:
            resp = requests.get(f"{BASE}/api/query", params={"query": query}, timeout=TIMEOUT)
            data = resp.json()

            if resp.status_code != 200:
                print(f"  FAIL - HTTP {resp.status_code}: {data.get('error', 'Unknown')}")
                failed += 1
                continue

            if data.get("error"):
                print(f"  FAIL - {data['error']}")
                failed += 1
                continue

            answer = data.get("answer", "")
            sources = data.get("sources", [])
            meta = data.get("metadata", {})

            has_answer = answer and len(answer.strip()) >= 50
            has_sources = len(sources) >= 1

            if has_answer and has_sources:
                print(f"  OK - answer {len(answer)} chars, {len(sources)} sources")
                passed += 1
            else:
                issues = []
                if not has_answer:
                    issues.append("answer too short")
                if not has_sources:
                    issues.append("no sources")
                print(f"  WARN - {', '.join(issues)}")

            # Brief answer preview
            preview = (answer or "No answer")[:120].replace("\n", " ")
            print(f"  Preview: {preview}...")

        except requests.exceptions.Timeout:
            print("  FAIL - timeout")
            failed += 1
        except Exception as e:
            print(f"  FAIL - {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULT: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_queries())
