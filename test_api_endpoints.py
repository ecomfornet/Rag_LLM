"""
API Endpoint Tests
==================
Tests for Flask app routes: /health, /api/diagnose, /api/query, /api/clear, /stats.
Requires: app running (python app.py) on localhost:5000

Run: python test_api_endpoints.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

BASE = "http://localhost:5000"
TIMEOUT = 90  # Query can be slow (agentic pipeline)


def test_health():
    """GET /health - basic health check."""
    print("\n[TEST] GET /health")
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert "status" in data, "Missing 'status' in response"
        assert data["status"] == "healthy", f"Status: {data.get('status')}"
        print("  OK - status:", data.get("status"))
        return True
    except requests.exceptions.ConnectionError:
        print("  SKIP - app not running (start: python app.py)")
        return False
    except AssertionError as e:
        print(f"  FAIL - {e}")
        return False


def test_diagnose_no_query():
    """GET /api/diagnose - DB check without running query."""
    print("\n[TEST] GET /api/diagnose?run_query=false")
    try:
        r = requests.get(f"{BASE}/api/diagnose", params={"run_query": "false"}, timeout=15)
        assert r.status_code == 200
        data = r.json()
        assert "neo4j" in data
        assert "qdrant" in data
        assert data["neo4j"]["ok"], f"Neo4j: {data['neo4j'].get('error')}"
        assert data["qdrant"]["ok"], f"Qdrant: {data['qdrant'].get('error')}"
        print(f"  OK - neo4j docs={data['neo4j'].get('docs', 0)}, qdrant vectors={data['qdrant'].get('vectors', 0)}")
        return True
    except requests.exceptions.ConnectionError:
        print("  SKIP - app not running")
        return False
    except AssertionError as e:
        print(f"  FAIL - {e}")
        return False


def test_diagnose_with_query():
    """GET /api/diagnose?query=... - full diagnose including query."""
    print("\n[TEST] GET /api/diagnose?query=... (runs pipeline query)")
    try:
        q = "Quel est le CGI?"
        r = requests.get(f"{BASE}/api/diagnose", params={"query": q, "run_query": "true"}, timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        qt = data.get("query_test", {})
        assert qt.get("run"), "query_test.run should be True"
        assert qt.get("answer_preview"), "Expected answer preview"
        print(f"  OK - answer_preview len={len(qt.get('answer_preview', ''))}, sources={qt.get('sources_count', 0)}")
        return True
    except requests.exceptions.ConnectionError:
        print("  SKIP - app not running")
        return False
    except AssertionError as e:
        print(f"  FAIL - {e}")
        return False


def test_query_get():
    """GET /api/query?query=..."""
    print("\n[TEST] GET /api/query?query=...")
    try:
        q = "Définition SARL OHADA"
        r = requests.get(f"{BASE}/api/query", params={"query": q}, timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert "error" not in data or not data["error"], f"API error: {data.get('error')}"
        assert "answer" in data
        assert "sources" in data
        assert "metadata" in data
        print(f"  OK - answer len={len(data.get('answer', ''))}, sources={len(data.get('sources', []))}")
        return True
    except requests.exceptions.ConnectionError:
        print("  SKIP - app not running")
        return False
    except AssertionError as e:
        print(f"  FAIL - {e}")
        return False


def test_query_post():
    """POST /api/query with JSON body."""
    print("\n[TEST] POST /api/query (JSON body)")
    try:
        r = requests.post(
            f"{BASE}/api/query",
            json={"query": "Arrêt N° 004 du 11 Janvier 2017"},
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        assert r.status_code == 200
        data = r.json()
        assert "error" not in data or not data["error"]
        assert "answer" in data
        print(f"  OK - answer len={len(data.get('answer', ''))}")
        return True
    except requests.exceptions.ConnectionError:
        print("  SKIP - app not running")
        return False
    except AssertionError as e:
        print(f"  FAIL - {e}")
        return False


def test_query_no_param():
    """GET /api/query without query - expect 400."""
    print("\n[TEST] GET /api/query (no query param -> 400)")
    try:
        r = requests.get(f"{BASE}/api/query", timeout=5)
        assert r.status_code == 400
        data = r.json()
        assert "error" in data
        print("  OK - 400 as expected")
        return True
    except requests.exceptions.ConnectionError:
        print("  SKIP - app not running")
        return False
    except AssertionError as e:
        print(f"  FAIL - {e}")
        return False


def test_query_too_long():
    """Query exceeding MAX_QUERY_LENGTH -> 400."""
    print("\n[TEST] POST /api/query (query too long -> 400)")
    try:
        r = requests.post(
            f"{BASE}/api/query",
            json={"query": "x" * 3000},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        assert r.status_code == 400
        data = r.json()
        assert "error" in data and "long" in data["error"].lower()
        print("  OK - 400 as expected")
        return True
    except requests.exceptions.ConnectionError:
        print("  SKIP - app not running")
        return False
    except AssertionError as e:
        print(f"  FAIL - {e}")
        return False


def test_stats():
    """GET /stats - cache statistics."""
    print("\n[TEST] GET /stats")
    try:
        r = requests.get(f"{BASE}/stats", timeout=5)
        assert r.status_code == 200
        data = r.json()
        # Stats structure may vary; just ensure it's a dict
        assert isinstance(data, dict)
        print("  OK - cache stats returned")
        return True
    except requests.exceptions.ConnectionError:
        print("  SKIP - app not running")
        return False
    except AssertionError as e:
        print(f"  FAIL - {e}")
        return False


def test_clear_cache():
    """POST /api/clear - clear all caches."""
    print("\n[TEST] POST /api/clear")
    try:
        r = requests.post(f"{BASE}/api/clear", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "cleared"
        print("  OK - caches cleared")
        return True
    except requests.exceptions.ConnectionError:
        print("  SKIP - app not running")
        return False
    except AssertionError as e:
        print(f"  FAIL - {e}")
        return False


def main():
    print("=" * 60)
    print("API ENDPOINT TESTS")
    print("=" * 60)
    print("Ensure app is running: python app.py")
    print("Base URL:", BASE)

    tests = [
        test_health,
        test_diagnose_no_query,
        test_stats,
        test_clear_cache,
        test_query_no_param,
        test_query_too_long,
        test_query_get,
        test_query_post,
        test_diagnose_with_query,
    ]

    results = []
    for t in tests:
        try:
            results.append(t())
        except Exception as e:
            print(f"  ERROR - {e}")
            results.append(False)

    passed = sum(1 for r in results if r)
    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{len(results)} passed")
    print("=" * 60)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
