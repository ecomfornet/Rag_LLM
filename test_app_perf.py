"""
App Performance Test
====================
Calls app.py HTTP API and prints wall-clock + RAG latency_ms for complex queries.

Run (in another terminal while app.py is running):
    python test_app_perf.py
"""
import time
import requests


BASE = "http://localhost:5000"


def wait_health(timeout: int = 30) -> bool:
    print("Waiting for /health ...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{BASE}/health", timeout=3)
            if r.status_code == 200:
                print("Health OK:", r.text)
                return True
            print("Health status:", r.status_code, r.text)
        except Exception as e:
            print("Health error:", e)
        time.sleep(1)
    print("Health check timed out.")
    return False


def run_query(q: str):
    print("\n" + "=" * 70)
    print("QUERY:", q)
    t0 = time.time()
    r = requests.get(f"{BASE}/api/query", params={"query": q}, timeout=120)
    wall_ms = (time.time() - t0) * 1000.0
    print("HTTP status:", r.status_code)
    try:
        data = r.json()
    except Exception as e:
        print("JSON error:", e, "body:", r.text[:300])
        return
    meta = data.get("metadata", {}) or {}
    lm = meta.get("latency_ms", {}) or {}
    total_ms = lm.get("total", lm)
    answer = (data.get("answer") or "").strip()
    sources = data.get("sources") or []
    print("Wall ms:", round(wall_ms, 1))
    print("RAG total ms:", total_ms)
    print("Answer len:", len(answer), "Sources:", len(sources))


def main():
    if not wait_health():
        return 1

    q1 = "Dans l'arrêt CCJA N°173/2021, pourquoi la Cour a considéré qu'il fallait renvoyer les parties à l'arbitrage ?"
    q2 = "Quel est le taux d'imposition sur les plus-values de cession d'actions selon le CGI 2024 ?"

    for _ in range(2):
        run_query(q1)
    for _ in range(2):
        run_query(q2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

