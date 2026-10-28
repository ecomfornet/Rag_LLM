"""
GraphRAG Full Diagnostic Test
=============================
Verifies Neo4j, Qdrant, relations, and runs a real query through the pipeline.
Use: python test_graphrag_diagnostic.py
"""
import os
import sys
import json

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# Test queries (French legal - OHADA/Guinée)
TEST_QUERIES = [
    "Quel est le taux d'imposition sur les plus-values selon le CGI?",
    "Arrêt N° 004 du 11 Janvier 2017",
    "Quelles sont les formalités de création d'une SARL en Guinée?",
    "Définition de la société anonyme OHADA",
    "Abrogation d'un décret fiscal 2024",
]
TEST_QUERY = TEST_QUERIES[0]


def run_neo4j_diagnostics():
    """Verify Neo4j connectivity, node counts, and relationships."""
    print("\n" + "=" * 70)
    print("NEO4J DIAGNOSTICS")
    print("=" * 70)
    issues = []

    try:
        from neo4j import GraphDatabase
        from neo4j.exceptions import ServiceUnavailable

        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )

        with driver.session(database=config.NEO4J_DATABASE) as session:
            # 1. Connectivity
            session.run("RETURN 1")
            print("[OK] Neo4j connection successful")
            print(f"     URI: {config.NEO4J_URI}")

            # 2. Node counts
            result = session.run("MATCH (d:Document) RETURN count(d) as c")
            doc_count = result.single()["c"]
            result = session.run("MATCH (c:Chunk) RETURN count(c) as c")
            chunk_count = result.single()["c"]
            print(f"[OK] Documents: {doc_count}, Chunks: {chunk_count}")

            if doc_count == 0:
                issues.append("Neo4j: No Document nodes found - run ingestion first")
            if chunk_count == 0:
                issues.append("Neo4j: No Chunk nodes found - run chunking/import")

            # 3. Relationship types and counts
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as cnt
                ORDER BY cnt DESC
            """)
            rels = list(result)
            print("\n[+] Relationship types in DB:")
            expected = {"CITES", "AMENDS", "REPLACES", "BELONGS_TO", "MENTIONS", "INTERPRETS"}
            found_types = {r["rel_type"] for r in rels}
            for rec in rels:
                marker = "  " if rec["rel_type"] in expected else "  [!] "
                print(f"     {marker}{rec['rel_type']}: {rec['cnt']}")

            if not found_types:
                issues.append("Neo4j: No relationships found - graph expansion will not work")

            # Check BELONGS_TO (Chunk->Document) - critical
            belongs_to = next((r["cnt"] for r in rels if r["rel_type"] == "BELONGS_TO"), 0)
            if belongs_to == 0 and chunk_count > 0:
                issues.append("Neo4j: No BELONGS_TO relations - Chunks not linked to Documents")

            # 4. Sample Document with relations
            if doc_count > 0:
                result = session.run("""
                    MATCH (d:Document)-[r]-(other)
                    RETURN d.id as doc_id, d.title as title, type(r) as rel, labels(other)[0] as other_label
                    LIMIT 5
                """)
                samples = list(result)
                if samples:
                    print("\n[+] Sample document relations:")
                    for s in samples[:5]:
                        print(f"     Doc '{s['doc_id'][:40]}...' --[{s['rel']}]--> {s['other_label']}")

        driver.close()
        return issues

    except ServiceUnavailable as e:
        issues.append(f"Neo4j: Connection failed - {e}")
        print(f"[ERROR] Neo4j: {e}")
        return issues
    except Exception as e:
        issues.append(f"Neo4j: {e}")
        print(f"[ERROR] Neo4j: {e}")
        import traceback
        traceback.print_exc()
        return issues


def run_qdrant_diagnostics():
    """Verify Qdrant connectivity, collection, and sample vectors."""
    print("\n" + "=" * 70)
    print("QDRANT DIAGNOSTICS")
    print("=" * 70)
    issues = []

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            https=config.QDRANT_USE_HTTPS,
            api_key=config.QDRANT_API_KEY or None,
            timeout=10
        )

        collections = client.get_collections()
        names = [c.name for c in collections.collections]
        print(f"[OK] Qdrant connection: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
        print(f"     Collections: {names}")

        if config.QDRANT_COLLECTION not in names:
            issues.append(f"Qdrant: Collection '{config.QDRANT_COLLECTION}' not found")
            print(f"[!] Collection '{config.QDRANT_COLLECTION}' not found")
            return issues

        info = client.get_collection(config.QDRANT_COLLECTION)
        count = info.points_count
        print(f"[OK] Collection '{config.QDRANT_COLLECTION}': {count} vectors")

        if count == 0:
            issues.append("Qdrant: Collection is empty - run embedding/import")

        # Vector dimension check
        vectors_cfg = info.config.params.vectors
        if hasattr(vectors_cfg, "size"):
            vec_size = vectors_cfg.size
        else:
            vec_size = list(vectors_cfg.values())[0].size
        expected = config.EMBEDDING_DIMENSION
        if vec_size != expected:
            issues.append(f"Qdrant: Vector dim {vec_size} != expected {expected} (EMBEDDING_DIMENSION)")

        # Sample payloads
        if count > 0:
            scroll_result = client.scroll(
                collection_name=config.QDRANT_COLLECTION,
                limit=3,
                with_payload=True,
                with_vectors=False
            )
            points = scroll_result[0]
            print("\n[+] Sample payloads:")
            for i, p in enumerate(points, 1):
                payload = p.payload or {}
                doc_id = payload.get("doc_id", "?")
                chunk_id = payload.get("chunk_id", "?")
                content = (payload.get("content") or "")[:60]
                print(f"     {i}. doc_id={doc_id[:40] if doc_id else '?'}... chunk_id={chunk_id}")
                if not payload.get("doc_id"):
                    issues.append("Qdrant: Some payloads missing doc_id")

        return issues

    except Exception as e:
        issues.append(f"Qdrant: {e}")
        print(f"[ERROR] Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return issues


def run_pipeline_query(query: str = TEST_QUERY):
    """Run a real query through the pipeline (Agentic or Classic)."""
    print("\n" + "=" * 70)
    print("PIPELINE QUERY TEST")
    print("=" * 70)
    issues = []

    try:
        if config.LANGGRAPH_ENABLED:
            from agents_langgraph import AgenticPipeline
            pipeline = AgenticPipeline()
            print(f"[*] Using AgenticPipeline (LangGraph)")
        else:
            from main import GraphRAGPipeline
            pipeline = GraphRAGPipeline()
            print(f"[*] Using GraphRAGPipeline (classic)")

        print(f"[*] Query: {query[:80]}...")
        response = pipeline.query(query)

        answer = response.get("answer", "")
        sources = response.get("sources", [])
        metadata = response.get("metadata", {})

        print(f"\n[OK] Query completed")
        print(f"     Answer length: {len(answer)} chars")
        print(f"     Sources: {len(sources)}")
        print(f"     Chunks retrieved: {metadata.get('num_chunks', '?')}")
        print(f"     Latency: {metadata.get('latency_ms', {}).get('total', '?')} ms")

        if not answer or len(answer.strip()) < 50:
            issues.append("Pipeline: Answer too short or empty - check retrieval/LLM")

        if not sources:
            issues.append("Pipeline: No sources returned - retrieval may be failing")

        # Show first 200 chars of answer
        preview = (answer or "No answer")[:250].replace("\n", " ")
        print(f"\n[+] Answer preview:\n     {preview}...")

        # Close connections
        if hasattr(pipeline, "close"):
            pipeline.close()

        return issues

    except Exception as e:
        issues.append(f"Pipeline: {e}")
        print(f"[ERROR] Pipeline: {e}")
        import traceback
        traceback.print_exc()
        return issues


def run_qdrant_neo4j_alignment():
    """Verify Qdrant chunk_ids match Neo4j Chunk nodes."""
    print("\n" + "=" * 70)
    print("QDRANT-NEO4J ALIGNMENT")
    print("=" * 70)
    issues = []

    try:
        from neo4j import GraphDatabase
        from qdrant_client import QdrantClient

        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        qdrant = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            https=config.QDRANT_USE_HTTPS,
            api_key=config.QDRANT_API_KEY or None,
            timeout=10
        )

        # Sample 50 chunk_ids from Qdrant
        scroll = qdrant.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=50,
            with_payload=True,
            with_vectors=False
        )
        qdrant_chunk_ids = set()
        qdrant_doc_ids = set()
        for pt in scroll[0]:
            p = pt.payload or {}
            if p.get("chunk_id"):
                qdrant_chunk_ids.add(p["chunk_id"])
            if p.get("doc_id"):
                qdrant_doc_ids.add(str(p["doc_id"]))

        # Check these exist in Neo4j
        with driver.session(database=config.NEO4J_DATABASE) as session:
            missing_chunks = 0
            missing_docs = 0
            for cid in list(qdrant_chunk_ids)[:20]:  # Sample 20
                r = session.run("MATCH (c:Chunk {id: $id}) RETURN c.id as id", id=cid)
                if not r.single():
                    missing_chunks += 1
            for did in list(qdrant_doc_ids)[:20]:
                r = session.run("MATCH (d:Document {id: $id}) RETURN d.id as id", id=did)
                if not r.single():
                    missing_docs += 1

        driver.close()

        if missing_chunks == 0 and missing_docs == 0:
            print("[OK] Qdrant payloads align with Neo4j (sample check passed)")
        else:
            if missing_chunks > 0:
                issues.append(f"Qdrant-Neo4j: {missing_chunks}/20 sampled chunk_ids not found in Neo4j")
            if missing_docs > 0:
                issues.append(f"Qdrant-Neo4j: {missing_docs}/20 sampled doc_ids not found in Neo4j")
            print(f"[!] Alignment issues: chunks={missing_chunks}, docs={missing_docs}")

        return issues

    except Exception as e:
        issues.append(f"Alignment check: {e}")
        print(f"[ERROR] {e}")
        return issues


def run_api_query_test(query: str = TEST_QUERY):
    """Test via HTTP /api/query (requires app running on port 5000)."""
    print("\n" + "=" * 70)
    print("API QUERY TEST (requires app running: python app.py)")
    print("=" * 70)

    try:
        import requests
        url = "http://localhost:5000/api/query"
        resp = requests.get(url, params={"query": query}, timeout=60)
        data = resp.json()

        if resp.status_code != 200:
            print(f"[ERROR] HTTP {resp.status_code}: {data.get('error', 'Unknown')}")
            return [f"API: HTTP {resp.status_code}"]

        if data.get("error"):
            print(f"[ERROR] {data['error']}")
            return [f"API: {data['error']}"]

        print(f"[OK] API query succeeded")
        print(f"     Answer length: {len(data.get('answer', ''))} chars")
        print(f"     Sources: {len(data.get('sources', []))}")
        return []

    except requests.exceptions.ConnectionError:
        print("[SKIP] App not running - start with: python app.py")
        return []
    except Exception as e:
        print(f"[ERROR] API test: {e}")
        return [f"API: {e}"]


def main():
    print("\n" + "=" * 70)
    print("GRAPHRAG FULL DIAGNOSTIC")
    print("=" * 70)
    print(f"Config: LANGGRAPH_ENABLED={config.LANGGRAPH_ENABLED}")
    print(f"       NEO4J={config.NEO4J_URI}, QDRANT={config.QDRANT_HOST}:{config.QDRANT_PORT}")

    all_issues = []
    all_issues.extend(run_neo4j_diagnostics())
    all_issues.extend(run_qdrant_diagnostics())
    all_issues.extend(run_qdrant_neo4j_alignment())
    all_issues.extend(run_pipeline_query(TEST_QUERY))

    # Run extra queries if --multi passed (slow: ~1 min per query)
    if "--multi" in sys.argv and not any("Pipeline:" in i for i in all_issues):
        for q in TEST_QUERIES[1:3]:
            all_issues.extend(run_pipeline_query(q))

    # Optionally test API if server is running
    # all_issues.extend(run_api_query_test(TEST_QUERY))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if not all_issues:
        print("[OK] All checks passed. GraphRAG is healthy.")
    else:
        print("[!] Issues found:")
        for i, issue in enumerate(all_issues, 1):
            print(f"     {i}. {issue}")

    print()
    return 0 if not all_issues else 1


if __name__ == "__main__":
    sys.exit(main())
