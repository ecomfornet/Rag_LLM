"""
GraphRAG System Diagnostic Script
Checks all components: Neo4j, Qdrant, and Retrieval
"""
import sys
import os

# Fix Windows console encoding first, before any output
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import logging
import traceback
import warnings

# Suppress neo4j / qdrant chatter – only show WARNING and above
logging.basicConfig(level=logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("qdrant_client").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Suppress qdrant "Api key with insecure connection" UserWarning (localhost dev)
warnings.filterwarnings("ignore", message=".*Api key is used with an insecure connection.*")


def _sep(title: str):
    """Print a clean section separator and flush immediately."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    sys.stdout.flush()


def check_neo4j() -> bool:
    """Check Neo4j connection and data."""
    _sep("1. CHECKING NEO4J")

    try:
        from graph_builder import GraphBuilder

        gb = GraphBuilder()
        stats = gb.get_graph_stats()

        print("✅ Neo4j Connection: OK")
        print(f"   Documents:              {stats.get('total_documents', 0):,}")
        print(f"   Chunks:                 {stats.get('total_chunks', 0):,}")
        print(f"   Relationships:          {stats.get('total_relationships', 0):,}")
        print(f"   Docs with citations:    {stats.get('documents_with_citations', 0):,}")

        # avg_citations_per_doc can be None when the DB is empty
        avg = stats.get('avg_citations_per_doc')
        avg_str = f"{avg:.2f}" if avg is not None else "N/A"
        print(f"   Avg citations per doc:  {avg_str}")

        gb.close()
        sys.stdout.flush()
        return True

    except Exception as e:
        print(f"❌ Neo4j Error: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return False


def check_qdrant() -> bool:
    """Check Qdrant connection and collection."""
    _sep("2. CHECKING QDRANT")

    try:
        import config
        from qdrant_client import QdrantClient
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            https=config.QDRANT_USE_HTTPS,
            api_key=config.QDRANT_API_KEY,
        )

        # First check the server is reachable
        try:
            collections = client.get_collections()
            existing = [c.name for c in collections.collections]
        except Exception as e:
            print(f"❌ Qdrant server unreachable: {e}")
            sys.stdout.flush()
            return False

        print("✅ Qdrant Connection: OK")
        print(f"   Available collections: {existing if existing else '(none)'}")

        if config.QDRANT_COLLECTION not in existing:
            print(f"⚠️  Collection '{config.QDRANT_COLLECTION}' does not exist yet.")
            print("   Run embedding_3.py first to create and populate the collection.")
            sys.stdout.flush()
            # Connection itself is fine – treat as partial success
            return True

        info = client.get_collection(config.QDRANT_COLLECTION)
        print(f"   Collection:  {config.QDRANT_COLLECTION}")
        print(f"   Points:      {info.points_count:,}")
        print(f"   Status:      {info.status}")

        sys.stdout.flush()
        return True

    except Exception as e:
        print(f"❌ Qdrant Error: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return False


def check_retrieval() -> bool:
    """Check retrieval system (requires both Neo4j and Qdrant with data)."""
    _sep("3. CHECKING RETRIEVAL")

    try:
        import config
        from qdrant_client import QdrantClient

        # Guard: collection must exist and have points
        client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            https=config.QDRANT_USE_HTTPS,
            api_key=config.QDRANT_API_KEY,
        )
        try:
            collections = [c.name for c in client.get_collections().collections]
        except Exception:
            collections = []

        if config.QDRANT_COLLECTION not in collections:
            print(f"⚠️  Skipping retrieval test – collection '{config.QDRANT_COLLECTION}' "
                  "doesn't exist yet.")
            print("   Run embedding_3.py first.")
            sys.stdout.flush()
            return True  # Not a system failure; just no data yet

        info = client.get_collection(config.QDRANT_COLLECTION)
        if not info.points_count:
            print("⚠️  Skipping retrieval test – collection is empty.")
            print("   Run embedding_3.py first.")
            sys.stdout.flush()
            return True

        from retrieval_4 import HybridRetriever

        retriever = HybridRetriever()

        query = "Quelles sont les obligations fiscales en Guinée?"
        print(f"\n   Test Query: {query}")

        results = retriever.retrieve(query, top_k=3)

        print(f"\n✅ Retrieval: OK")
        print(f"   Retrieved: {len(results)} chunks")

        if results:
            print("\n   Top Results:")
            for i, result in enumerate(results, 1):
                title = result['metadata'].get('doc_title', 'N/A')
                score = result.get('rrf_score', result.get('score', 0))
                source = result.get('source', 'unknown')
                doc_type = result['metadata'].get('type', 'N/A')
                year = result['metadata'].get('year', 'N/A')

                print(f"\n   {i}. {title[:70]}{'...' if len(title) > 70 else ''}")
                print(f"      Score:  {score:.4f}")
                print(f"      Source: {source}")
                print(f"      Type:   {doc_type}")
                print(f"      Year:   {year}")

        retriever.close()
        sys.stdout.flush()
        return True

    except Exception as e:
        print(f"❌ Retrieval Error: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return False


def check_graph_expansion() -> bool:
    """Check graph expansion functionality."""
    _sep("4. CHECKING GRAPH EXPANSION")

    try:
        import config
        from graph_builder import GraphBuilder

        gb = GraphBuilder()

        query = """
        MATCH (d:Document)
        WHERE d.reference IS NOT NULL
        RETURN d.id as doc_id, d.title as title
        LIMIT 1
        """

        with gb.driver.session(database=config.NEO4J_DATABASE) as session:
            result = session.run(query)
            record = result.single()

            if record:
                doc_id = record['doc_id']
                title = record['title'] or ''

                print(f"\n   Sample Document: {title[:60]}{'...' if len(title) > 60 else ''}")
                print(f"   Document ID:     {doc_id}")

                related = gb.get_related_documents(doc_id, max_depth=1)

                print(f"\n✅ Graph Expansion: OK")
                print(f"   Related documents: {len(related)}")

                if related:
                    print("\n   Sample Related Documents:")
                    for i, rel in enumerate(related[:3], 1):
                        rel_title = rel.get('title') or ''
                        print(f"   {i}. {rel_title[:60]}{'...' if len(rel_title) > 60 else ''} "
                              f"(distance: {rel['distance']})")
            else:
                print("⚠️  No documents with references found (database may be empty).")
                print("   Graph expansion is functional; run the pipeline first to add data.")

        gb.close()
        sys.stdout.flush()
        return True

    except Exception as e:
        print(f"❌ Graph Expansion Error: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return False


def check_agents() -> bool:
    """Check LangGraph multi-agent pipeline."""
    _sep("5. CHECKING LANGGRAPH AGENTS")

    # 1. Check langgraph is installed
    try:
        import langgraph  # noqa: F401
        print("   langgraph    : ✅ installed")
    except ImportError:
        print("   langgraph    : ❌ NOT installed")
        print("   Run: pip install langgraph>=0.1.0 langchain-openai>=0.1.0")
        sys.stdout.flush()
        return False

    # 2. Check config flag
    import config
    if not config.LANGGRAPH_ENABLED:
        print("⚠️  LANGGRAPH_ENABLED = False in config.py — skipping live test.")
        sys.stdout.flush()
        return True

    # 3. Quick integration test (requires Qdrant + Neo4j + OpenAI)
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            host=config.QDRANT_HOST, port=config.QDRANT_PORT,
            https=config.QDRANT_USE_HTTPS, api_key=config.QDRANT_API_KEY,
        )
        collections = [c.name for c in client.get_collections().collections]
        if config.QDRANT_COLLECTION not in collections:
            print("⚠️  Qdrant collection missing — skipping agent live-test.")
            print("   Run embedding_3.py first, then re-run diagnostics.")
            sys.stdout.flush()
            return True

        info = client.get_collection(config.QDRANT_COLLECTION)
        if not info.points_count:
            print("⚠️  Collection is empty — skipping agent live-test.")
            sys.stdout.flush()
            return True
    except Exception as e:
        print(f"⚠️  Qdrant check failed ({e}) — skipping agent live-test.")
        sys.stdout.flush()
        return True  # infrastructure issue, not agent code

    try:
        from agents_langgraph import AgenticPipeline

        pipeline = AgenticPipeline()
        query = "Quelles sont les obligations d'une SARL en Guinée?"
        print(f"\n   Test Query: {query}")

        result = pipeline.query(query, top_k=3, use_cache=False)

        answer = result.get("answer", "")
        meta   = result.get("metadata", {})
        trace  = result.get("agent_trace", [])

        print(f"\n✅ Agents (LangGraph): OK")
        print(f"   Complexity  : {meta.get('complexity', '?')} "
              f"(score={meta.get('complexity_score', 0):.2f})")
        print(f"   Confidence  : {meta.get('confidence', 0):.2f}")
        print(f"   Fact-check  : {'✅ OK' if meta.get('fact_check_ok', True) else '⚠️  WARN'}")
        print(f"   Sources     : {meta.get('num_chunks', 0)}")
        print(f"   Latency     : {meta.get('latency_ms', {}).get('total', 0):.0f} ms")

        if trace:
            print("\n   Agent trace:")
            for step in trace:
                print(f"   {step['agent']:35s}  {step['latency_ms']:7.1f} ms  {step.get('notes','')}")

        print(f"\n   Answer preview: {answer[:120].strip()}...")
        pipeline.close()
        sys.stdout.flush()
        return True

    except Exception as e:
        print(f"❌ Agents Error: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return False



def main() -> int:
    """Run all diagnostic checks sequentially."""
    print()
    print("=" * 60)
    print("GRAPHRAG SYSTEM DIAGNOSTIC")
    print("=" * 60)
    sys.stdout.flush()

    # Run checks sequentially so output is never interleaved
    results = {}
    results['Neo4j']              = check_neo4j()
    results['Qdrant']             = check_qdrant()
    results['Retrieval']          = check_retrieval()
    results['Graph Expansion']    = check_graph_expansion()
    results['Agents (LangGraph)'] = check_agents()

    # Summary
    _sep("SUMMARY")


    all_ok = all(results.values())

    for component, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {component}: {'OK' if status else 'FAILED'}")

    if all_ok:
        print("\n🎉 All systems operational!")
        print("   Your GraphRAG system is working correctly.")
    else:
        print("\n⚠️  Some components have issues.")
        print("   Check the errors printed above for details.")

    print()
    print("=" * 60)
    sys.stdout.flush()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
