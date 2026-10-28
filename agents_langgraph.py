"""
agents_langgraph.py
===================
Multi-Agent Orchestration Layer — Strategy 3 Expert
-----------------------------------------------------
Implements a 6-node LangGraph StateGraph on top of the existing
HybridRetriever / LLMGenerator infrastructure.

Graph topology
--------------
START
  └─► router_node
        ├─ (simple)  ─► simple_path_node ─► END
        └─ (complex) ─► graph_navigator_node
                              └─► vector_specialist_node
                                        └─► temporal_analyst_node
                                                  └─► fusion_reranker_node
                                                            └─► legal_reasoner_node
                                                                      └─► END

No embeddings or Neo4j data are modified — this is a pure orchestration layer.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Any, Dict, List, Optional, TypedDict

# --------------------------------------------------------------------------- #
# LangGraph / LangChain imports (installed via requirements.txt)               #
# --------------------------------------------------------------------------- #
try:
    from langgraph.graph import StateGraph, END, START
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False

from openai import OpenAI

import config
from retrieval_4 import HybridRetriever
from generation_5 import LLMGenerator, ResponseFormatter
from caching_6 import CacheManager

logger = logging.getLogger(__name__)


# =========================================================================== #
# Helper: Future Resolution                                                     #
# =========================================================================== #

def _resolve_future(fut_or_val: Any, default: Any = None) -> Any:
    """
    Safely resolve a concurrent.futures.Future or return the value if already resolved.
    Catches and logs exceptions, returning default on failure.
    """
    if fut_or_val is None:
        return default
    if hasattr(fut_or_val, "result"):
        try:
            res = fut_or_val.result()
            return res if res is not None else default
        except Exception as e:
            logger.warning(f"Error resolving future: {e}")
            return default
    return fut_or_val


# =========================================================================== #
# Shared State                                                                  #
# =========================================================================== #

class AgentState(TypedDict, total=False):
    """State dict passed between every node in the LangGraph."""

    # Input
    query:          str
    top_k:          int
    filters:        Optional[Dict[str, Any]]
    use_cache:      bool
    use_graph:      bool

    # Pre-warming futures & resolved results (Contract C)
    prewarm_seed_vec:    Optional[Any]  # Future or List[Dict] (seed vector search)
    prewarm_title_ref:   Optional[Any]  # Future or List[Dict] (title reference search)
    prewarm_bm25:        Optional[Any]  # Future or List[Dict] (BM25 sparse search)
    prewarm_plan:        Optional[Any]  # Future or Dict (query analysis plan)

    # Router output
    complexity:     str        # "simple" | "complex" | "multi-doc"
    complexity_score: float    # 0-1
    legal_entities: List[str]  # detected entities (SARL, Article 15, …)

    # Graph navigator output
    graph_doc_ids:  List[str]  # candidate doc_ids from Neo4j traversal

    # Vector specialist output (Contract C)
    vector_result_lists: List[List[Dict]]  # independent ranked lists for multi-list RRF
    vector_results:      List[Dict]        # flat list for backward compatibility
    queries:             List[str]         # original + expanded variants
    metadata_filters:    Dict[str, Any]

    # Temporal analyst output
    temporal_filter: Optional[Dict[str, Any]]  # e.g. {'year': 2023}
    temporal_detected: bool

    # Fusion + reranker output
    fused_results:  List[Dict]  # RRF-merged + reranked top-K chunks
    context:        str         # formatted context string

    # Legal reasoner + fact-checker output
    llm_result:     Dict        # raw generate_response() output
    answer:         str
    confidence:     float
    fact_check_ok:  bool
    fact_check_issues: List[str]

    # Final response (mirrors classic pipeline output)
    response:       Dict
    agent_trace:    List[Dict]  # one entry per agent: {name, latency_ms, notes}

    # Cache
    cached:         bool
    latency_ms:     float


# =========================================================================== #
# Helper utilities                                                              #
# =========================================================================== #

def _trace(state: AgentState, name: str, start: float, notes: str = "") -> None:
    """Append a trace entry for the given agent."""
    entry = {
        "agent":      name,
        "latency_ms": round((time.time() - start) * 1000, 1),
        "notes":      notes,
    }
    state.setdefault("agent_trace", []).append(entry)


# =========================================================================== #
# Agent 1 — Router                                                              #
# =========================================================================== #

class RouterAgent:
    """
    Classifies query complexity (simple / complex / multi-doc) using GPT-4o-mini.
    Detects legal entities mentioned in the query.
    Decides which downstream path the StateGraph follows.
    Concurrently pre-warms seed vector, title reference, and BM25 searches on shared pool.
    """

    SYSTEM_PROMPT = (
        "Tu es un classificateur de requêtes juridiques OHADA/Guinée.\n"
        "Pour la requête fournie, réponds en JSON strict (pas de markdown) avec:\n"
        "{\n"
        '  "complexity": "simple" | "complex" | "multi-doc",\n'
        '  "score": <float 0-1>,\n'
        '  "entities": ["<entité1>", "<entité2>"],\n'
        '  "reason": "<courte justification>"\n'
        "}\n\n"
        "Règles:\n"
        "- simple   : question factuelle directe, une seule loi/article visé\n"
        "- complex  : raisonnement multi-étapes, jurisprudence (arrêts, jugements), hiérarchie de normes\n"
        "- multi-doc: comparaison de plusieurs textes ou évolutions dans le temps\n"
        "ATTENTION: Toute requête mentionnant un arrêt, un jugement, une affaire ou un numéro de décision DOIT être classée comme 'complex'."
    )

    def __init__(
        self,
        client: OpenAI,
        retriever: Optional[HybridRetriever] = None,
        pool: Optional[ThreadPoolExecutor] = None,
    ):
        self.client = client
        self.retriever = retriever
        self.pool = pool

    def run(self, state: AgentState) -> AgentState:
        t0 = time.time()
        query = state["query"]
        filters = state.get("filters")
        pool = self.pool or state.get("_pool")

        # ── Launch concurrent pre-warming on shared thread pool ─── #
        fut_seed_vec = None
        fut_title_ref = None
        fut_bm25 = None

        if pool is not None and self.retriever is not None:
            # 1. Seed vector search (Dense Top K)
            try:
                fut_seed_vec = pool.submit(
                    self.retriever.vector_search,
                    query,
                    top_k=config.DENSE_TOP_K,
                    filters=filters,
                )
            except Exception as e:
                logger.warning(f"RouterAgent: failed to submit prewarm vector_search: {e}")

            # 2. Title reference search (top_k=15 for exact reference match)
            try:
                fut_title_ref = pool.submit(
                    self.retriever.title_reference_search,
                    query,
                    top_k=15,
                )
            except Exception as e:
                logger.warning(f"RouterAgent: failed to submit prewarm title_reference_search: {e}")

            # 3. BM25 sparse search
            if getattr(config, "BM25_ENABLED", True):
                try:
                    fut_bm25 = pool.submit(
                        self.retriever.bm25_search,
                        query,
                        top_k=config.BM25_TOP_K,
                    )
                except Exception as e:
                    logger.warning(f"RouterAgent: failed to submit prewarm bm25_search: {e}")

        # Set prewarm futures immediately on state
        state["prewarm_seed_vec"] = fut_seed_vec
        state["prewarm_title_ref"] = fut_title_ref
        state["prewarm_bm25"] = fut_bm25
        state["prewarm_plan"] = None

        # ── LLM classification executes concurrently on main thread ── #
        try:
            resp = self.client.chat.completions.create(
                model=config.SIMPLE_QUERY_MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": query},
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
                timeout=5.0
            )
            import json
            data = json.loads(resp.choices[0].message.content)
            complexity       = data.get("complexity", "complex")
            complexity_score = float(data.get("score", 0.5))
            legal_entities   = data.get("entities", [])
            reason           = data.get("reason", "")
        except Exception as e:
            logger.warning(f"RouterAgent fallback (error: {e})")
            complexity       = "complex"
            complexity_score = 0.5
            legal_entities   = []
            reason           = f"parse error: {e}"

        # Hard override for explicit jurisprudence references
        import re
        JURIS_RE = re.compile(
            r"(arrêt|jugement|arrêts|décision|chambre|tribunal|cour)\s*(n[°o]?\s*\d+|du\s+\d+)",
            re.IGNORECASE
        )
        if JURIS_RE.search(query):
            complexity = "complex"
            complexity_score = 0.9  # Force above AGENT_SIMPLE_THRESHOLD
            reason = f"hard override: jurisprudence pattern matched -> original was {complexity}({complexity_score})"

        state["complexity"]       = complexity
        state["complexity_score"] = complexity_score
        state["legal_entities"]   = legal_entities
        _trace(state, "RouterAgent", t0, f"{complexity} (score={complexity_score:.2f}) [prewarmed] | {reason}")
        logger.info(f"[Router] complexity={complexity} score={complexity_score:.2f} entities={legal_entities}")
        return state


# =========================================================================== #
# Agent 2 — Graph Navigator                                                     #
# =========================================================================== #

class GraphNavigatorAgent:
    """
    Explores Neo4j for documents related to the detected legal entities.
    Builds a candidate doc_id set used by the Vector Specialist.
    Concurrently starts query analysis on thread pool and reuses prewarmed seed vector search.
    """

    def __init__(self, retriever: HybridRetriever, pool: Optional[ThreadPoolExecutor] = None):
        self.retriever = retriever
        self.pool = pool

    def run(self, state: AgentState) -> AgentState:
        t0 = time.time()
        entities  = state.get("legal_entities", [])
        filters   = state.get("filters")
        query     = state["query"]
        use_graph = state.get("use_graph", True)
        pool      = self.pool or state.get("_pool")

        # 1. Asynchronously dispatch analyze_query to thread pool as prewarm_plan
        if state.get("prewarm_plan") is None and pool is not None:
            try:
                state["prewarm_plan"] = pool.submit(self.retriever.analyze_query, query)
                logger.debug("[GraphNavigator] Dispatched analyze_query to thread pool")
            except Exception as e:
                logger.warning(f"GraphNavigator: failed to submit prewarm analyze_query: {e}")

        graph_doc_ids: List[str] = []

        try:
            # 2. Consume pre-warmed seed vector search or fallback to fresh search
            init_results = _resolve_future(state.get("prewarm_seed_vec"), default=None)
            if init_results is None:
                init_results = self.retriever.vector_search(
                    query,
                    top_k=config.DENSE_TOP_K,
                    filters=filters,
                )

            # Store resolved List[Dict] back into state so VectorSpecialist can reuse it without re-running
            state["prewarm_seed_vec"] = init_results

            seed_doc_ids = list({r["doc_id"] for r in init_results if r.get("doc_id")})

            # 3. Expand via graph traversal (Contract A: get_related_documents_batch via retriever)
            if use_graph and config.GRAPH_EXPANSION_ENABLED and seed_doc_ids:
                expanded = self.retriever.graph_expansion(seed_doc_ids)
                graph_doc_ids = expanded
            else:
                graph_doc_ids = seed_doc_ids

            notes = (
                f"seed={len(seed_doc_ids)} -> expanded={len(graph_doc_ids)} doc_ids "
                f"| entities={entities}"
            )
        except Exception as e:
            logger.warning(f"GraphNavigatorAgent error (non-fatal): {e}")
            graph_doc_ids = []
            notes = f"error: {e}"

        state["graph_doc_ids"] = graph_doc_ids
        _trace(state, "GraphNavigatorAgent", t0, notes)
        logger.info(f"[GraphNavigator] {len(graph_doc_ids)} candidate doc_ids")
        return state


# =========================================================================== #
# Agent 3 — Vector Specialist                                                   #
# =========================================================================== #

class VectorSpecialistAgent:
    """
    Runs query expansion, parallel vector search, and BM25 sparse search.
    Reuses pre-warmed searches from Router and GraphNavigator.
    Batch-embeds all query variants with embed_queries_batch in a single call.
    Produces both vector_result_lists (for multi-list RRF) and vector_results (flat).
    """

    def __init__(self, retriever: HybridRetriever, pool: Optional[ThreadPoolExecutor] = None):
        self.retriever = retriever
        self.pool = pool

    def run(self, state: AgentState) -> AgentState:
        t0      = time.time()
        query   = state["query"]
        filters = state.get("filters")
        pool    = self.pool or state.get("_pool")

        result_lists: List[List[Dict]] = []
        graph_doc_ids = state.get("graph_doc_ids", [])

        # 1. Resolve pre-warmed seed vector results (already resolved by GraphNavigator)
        prewarm_seed = state.get("prewarm_seed_vec")
        exact_vec_results: List[Dict] = _resolve_future(prewarm_seed, default=[])
        if not exact_vec_results:
            if pool is not None:
                try:
                    exact_vec_results = pool.submit(self.retriever.vector_search, query, top_k=config.DENSE_TOP_K, filters=filters).result()
                except Exception as e:
                    logger.warning(f"Vector search failed: {e}")
                    exact_vec_results = []
            else:
                exact_vec_results = self.retriever.vector_search(query, top_k=config.DENSE_TOP_K, filters=filters)

        # 2. Resolve pre-warmed title/reference search
        prewarm_title = state.get("prewarm_title_ref")
        title_ref: List[Dict] = _resolve_future(prewarm_title, default=[])
        if not title_ref and prewarm_title is None:
            if pool is not None:
                try:
                    title_ref = pool.submit(self.retriever.title_reference_search, query, top_k=15).result()
                except Exception as e:
                    logger.warning(f"Title/reference search failed: {e}")
            else:
                try:
                    title_ref = self.retriever.title_reference_search(query, top_k=15)
                except Exception as e:
                    logger.warning(f"Title/reference search failed: {e}")

        # Ensure source tags on title_ref and exact_vec
        for r in title_ref:
            if "source" not in r:
                r["source"] = "title_reference"

        if title_ref:
            result_lists.append(title_ref)

        if exact_vec_results:
            for r in exact_vec_results:
                if "source" not in r:
                    r["source"] = "vector"
            result_lists.append(exact_vec_results)

        # 3. Launch graph search on expanded documents immediately on pool
        graph_future = None
        local_pool = None
        exec_pool = pool
        if exec_pool is None:
            local_pool = ThreadPoolExecutor(max_workers=8)
            exec_pool = local_pool

        try:
            if graph_doc_ids and config.GRAPH_EXPANSION_ENABLED:
                seed_doc_ids = {r["doc_id"] for r in exact_vec_results if r.get("doc_id")}
                graph_only   = [d for d in graph_doc_ids if d not in seed_doc_ids]
                if graph_only:
                    graph_future = exec_pool.submit(self.retriever.search_expanded_documents, query, graph_only, top_k=8)

            # 4. Resolve pre-warmed base BM25 search
            bm25_future = None
            bm25_res: List[Dict] = []
            prewarm_bm25 = state.get("prewarm_bm25")
            if prewarm_bm25 is not None:
                bm25_res = _resolve_future(prewarm_bm25, default=[])
            elif config.BM25_ENABLED:
                bm25_future = exec_pool.submit(self.retriever.bm25_search, query, top_k=config.BM25_TOP_K)

            # 5. Resolve pre-warmed query analysis plan
            plan = _resolve_future(state.get("prewarm_plan"), default=None)
            if plan is None:
                try:
                    plan = self.retriever.analyze_query(query)
                except Exception as e:
                    logger.warning(f"Query analysis failed: {e}")
                    plan = {"sub_queries": [], "hyde_document": "", "keywords": []}

            state["prewarm_plan"] = plan

            # 6. Collect all query variants: [query] + sub_queries + ([hyde] if hyde else [])
            sub_queries = [sq for sq in plan.get("sub_queries", []) if sq and isinstance(sq, str) and sq.strip()]
            hyde = plan.get("hyde_document", "")
            hyde_doc = [hyde.strip()] if (hyde and isinstance(hyde, str) and hyde.strip()) else []
            queries = [query] + sub_queries + hyde_doc
            state["queries"] = queries

            # 7. Batch-embed all query variants in a single OpenAI call (Contract B)
            if hasattr(self.retriever, "embed_queries_batch") and queries:
                try:
                    self.retriever.embed_queries_batch(queries)
                except Exception as e:
                    logger.warning(f"Batch embedding failed: {e}")

            # 8. Submit variant vector & BM25 searches on thread pool
            variant_futures = []
            bm25_variant_futures = []

            for sq in sub_queries:
                variant_futures.append(exec_pool.submit(self.retriever.vector_search, sq, top_k=config.DENSE_TOP_K, filters=filters))
                if config.BM25_ENABLED:
                    bm25_variant_futures.append(exec_pool.submit(self.retriever.bm25_search, sq, top_k=config.BM25_TOP_K))

            if hyde_doc:
                variant_futures.append(exec_pool.submit(self.retriever.vector_search, hyde_doc[0], top_k=config.DENSE_TOP_K, filters=filters))

            keywords = plan.get("keywords", [])
            if keywords and config.BM25_ENABLED and isinstance(keywords, list):
                kw_str = " ".join(str(k) for k in keywords)
                if kw_str.strip():
                    bm25_variant_futures.append(exec_pool.submit(self.retriever.bm25_search, kw_str, top_k=config.BM25_TOP_K))

            # Await exact BM25 if deferred
            if bm25_future:
                try:
                    bm25_res = bm25_future.result()
                except Exception as e:
                    logger.warning(f"Exact BM25 search failed: {e}")

            if bm25_res:
                for r in bm25_res:
                    if "source" not in r:
                        r["source"] = "bm25"
                result_lists.append(bm25_res)

            # Await variant vector searches
            for fut in as_completed(variant_futures):
                try:
                    res = fut.result()
                    if res:
                        for r in res:
                            if "source" not in r:
                                r["source"] = "vector"
                        result_lists.append(res)
                except Exception as e:
                    logger.warning(f"VectorSpecialist variant failed: {e}")

            # Await variant BM25 searches
            for fut in as_completed(bm25_variant_futures):
                try:
                    res = fut.result()
                    if res:
                        for r in res:
                            if "source" not in r:
                                r["source"] = "bm25"
                        result_lists.append(res)
                except Exception as e:
                    logger.warning(f"Variant BM25 search failed: {e}")

            # Await graph search
            if graph_future:
                try:
                    graph_res = graph_future.result()
                    if graph_res:
                        for r in graph_res:
                            if "source" not in r:
                                r["source"] = "graph_expansion"
                        result_lists.append(graph_res)
                except Exception as e:
                    logger.warning(f"Graph search failed: {e}")

        finally:
            if local_pool is not None:
                local_pool.shutdown(wait=True)

        # 9. Extract metadata filters from plan
        metadata_filters = plan.get("metadata_filters", {}) if plan else {}
        logger.info(f"VectorSpecialist metadata_filters extracted: {metadata_filters}")

        # 10. Populate state with multi-list results AND flat results (Contract C)
        state["vector_result_lists"] = [lst for lst in result_lists if lst]

        all_results: List[Dict] = []
        for lst in result_lists:
            for res in lst:
                if "source" not in res:
                    res["source"] = "vector"
                all_results.append(res)

        state["vector_results"] = all_results
        state["metadata_filters"] = metadata_filters
        logger.info(f"VectorSpecialist collected {len(all_results)} total chunks across {len(result_lists)} lists")

        total = sum(len(l) for l in result_lists)
        _trace(state, "VectorSpecialistAgent", t0,
               f"{len(queries)} queries x dense + BM25 -> {total} raw results ({len(result_lists)} lists)")
        logger.info(f"[VectorSpecialist] {len(result_lists)} result lists, {total} raw results")
        return state


# =========================================================================== #
# Agent 4 — Temporal Analyst                                                    #
# =========================================================================== #

class TemporalAnalystAgent:
    """
    Detects year / date references in the query.
    Sets a metadata filter so downstream searches are restricted to the
    relevant time window.  (Purely analytical — does not call any DB.)
    """

    YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")

    def run(self, state: AgentState) -> AgentState:
        t0    = time.time()
        query = state["query"]

        temporal_detected = False
        temporal_filter: Optional[Dict[str, Any]] = None

        # 1. Detect explicit year in query
        year_match = self.YEAR_RE.search(query)
        if year_match:
            year = int(year_match.group(1))
            temporal_filter   = {"year": year}
            temporal_detected = True
            _trace(state, "TemporalAnalystAgent", t0, f"year={year} detected")
            logger.info(f"[TemporalAnalyst] year filter: {year}")
        else:
            # 2. Detect temporal keywords
            q_lower = query.lower()
            matched = [kw for kw in config.AGENT_TEMPORAL_KEYWORDS if kw in q_lower]
            if matched:
                temporal_detected = True
                # Don't set a hard year filter — just log the intent
                _trace(state, "TemporalAnalystAgent", t0, f"temporal keywords: {matched}")
                logger.info(f"[TemporalAnalyst] temporal keywords found: {matched}")
            else:
                _trace(state, "TemporalAnalystAgent", t0, "no temporal signal")

        state["temporal_detected"] = temporal_detected
        state["temporal_filter"]   = temporal_filter
        return state


# =========================================================================== #
# Agent 5 — Fusion & Reranker                                                   #
# =========================================================================== #

class FusionRerankerAgent:
    """
    Applies Reciprocal Rank Fusion across all result lists collected by the
    Vector Specialist, then runs LLM reranking to produce the final top-K chunks.
    Also formats the context string for the Legal Reasoner.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def run(self, state: AgentState) -> AgentState:
        t0      = time.time()
        query   = state["query"]
        top_k   = state.get("top_k", config.FINAL_TOP_K)

        # ── 1. Multi-list RRF (Contract C) ────────────────────────────────── #
        # VectorSpecialistAgent supplies independent ranked lists in vector_result_lists
        result_lists = state.get("vector_result_lists")
        if not result_lists:
            # Fallback for backward compatibility or when vector_result_lists is not set
            raw = state.get("vector_results", [])
            result_lists = [raw] if raw else []

        # Filter out empty lists
        active_lists = [lst for lst in result_lists if lst]

        # ── 2. Track source tags to preserve provenance across RRF ───────── #
        graph_chunk_ids = set()
        title_ref_chunk_ids = set()
        for lst in active_lists:
            for chunk in lst:
                cid = chunk.get("chunk_id")
                src = chunk.get("source")
                if src == "graph_expansion":
                    graph_chunk_ids.add(cid)
                elif src == "title_reference":
                    title_ref_chunk_ids.add(cid)

        # Also inspect flat vector_results if present
        for chunk in state.get("vector_results", []):
            cid = chunk.get("chunk_id")
            src = chunk.get("source")
            if src == "graph_expansion":
                graph_chunk_ids.add(cid)
            elif src == "title_reference":
                title_ref_chunk_ids.add(cid)

        # ── 3. Execute Reciprocal Rank Fusion ─────────────────────────────── #
        if active_lists:
            fused = self.retriever.reciprocal_rank_fusion(active_lists)
        else:
            fused = []

        # ── 4. Restore high-value source tags on fused chunks ─────────────── #
        for chunk in fused:
            cid = chunk.get("chunk_id")
            if cid in graph_chunk_ids:
                chunk["source"] = "graph_expansion"
            elif cid in title_ref_chunk_ids and chunk.get("source") != "graph_expansion":
                chunk["source"] = "title_reference"

        # ── 5. Apply soft temporal boosting if detected ───────────────────── #
        temporal_filter = state.get("temporal_filter")
        if temporal_filter and "year" in temporal_filter:
            target_year = str(temporal_filter["year"])
            for chunk in fused:
                m = chunk.get("metadata", {}) or {}
                chunk_year = str(m.get("year", ""))
                if chunk_year == target_year:
                    chunk["rrf_score"] = chunk.get("rrf_score", 0.1) * 1.2
            logger.info(f"[FusionReranker] Applied soft boost for year {target_year}")

        # ── 6. Apply title/reference boosting (exact doc match) ───────────── #
        import re
        q_lower = query.lower()
        refs = re.findall(r"\b\d+/\d{4}\b", q_lower)
        refs += re.findall(r"n[°\s]*\s*(\d+)", q_lower)
        title_filters = state.get("metadata_filters", {}).get("doc_title")
        if isinstance(title_filters, str):
            title_filters = [title_filters]
        ref_boost_count = 0
        title_boost_count = 0
        for chunk in fused:
            m = chunk.get("metadata", {}) or {}
            doc_title = str(m.get("doc_title", "")).lower()
            reference = str(m.get("reference", "")).lower()
            title_ref = doc_title + " " + reference
            if refs and any(r in title_ref for r in refs):
                chunk["rrf_score"] = chunk.get("rrf_score", 0.1) * 4.0
                ref_boost_count += 1
            elif title_filters and any(str(tf).lower() in doc_title for tf in title_filters if tf):
                chunk["rrf_score"] = chunk.get("rrf_score", 0.1) * 3.5
                title_boost_count += 1
        if ref_boost_count or title_boost_count:
            logger.info(f"[FusionReranker] Applied boosts: ref={ref_boost_count}, title={title_boost_count}")

        # Re-sort after all boosts to ensure boosted chunks are in top-N for reranking
        fused.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)

        # ── 7. LLM reranking ─────────────────────────────────────────────── #
        reranked = self.retriever.rerank_with_llm(query, fused, top_k=top_k)

        # ── 8. Exact reference promotion ──────────────────────────────────── #
        if refs:
            matching = [r for r in reranked if r.get("source") == "title_reference"]
            matching += [
                r for r in reranked
                if r.get("source") != "title_reference"
                and any(ref in (str((r.get("metadata", {}) or {}).get("doc_title", "")) + " " +
                                str((r.get("metadata", {}) or {}).get("reference", ""))).lower() for ref in refs)
            ]
            matching = list({c["chunk_id"]: c for c in matching}.values())
            non_matching = [r for r in reranked if r["chunk_id"] not in {c["chunk_id"] for c in matching}]
            if matching:
                reranked = matching + non_matching
                reranked = reranked[:top_k]
                logger.info(f"[FusionReranker] Promoted {len(matching)} exact-match chunks to top")

        # ── 9. Guarantee minimum graph slots ──────────────────────────────── #
        MIN_GRAPH_SLOTS = 2
        graph_in_reranked = [r for r in reranked if r.get("source") == "graph_expansion"]
        if len(graph_in_reranked) < MIN_GRAPH_SLOTS:
            graph_candidates = [
                r for r in fused
                if r.get("source") == "graph_expansion"
                and r["chunk_id"] not in {x["chunk_id"] for x in reranked}
            ]
            needed = MIN_GRAPH_SLOTS - len(graph_in_reranked)
            to_add = graph_candidates[:needed]
            if to_add:
                for g in to_add:
                    # Find lowest-scoring non-graph result to replace, preserving exact reference matches
                    non_graph = [
                        (i, r) for i, r in enumerate(reranked)
                        if r.get("source") != "graph_expansion" and r.get("source") != "title_reference"
                    ]
                    if not non_graph:
                        non_graph = [
                            (i, r) for i, r in enumerate(reranked)
                            if r.get("source") != "graph_expansion"
                        ]
                    if non_graph:
                        replace_idx = non_graph[-1][0]
                        reranked[replace_idx] = g
                logger.info(
                    f"[FusionReranker] Promoted {len(to_add)} graph results "
                    f"into top-{top_k} (min_graph_slots={MIN_GRAPH_SLOTS})"
                )

        # ── 10. Enforce cross-reference resolution in RAM ─────────────────── #
        # Positioned after graph slot guarantee so cross-ref chunks are never displaced
        try:
            reranked = self.retriever._resolve_cross_references(reranked, max_refs=4)
        except Exception as e:
            logger.warning(f"[FusionReranker] Cross-reference resolution failed (non-fatal): {e}")

        state["fused_results"] = reranked

        # ── 11. Build context string ──────────────────────────────────────── #
        context_parts = []
        for i, chunk in enumerate(reranked, 1):
            m = chunk.get("metadata", {}) or {}
            header_parts = [m.get("doc_title", "Sans titre")]
            if m.get("type"):        header_parts.append(f"type={m['type']}")
            if m.get("reference"):   header_parts.append(f"ref={m['reference']}")
            if m.get("year"):        header_parts.append(f"annee={m['year']}")
            if m.get("institution"): header_parts.append(f"inst={m['institution']}")
            if m.get("date"):        header_parts.append(f"date={str(m['date'])[:10]}")
            context_parts.append(
                f"[Document {i}] {' | '.join(header_parts)}\n"
                f"{chunk.get('content', '')}\n"
                f"Source: {m.get('source_url', 'N/A')}"
            )

        state["context"] = "\n\n---\n\n".join(context_parts)
        num_fused = len(fused)
        _trace(state, "FusionRerankerAgent", t0,
               f"RRF {len(active_lists)} lists ({num_fused} unique) -> reranked -> top {len(reranked)}")
        logger.info(f"[FusionReranker] {len(active_lists)} lists ({num_fused} unique) -> top {len(reranked)}")
        return state


# =========================================================================== #
# Agent 6 — Legal Reasoner + Fact-Checker                                       #
# =========================================================================== #

class LegalReasonerAgent:
    """
    Calls the LLMGenerator to produce a chain-of-thought legal answer,
    then runs a lightweight fact-check to ensure every affirmation in the
    answer maps to at least one retrieved chunk.
    Includes Self-RAG retry logic for low-confidence answers.
    """

    # Matches common legal locutions used as citation anchors
    CITATION_RE = re.compile(
        r"(article\s+\d+|loi\s+[A-Z0-9/\-]+|décret\s+[A-Z0-9/\-]+"
        r"|code\s+[A-Za-z]+|arrêté|ordonnance|jugement|arrêt)",
        re.IGNORECASE,
    )

    def __init__(self, generator: LLMGenerator, retriever: HybridRetriever):
        self.generator = generator
        self.retriever = retriever

    # ------------------------------------------------------------------ #
    def _parse_confidence(self, answer: str) -> float:
        m = re.search(r"\*\*CONFIANCE:\*\*\s*([0-9.]+)", answer)
        if m:
            try:
                return min(1.0, max(0.0, float(m.group(1))))
            except ValueError:
                pass
        return 1.0

    # ------------------------------------------------------------------ #
    def _fact_check(self, answer: str, chunks: List[Dict]) -> tuple[bool, List[str]]:
        """
        Lightweight fact-checker: extracts citation anchors from the answer
        and checks that each appears in at least one retrieved chunk.
        Returns (ok, issues).
        """
        issues: List[str] = []
        citations_in_answer = self.CITATION_RE.findall(answer)

        if not citations_in_answer:
            return True, []

        # Build a corpus of all retrieved text (lower-cased)
        corpus = " ".join(c["content"].lower() for c in chunks)

        unmatched = [
            cit for cit in citations_in_answer
            if cit.lower() not in corpus
        ]

        ratio_matched = 1.0 - (len(unmatched) / len(citations_in_answer))
        ok = ratio_matched >= config.AGENT_FACTCHECK_MIN_RATIO

        if unmatched:
            issues.append(
                f"Citations not found in retrieved context: {unmatched[:5]}"
            )
        return ok, issues

    # ------------------------------------------------------------------ #
    def run(self, state: AgentState) -> AgentState:
        t0      = time.time()
        query   = state["query"]
        chunks  = state.get("fused_results", [])
        context = state.get("context", "")
        top_k   = state.get("top_k", config.FINAL_TOP_K)
        filters = state.get("filters")
        use_graph = state.get("use_graph", True)

        # Self-RAG loop
        llm_result: Dict = {}
        for attempt in range(1, config.AGENT_MAX_ITERATIONS + 1):
            try:
                llm_result = self.generator.generate_response(query, context, chunks=chunks)
            except Exception as e:
                logger.error(f"LegalReasonerAgent generation error: {e}")
                llm_result = {
                    "answer":  f"Erreur de génération: {e}",
                    "model":   config.LLM_MODEL,
                    "usage":   {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "sources": [],
                }
                break

            confidence = self._parse_confidence(llm_result["answer"])
            logger.info(f"[LegalReasoner] attempt={attempt} confidence={confidence:.2f}")

            if confidence >= config.SELF_RAG_CONFIDENCE_THRESHOLD:
                break

            if attempt < config.AGENT_MAX_ITERATIONS:
                # Expand query and re-retrieve
                expanded_q = f"{query} — précisez avec les références légales exactes."
                try:
                    new_chunks, new_context = self.retriever.retrieve_with_context(
                        query=expanded_q, top_k=top_k, filters=filters, use_graph=use_graph
                    )
                    chunks, context = new_chunks, new_context
                except Exception as e:
                    logger.warning(f"Self-RAG re-retrieval failed: {e}")
                    break

        answer     = llm_result.get("answer", "")
        confidence = self._parse_confidence(answer)
        fc_ok, fc_issues = self._fact_check(answer, chunks)

        state["llm_result"]       = llm_result
        state["answer"]           = answer
        state["confidence"]       = confidence
        state["fact_check_ok"]    = fc_ok
        state["fact_check_issues"] = fc_issues
        state["fused_results"]    = chunks    # may be updated by Self-RAG
        state["context"]          = context

        _trace(state, "LegalReasonerAgent", t0,
               f"confidence={confidence:.2f} fact_check={'OK' if fc_ok else 'WARN'}")
        return state


# =========================================================================== #
# Simple fast-path node (for non-complex queries)                               #
# =========================================================================== #

class SimpleFastPath:
    """
    Handles simple queries: runs a single vector search (no graph expansion,
    no multi-hop) and generates a response directly — minimising latency.
    """

    def __init__(self, retriever: HybridRetriever, generator: LLMGenerator):
        self.retriever = retriever
        self.generator = generator

    def run(self, state: AgentState) -> AgentState:
        t0      = time.time()
        query   = state["query"]
        top_k   = state.get("top_k", config.FINAL_TOP_K)
        filters = state.get("filters")

        try:
            chunks, context = self.retriever.retrieve_with_context(
                query=query, top_k=top_k, filters=filters, use_graph=False
            )
            llm_result = self.generator.generate_response(query, context, chunks=chunks)
        except Exception as e:
            logger.error(f"SimpleFastPath error: {e}")
            chunks     = []
            context    = ""
            llm_result = {
                "answer":  f"Erreur: {e}",
                "model":   config.LLM_MODEL,
                "usage":   {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "sources": [],
            }

        state["fused_results"] = chunks
        state["context"]       = context
        state["llm_result"]    = llm_result
        state["answer"]        = llm_result.get("answer", "")
        state["confidence"]    = 1.0
        state["fact_check_ok"] = True
        state["fact_check_issues"] = []
        _trace(state, "SimpleFastPath", t0, "fast path (no graph, no multi-hop)")
        return state


# =========================================================================== #
# Build LangGraph StateGraph                                                    #
# =========================================================================== #

def _build_graph(
    router:    RouterAgent,
    navigator: GraphNavigatorAgent,
    vector:    VectorSpecialistAgent,
    temporal:  TemporalAnalystAgent,
    fusion:    FusionRerankerAgent,
    reasoner:  LegalReasonerAgent,
    fast_path: SimpleFastPath,
):
    """Wire all agents into a LangGraph StateGraph and compile it."""

    if not _LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph is not installed. Run: pip install langgraph>=0.1.0"
        )

    # ---------------------------------------------------------------------- #
    # Node wrappers (LangGraph nodes must be callables: state → state)        #
    # ---------------------------------------------------------------------- #

    def _router(state: AgentState) -> AgentState:
        return router.run(state)

    def _fast_path(state: AgentState) -> AgentState:
        return fast_path.run(state)

    def _graph_nav(state: AgentState) -> AgentState:
        return navigator.run(state)

    def _vector(state: AgentState) -> AgentState:
        return vector.run(state)

    def _temporal(state: AgentState) -> AgentState:
        return temporal.run(state)

    def _fusion(state: AgentState) -> AgentState:
        return fusion.run(state)

    def _reasoner(state: AgentState) -> AgentState:
        return reasoner.run(state)

    # ---------------------------------------------------------------------- #
    # Routing condition                                                         #
    # ---------------------------------------------------------------------- #

    def _route_after_router(state: AgentState) -> str:
        """Branch: simple → fast_path_node, otherwise → graph_navigator_node."""
        score      = state.get("complexity_score", 0.5)
        complexity = state.get("complexity", "complex")
        if complexity == "simple" and score < config.AGENT_SIMPLE_THRESHOLD:
            return "fast_path_node"
        return "graph_navigator_node"

    # ---------------------------------------------------------------------- #
    # Build graph                                                              #
    # ---------------------------------------------------------------------- #

    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("router_node",          _router)
    builder.add_node("fast_path_node",       _fast_path)
    builder.add_node("graph_navigator_node", _graph_nav)
    builder.add_node("vector_specialist_node", _vector)
    builder.add_node("temporal_analyst_node", _temporal)
    builder.add_node("fusion_reranker_node", _fusion)
    builder.add_node("legal_reasoner_node",  _reasoner)

    # Entry point
    builder.set_entry_point("router_node")

    # Conditional edge after router
    builder.add_conditional_edges(
        "router_node",
        _route_after_router,
        {
            "fast_path_node":       "fast_path_node",
            "graph_navigator_node": "graph_navigator_node",
        },
    )

    # Fast path goes straight to END
    builder.add_edge("fast_path_node", END)

    # Complex path: navigator → vector → temporal → fusion → reasoner → END
    builder.add_edge("graph_navigator_node",    "vector_specialist_node")
    builder.add_edge("vector_specialist_node",  "temporal_analyst_node")
    builder.add_edge("temporal_analyst_node",   "fusion_reranker_node")
    builder.add_edge("fusion_reranker_node",    "legal_reasoner_node")
    builder.add_edge("legal_reasoner_node",     END)

    return builder.compile()


# =========================================================================== #
# Public AgenticPipeline                                                        #
# =========================================================================== #

class AgenticPipeline:
    """
    Drop-in replacement for GraphRAGPipeline when LANGGRAPH_ENABLED = True.

    Usage:
        pipeline = AgenticPipeline()
        result   = pipeline.query("Obligations d'une SARL en Guinée?")
    """

    # ---------------------------------------------------------------------- #
    def __init__(self):
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError(
                "langgraph is not installed. "
                "Run: pip install langgraph>=0.1.0 langchain-openai>=0.1.0"
            )

        logger.info("Initialising AgenticPipeline (LangGraph)…")

        # Shared backend components
        self.client    = OpenAI(api_key=config.OPENAI_API_KEY, timeout=30.0)
        self.retriever = HybridRetriever()
        self.generator = LLMGenerator()
        self.formatter = ResponseFormatter()
        self.cache     = CacheManager()

        # Shared persistent thread pool (12 workers for parallel agent orchestration)
        self._pool = ThreadPoolExecutor(max_workers=12, thread_name_prefix="teamworker")

        # Instantiate agents
        self._router    = RouterAgent(self.client, retriever=self.retriever, pool=self._pool)
        self._navigator = GraphNavigatorAgent(self.retriever, pool=self._pool)
        self._vector    = VectorSpecialistAgent(self.retriever, pool=self._pool)
        self._temporal  = TemporalAnalystAgent()
        self._fusion    = FusionRerankerAgent(self.retriever)
        self._reasoner  = LegalReasonerAgent(self.generator, self.retriever)
        self._fast      = SimpleFastPath(self.retriever, self.generator)

        # Compile LangGraph
        self._graph = _build_graph(
            self._router,
            self._navigator,
            self._vector,
            self._temporal,
            self._fusion,
            self._reasoner,
            self._fast,
        )

        logger.info("AgenticPipeline ready")

    # ---------------------------------------------------------------------- #
    def query(
        self,
        question:  str,
        top_k:     int  = None,
        filters:   Dict = None,
        use_cache: bool = True,
        use_graph: bool = True,
    ) -> Dict:
        """
        Run a query through the multi-agent pipeline.

        Returns a dict compatible with GraphRAGPipeline.query() output,
        with an extra 'agent_trace' key listing per-agent latencies.
        """
        start_time = time.time()
        top_k = top_k or config.FINAL_TOP_K

        # ── Cache check ──────────────────────────────────────────────────── #
        if use_cache:
            cached = self.cache.get_response(question, "", filters=filters)
            if cached:
                return {
                    **cached,
                    "cached":     True,
                    "latency_ms": (time.time() - start_time) * 1000,
                }

        # ── Build initial state ──────────────────────────────────────────── #
        initial_state: AgentState = {
            "query":        question,
            "top_k":        top_k,
            "filters":      filters,
            "use_cache":    use_cache,
            "use_graph":    use_graph,
            "agent_trace":  [],
            "_pool":        self._pool,
        }

        # ── Run LangGraph ────────────────────────────────────────────────── #
        try:
            final_state: AgentState = self._graph.invoke(
                initial_state,
                config={"recursion_limit": config.AGENT_MAX_ITERATIONS + 10},
            )
        except Exception as e:
            logger.error(f"AgenticPipeline graph execution error: {e}", exc_info=True)
            raise

        # ── Extract results from final state ─────────────────────────────── #
        chunks     = final_state.get("fused_results", [])
        llm_result = final_state.get("llm_result", {})
        answer     = final_state.get("answer", "")
        context    = final_state.get("context", "")

        # Format + validate
        sections = self.formatter.extract_sections(answer)
        is_valid, issues = self.formatter.validate_response(answer, context)
        cost = self.generator.compute_cost(llm_result.get("usage", {}))

        total_ms = (time.time() - start_time) * 1000

        # Compute breakdown across retrieval vs generation stages
        retrieval_ms = 0.0
        generation_ms = 0.0
        for step in final_state.get("agent_trace", []):
            agent_name = step.get("agent", "")
            step_lat = float(step.get("latency_ms", 0.0))
            if agent_name in ("LegalReasonerAgent", "SimpleFastPath"):
                generation_ms += step_lat
            else:
                retrieval_ms += step_lat

        response = {
            "query":   question,
            "answer":  answer,
            "sections": sections,
            "sources": self._dedup_sources(chunks),
            "validation": {
                "is_valid": is_valid,
                "issues":   issues,
            },
            "metadata": {
                "model":    llm_result.get("model", config.LLM_MODEL),
                "usage":    llm_result.get("usage", {}),
                "cost_usd": cost,
                "latency_ms": {
                    "total": round(total_ms, 2),
                    "retrieval": round(retrieval_ms, 2),
                    "generation": round(generation_ms, 2),
                },
                "num_chunks":    len(chunks),
                "graph_enabled": use_graph,
                "filters_used":  filters or {},
                "complexity":    final_state.get("complexity", "unknown"),
                "complexity_score": final_state.get("complexity_score", 0.5),
                "confidence":    final_state.get("confidence", 1.0),
                "fact_check_ok": final_state.get("fact_check_ok", True),
                "fact_check_issues": final_state.get("fact_check_issues", []),
            },
            "agent_trace": final_state.get("agent_trace", []),
            "cached": False,
        }

        # Cache
        if use_cache:
            self.cache.set_response(question, "", response, filters=filters)

        return response

    # ---------------------------------------------------------------------- #
    @staticmethod
    def _meta(chunk: dict, key: str, default=''):
        """Safe metadata accessor — flat or nested."""
        if key in chunk:
            return chunk[key]
        return chunk.get('metadata', {}).get(key, default)

    def _dedup_sources(self, chunks: list) -> list:
        """Deduplicate source cards by doc_id (one card per document, like classic pipeline)."""
        seen_doc_ids: set = set()
        out: list = []
        for c in chunks:
            doc_id = c.get('doc_id') or c.get('id', '') or ''
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            out.append({
                'doc_id':      doc_id,
                'title':       (self._meta(c, 'doc_title')
                                or self._meta(c, 'title')),
                'type':        self._meta(c, 'type'),
                'reference':   self._meta(c, 'reference'),
                'year':        self._meta(c, 'year'),
                'institution': self._meta(c, 'institution'),
                'date':        self._meta(c, 'date'),
                'modified':    self._meta(c, 'modified'),
                'source_url':  self._meta(c, 'source_url'),
                'score':       c.get('rrf_score', c.get('score', 0)),
                'source':      c.get('source', 'vector'),
                'content_preview': (
                    c.get('content', '')[:200] + '...'
                    if len(c.get('content', '')) > 200
                    else c.get('content', '')
                ),
            })
        return out

    # ---------------------------------------------------------------------- #
    def close(self):
        """Close all underlying connections and worker pool."""
        try:
            if hasattr(self, "_pool") and self._pool is not None:
                self._pool.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"Error shutting down thread pool: {e}")
        self.retriever.close()
        if hasattr(self.retriever, "graph_builder") and self.retriever.graph_builder:
            self.retriever.graph_builder.close()


# =========================================================================== #
# CLI smoke-test                                                                #
# =========================================================================== #

def main():
    """Quick smoke-test: run one query through the agentic pipeline."""
    import sys
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
    )

    if not _LANGGRAPH_AVAILABLE:
        print("[-] langgraph is not installed.")
        print("    Run: pip install langgraph>=0.1.0 langchain-openai>=0.1.0")
        sys.exit(1)

    pipeline = AgenticPipeline()
    query    = "Quel est le taux d'imposition sur les plus-values de cession d'actions selon le CGI 2024 ?"

    print(f"\n{'='*70}")
    print("[AGENT TEST] AgenticPipeline -- LangGraph Multi-Agent")
    print(f"{'='*70}")
    print(f"Query: {query}\n")


    try:
        result = pipeline.query(query, use_cache=False)

        print(result["answer"])
        print(f"\n{'-'*70}")
        print("Agent Trace:")
        for step in result.get("agent_trace", []):
            print(f"   {step['agent']:35s}  {step['latency_ms']:7.1f} ms  {step['notes']}")

        meta = result["metadata"]
        print(f"\n   Complexity  : {meta['complexity']} (score={meta['complexity_score']:.2f})")
        print(f"   Confidence  : {meta['confidence']:.2f}")
        print(f"   Fact-check  : {'OK' if meta['fact_check_ok'] else 'WARN'}")
        print(f"   Sources     : {meta['num_chunks']}")
        print(f"   Total latency: {meta['latency_ms']['total']:.0f} ms")
        print(f"   Cost        : ${meta['cost_usd']:.6f}")


    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
