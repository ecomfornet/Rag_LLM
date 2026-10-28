# Project: GraphRAG VPS Optimization

## Architecture
The system is a high-speed, 6-agent legal document retrieval pipeline using LangGraph, Qdrant vector database, and Neo4j graph database.

### Core Modules & Flow
1. **`graph_builder.py`**: Neo4j graph traversal and document relationship graph.
   - Configures connection pooling (`max_connection_pool_size=50`, `connection_acquisition_timeout=15.0`).
   - Executes batched UNWIND Cypher traversal for multi-hop citation expansion (`get_related_documents_batch`).
   - Retains backward compatibility with `get_related_documents`.
2. **`retrieval_4.py`**: Hybrid retrieval engine (Dense Vector + BM25 + Qdrant + Neo4j Graph).
   - In-RAM chunk mapping (`_doc_chunks_map: Dict[str, List[Chunk]]`) and document metadata map (`_doc_metadata_map`) loaded from `bm25_cache.pkl` at initialization.
   - O(1) in-RAM cross-reference resolution (`_resolve_cross_references`) and in-RAM title search (`title_reference_search`), eliminating unindexed Qdrant HTTP scroll loops.
   - Thread-safe LRU query embedding cache (`_embedding_cache`) with `embed_queries_batch` consolidating query variants into a single OpenAI embedding call.
   - Idempotent payload index creation on Qdrant key metadata fields (`doc_id`, `metadata.year`, etc.).
   - Batched graph seed expansion calling `graph_builder.get_related_documents_batch`.
3. **`generation_5.py`**: Prompt engineering and legal answer synthesis.
   - Explicit confidence rating format in `build_legal_prompt()` (`**CONFIANCE:** [0.0-1.0]`).
   - Robust regex parsing in `ResponseFormatter.extract_sections()` for `**FONDEMENT JURIDIQUE:**` and `**CONFIANCE:**`.
4. **`agents_langgraph.py`**: 6-Agent LangGraph Orchestration pipeline.
   - Shared persistent `ThreadPoolExecutor(max_workers=12)` in `AgenticPipeline.__init__` with clean shutdown in `close()`.
   - Concurrent pre-warming in `RouterAgent`: seed vector search, title reference search, BM25 search dispatched to thread pool while router LLM classifies the query.
   - Seed reuse in `GraphNavigatorAgent`: consumes `prewarm_seed_vec` and submits `analyze_query` asynchronously.
   - Batch embedding & pre-warm reuse in `VectorSpecialistAgent`: awaits pre-warmed searches and batch-embeds all query variants with `embed_queries_batch`.
   - Multi-list RRF & source tag preservation in `FusionRerankerAgent`: receives `vector_result_lists`, performs multi-list RRF, restores `graph_expansion` source tags, and enforces cross-reference resolution.
   - `SimpleFastPath` optimized to consume pre-warmed seeds for low-complexity queries.
5. **E2E Test Suites**:
   - `test_system.py`: Validates all subsystems (Neo4j, Qdrant, Hybrid Retrieval, Graph Expansion, Agents).
   - `run_complex_query.py`: Tests complex legal reasoning and exact target reference retrieval (arrêt CCJA N°173/2021) with full 6-agent trace.
   - `test_regression_quality.py`: Evaluates answer completeness, source citation quality, and wall-clock latency reduction.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Neo4j Connection Pool Tuning | Set `max_connection_pool_size=50` and `connection_acquisition_timeout=15.0` on driver | M1 | ORIGINAL_REQUEST §R3 |
| 2 | Batched UNWIND Cypher Traversal | Implement `get_related_documents_batch` using single UNWIND Cypher query | M1 | ORIGINAL_REQUEST §R3, Contract A |
| 3 | Graph Traversal Backward Compatibility | Retain `get_related_documents` for existing diagnostic callers | M1 | ORIGINAL_REQUEST §R3 |
| 4 | Thread-Safe LRU Embedding Cache | Implement thread-safe LRU cache for query embeddings in `HybridRetriever` | M2 | ORIGINAL_REQUEST §R2 |
| 5 | Batched Query Embeddings | Implement `embed_queries_batch` consolidating missing embeddings into 1 OpenAI call | M2 | ORIGINAL_REQUEST §R2, Contract B |
| 6 | In-RAM Chunk & Metadata Mapping | Build `_doc_chunks_map` and `_doc_metadata_map` from in-memory BM25 cache | M2 | ORIGINAL_REQUEST §R2 |
| 7 | Zero-Scroll Cross-Reference Resolution | Resolve article references via in-RAM chunk map without Qdrant HTTP scroll loops | M2 | ORIGINAL_REQUEST §R2 |
| 8 | Zero-Scroll Title Reference Search | Scan document titles/references in RAM without Qdrant HTTP scroll loops | M2 | ORIGINAL_REQUEST §R2 |
| 9 | Qdrant Payload Indexing | Create idempotent payload indexes on `doc_id`, `metadata.year`, `metadata.type`, etc. | M2 | ORIGINAL_REQUEST §R2 |
| 10 | Single-Call Graph Expansion in Retriever | Call `get_related_documents_batch` from `HybridRetriever.graph_expansion` | M2 | ORIGINAL_REQUEST §R2, Contract A |
| 11 | Prompt Confidence Format | Append `**CONFIANCE:** [Score entre 0.0 et 1.0]` to `build_legal_prompt` | M3 | ORIGINAL_REQUEST §R4, Contract D |
| 12 | Regex Parsing for Legal Basis & Confidence | Update `basis_match` and `confidence_match` in `ResponseFormatter.extract_sections` | M3 | ORIGINAL_REQUEST §R4, Contract D |
| 13 | Persistent Thread Pool Lifecycle | Add `ThreadPoolExecutor(max_workers=12)` to `AgenticPipeline` with clean `close()` | M4 | ORIGINAL_REQUEST §R1 |
| 14 | Extended AgentState | Add `prewarm_seed_vec`, `prewarm_title_ref`, `prewarm_bm25`, `prewarm_plan`, `vector_result_lists` to `AgentState` | M4 | ORIGINAL_REQUEST §R1, Contract C |
| 15 | Router Concurrent Pre-Warming | In `RouterAgent`, pre-warm seed vector, title ref, and BM25 searches on thread pool during router LLM call | M4 | ORIGINAL_REQUEST §R1 |
| 16 | Graph Navigator Asynchronous Query Analysis | In `GraphNavigatorAgent`, run `analyze_query` on thread pool and reuse `prewarm_seed_vec` | M4 | ORIGINAL_REQUEST §R1 |
| 17 | Vector Specialist Pre-Warm Reuse & Batch Embed | In `VectorSpecialistAgent`, reuse pre-warmed searches and batch-embed variants with `embed_queries_batch` | M4 | ORIGINAL_REQUEST §R1 |
| 18 | Fusion Reranker Multi-List RRF & Provenance | Pass `vector_result_lists` to RRF, preserve `graph_expansion` source tags, enforce cross-refs | M4 | ORIGINAL_REQUEST §R1 |
| 19 | Fast Path Pre-Warm Consumption | Allow `SimpleFastPath` to reuse pre-warmed seed vector search results | M4 | ORIGINAL_REQUEST §Non-Negotiable Guardrails |
| 20 | System Diagnostic Verification | Pass all diagnostics in `test_system.py` | M5 | ORIGINAL_REQUEST §Acceptance Criteria |
| 21 | Complex Query Trace & Target Retrieval | Pass `run_complex_query.py` retrieving CCJA N°173/2021 with full 6-agent trace | M5 | ORIGINAL_REQUEST §Acceptance Criteria |
| 22 | Regression Quality & Latency Benchmark | Pass `test_regression_quality.py` with answer length >= 80 and reduced latency | M5 | ORIGINAL_REQUEST §Acceptance Criteria |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Batched Cypher Traversal (`graph_builder.py`) | Features 1, 2, 3: Connection pool, `get_related_documents_batch`, backward compatibility | None | DONE |
| M2 | In-RAM Lookups & Batched Embeddings (`retrieval_4.py`) | Features 4, 5, 6, 7, 8, 9, 10: In-RAM maps, embedding cache, `embed_queries_batch`, payload indexes, batch graph expansion | M1 | DONE |
| M3 | Prompt Confidence & Section Parsing (`generation_5.py`) | Features 11, 12: `build_legal_prompt` confidence specification and regex parsing in `ResponseFormatter` | None | DONE |
| M4 | Parallel Agent Orchestration (`agents_langgraph.py`) | Features 13, 14, 15, 16, 17, 18, 19: Persistent thread pool, Router pre-warming, reuse in Graph/Vector agents, multi-list RRF | M1, M2, M3 | IN_PROGRESS |
| M5 | System Verification & Regression Quality (E2E Track) | Features 20, 21, 22: Code compilation, `test_system.py`, `run_complex_query.py`, `test_regression_quality.py`, latency audit | M1, M2, M3, M4 | PLANNED |

---

## Interface Contracts

### Contract A (`graph_builder.py` -> `retrieval_4.py`)
```python
def get_related_documents_batch(
    self,
    doc_ids: List[str],
    relationship_types: Optional[List[str]] = None,
    max_depth: int = 2
) -> Dict[str, int]:
    """Returns a dict mapping doc_id -> minimum_hop_distance using a single Neo4j UNWIND Cypher query."""
```
- Input: List of seed document IDs (`doc_ids`), optional relationship types (defaults to `CITES`), and max hop depth (1 or 2).
- Output: Dictionary mapping related document ID (`str`) to minimum hop distance (`int`).
- Behavior: Seeds with distance 0 may be omitted or included, but caller (`retrieval_4.py`) initializes seeds at distance 0 and preserves `min(existing, new)`.

### Contract B (`retrieval_4.py` -> `agents_langgraph.py`)
```python
def embed_queries_batch(
    self,
    queries: List[str]
) -> Dict[str, List[float]]:
    """Batches all uncached query strings in a single OpenAI embedding call (embeddings.create(input=missing)) and returns a dict mapping query -> embedding."""
```
- Input: List of query strings (e.g. `[query] + sub_queries + [hyde_document]`).
- Output: Dictionary mapping query text -> embedding vector (`List[float]`).
- Thread Safety: Employs `ThreadSafeLRUCache` to update cache under lock.

### Contract C (`agents_langgraph.py` internal state)
`AgentState` TypedDict extensions:
- `prewarm_seed_vec`: `Optional[concurrent.futures.Future]` or resolved `List[Dict]` for seed vector search.
- `prewarm_title_ref`: `Optional[concurrent.futures.Future]` or resolved `List[Dict]` for title reference search.
- `prewarm_bm25`: `Optional[concurrent.futures.Future]` or resolved `List[Dict]` for BM25 search.
- `prewarm_plan`: `Optional[concurrent.futures.Future]` or resolved `Dict` for `analyze_query`.
- `vector_result_lists`: `List[List[Dict]]` carrying independent ranked lists into `FusionRerankerAgent` for multi-list RRF.
- `vector_results`: `List[Dict]` flat list for backward compatibility with downstream agents.

### Contract D (`generation_5.py` -> `agents_langgraph.py`)
- `LLMGenerator.build_legal_prompt()` appends:
  ```
  **CONFIANCE:** [Score entre 0.0 et 1.0 évaluant la certitude juridique de la réponse basée uniquement sur les sources fournies]
  ```
- `ResponseFormatter.extract_sections()`:
  - `basis_match = re.search(r'\*\*FONDEMENT JURIDIQUE:\*\*\s*(.*?)(?=\s*\*\*CONFIANCE:|\Z)', response, re.DOTALL)`
  - `confidence_match = re.search(r'\*\*CONFIANCE:\*\*\s*([\d.]+)', response)`

---

## Code Layout & Write Boundaries
To ensure zero write conflicts:
- Milestone 1 exclusively modifies: `c:\Users\mouni\Desktop\graphrag\graph_builder.py`
- Milestone 2 exclusively modifies: `c:\Users\mouni\Desktop\graphrag\retrieval_4.py`
- Milestone 3 exclusively modifies: `c:\Users\mouni\Desktop\graphrag\generation_5.py`
- Milestone 4 exclusively modifies: `c:\Users\mouni\Desktop\graphrag\agents_langgraph.py`
- Milestone 5 verifies all modules and runs tests without modifying production source code unless bug fixes are mandated.
