"""
Retrieval Module - Hybrid Graph + Vector Search
Combines vector search with graph traversal and reranking
"""
import logging
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
import openai.resources.chat
import openai.resources.embeddings

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

import config
from graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining vector search and graph traversal"""
    
    def __init__(self):
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=30.0,   # prevent infinite hang on network issues
        )
        
        # Initialize Qdrant client
        # FIX 1: Added timeout=30 to prevent ReadTimeout errors on slow/batched queries
        self.qdrant = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            https=config.QDRANT_USE_HTTPS,
            api_key=config.QDRANT_API_KEY,
            timeout=30  # <-- FIX 1: was missing, caused ReadTimeout on graph expansion queries
        )
        
        # Initialize graph builder
        self.graph_builder = GraphBuilder()

        # BM25 index: built once at init from all Qdrant payloads
        self._bm25: Optional["BM25Okapi"] = None
        self._bm25_chunks: List[Dict] = []  # parallel list of chunk dicts
        if config.BM25_ENABLED and _BM25_AVAILABLE:
            self._build_bm25_index()

        self._embedding_cache = {}
        # In-process caches to avoid repeated expensive DB scans
        # ref_key -> list[chunks]
        self._title_ref_cache: Dict[str, List[Dict]] = {}

        logger.info("HybridRetriever initialized")
    
    # ------------------------------------------------------------------
    # BM25 STOPWORDS (French legal text)
    # Common function words that add noise to legal BM25 scoring.
    # These are filtered out during tokenization.
    # ------------------------------------------------------------------
    _FR_STOPWORDS = frozenset([
        "le", "la", "les", "un", "une", "des", "du", "de", "d",
        "et", "en", "au", "aux", "par", "sur", "sous", "pour",
        "dans", "avec", "sans", "est", "sont", "être", "a", "ont",
        "se", "si", "ne", "pas", "que", "qui", "dont", "où",
        "il", "elle", "ils", "elles", "on", "nous", "vous",
        "ce", "cette", "ces", "tout", "tous", "toute", "toutes",
        "leur", "leurs", "même", "plus", "moins", "très",
        "l", "s", "j", "m", "n", "c", "y",
        "à", "ou", "ni", "car", "mais", "or", "donc",
    ])

    @classmethod
    def _tokenize_fr(cls, text: str) -> list:
        """Tokenize French legal text, removing stopwords and short tokens.

        Handles alphanumeric references (173/2021) and apostrophe contractions.
        """
        import re
        # Match alphanumeric sequences, slashes, and the degree symbol (\u00b0)
        # Correctly captures references like 173/2021 or N\u00b0 173/2021
        tokens = re.findall(r"[a-z0-9/\u00b0\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb\u00ee\u00ef\u00f4\u00f9\u00fb\u00fc\u00e7]+", text.lower())
        return [
            t for t in tokens
            if t not in cls._FR_STOPWORDS and len(t) > 2
        ]



    # ------------------------------------------------------------------
    # BM25 INDEX
    # ------------------------------------------------------------------

    def _build_bm25_index(self):
        """
        Fetch ALL point payloads from Qdrant and build a BM25Okapi index.
        Uses a pickle cache (data/bm25_cache.pkl) so repeated starts avoid
        the ~10 s Qdrant scroll.  Cache expires after 24 hours.
        """
        import pickle
        from pathlib import Path

        cache_path = Path(config.DATA_DIR) / "bm25_cache.pkl"
        cache_max_age = 86400  # 24 hours

        # --- Try loading from cache first ---
        if cache_path.exists():
            try:
                age = time.time() - cache_path.stat().st_mtime
                if age < cache_max_age:
                    with open(cache_path, "rb") as f:
                        cached = pickle.load(f)
                    self._bm25_chunks = cached["chunks"]
                    self._bm25 = cached["bm25"]
                    logger.info(
                        f"BM25 index loaded from cache "
                        f"({len(self._bm25_chunks)} chunks, age={age:.0f}s)"
                    )
                    return
                else:
                    logger.info("BM25 cache expired, rebuilding...")
            except Exception as e:
                logger.warning(f"BM25 cache load failed, rebuilding: {e}")

        # --- Build from Qdrant ---
        try:
            logger.info("Building BM25 index from Qdrant payloads...")
            all_points = []
            offset = None
            while True:
                batch, next_offset = self.qdrant.scroll(
                    collection_name=config.QDRANT_COLLECTION,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                all_points.extend(batch)
                if next_offset is None:
                    break
                offset = next_offset

            self._bm25_chunks = [
                {
                    'chunk_id':    p.payload['chunk_id'],
                    'doc_id':      p.payload['doc_id'],
                    'content':     p.payload['content'],
                    'chunk_index': p.payload.get('chunk_index', 0),
                    'metadata':    p.payload['metadata'],
                    'score':       0.0,
                    'source':      'bm25',
                }
                for p in all_points
            ]
            # FIX 7: use French legal tokenizer instead of naive split()
            # Also include document title in the BM25 searchable text
            tokenized = [
                self._tokenize_fr(
                    f"{c['metadata'].get('doc_title', '')} {c['content']}"
                )
                for c in self._bm25_chunks
            ]
            self._bm25 = BM25Okapi(tokenized)
            logger.info(f"BM25 index built: {len(self._bm25_chunks)} chunks")

            # --- Persist to cache ---
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump({"chunks": self._bm25_chunks, "bm25": self._bm25}, f)
                logger.info(f"BM25 cache saved to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save BM25 cache: {e}")

        except Exception as e:
            logger.warning(f"BM25 index build failed (will skip BM25): {e}")
            self._bm25 = None
            self._bm25_chunks = []

    def bm25_search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        BM25 keyword search over the pre-built index.

        Args:
            query: Raw query string
            top_k: Number of results to return

        Returns:
            List of chunk dicts sorted by BM25 score (descending)
        """
        if not self._bm25 or not self._bm25_chunks:
            return []
        top_k = top_k or config.BM25_TOP_K
        # FIX 7: use same French legal tokenizer for queries and index
        tokens = self._tokenize_fr(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = dict(self._bm25_chunks[idx])
                chunk['score'] = float(scores[idx])
                results.append(chunk)
        logger.debug(f"BM25 search returned {len(results)} results")
        return results

    # ------------------------------------------------------------------
    # TITLE/REFERENCE SEARCH — Fetch chunks from docs whose title or
    # reference matches the query's reference (e.g. "173/2021", "N° 173").
    # Critical when the answer is in the title but not in chunk content.
    # ------------------------------------------------------------------
    _REF_PATTERNS = [
        (r"N[°\s]*\s*(\d+(?:/\d{4})?)", 1),   # N° 173/2021, N°173
        (r"arrêt\s+(?:CCJA\s+)?(?:N[°\s]*)?(\d+(?:/\d{4})?)", 1),
        (r"(\d{2,4}/\d{4})\b", 1),             # 173/2021
    ]

    @classmethod
    def _extract_refs(cls, query: str) -> List[str]:
        """Extract reference tokens like 173/2021 or 173 from a query."""
        import re
        refs: List[str] = []
        for pat, grp in cls._REF_PATTERNS:
            for m in re.finditer(pat, query, re.IGNORECASE):
                refs.append(m.group(grp).strip())
        # Deduplicate while preserving order
        return list(dict.fromkeys([r for r in refs if r]))

    def title_reference_search(self, query: str, top_k: int = 15) -> List[Dict]:
        """
        Search for chunks from documents whose title or reference contains
        the query's reference pattern (e.g. arrêt CCJA N°173/2021).
        Uses server-side full-text filters when available (fast), falling back
        to local substring filtering only if needed.
        """
        refs = self._extract_refs(query)
        if not refs:
            return []
        refs = list(dict.fromkeys(refs))  # dedupe, preserve order
        cache_key = "|".join(r.lower() for r in refs)
        if cache_key in self._title_ref_cache:
            cached = self._title_ref_cache[cache_key]
            return cached[:top_k]

        logger.info(f"Title/reference search: looking for refs {refs}")

        try:
            # Fast path: use Qdrant full-text filters for doc_title/reference.
            # This avoids scanning the full collection.
            from qdrant_client.models import Filter, FieldCondition, MatchText

            should_conditions = []
            for r in refs:
                should_conditions.append(FieldCondition(key="metadata.doc_title", match=MatchText(text=r)))
                should_conditions.append(FieldCondition(key="metadata.reference", match=MatchText(text=r)))

            scroll_filter = Filter(should=should_conditions) if should_conditions else None

            all_chunks: List[Dict] = []
            offset = None
            # Limit scanning to a reasonable bound even if no text index exists.
            max_points_scanned = 5000
            scanned = 0
            while True:
                batch, next_offset = self.qdrant.scroll(
                    collection_name=config.QDRANT_COLLECTION,
                    scroll_filter=scroll_filter,
                    limit=250,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                scanned += len(batch)
                for p in batch:
                    meta = p.payload.get("metadata", {}) or {}
                    all_chunks.append({
                        "chunk_id": p.payload["chunk_id"],
                        "doc_id": p.payload["doc_id"],
                        "content": p.payload.get("content", ""),
                        "chunk_index": p.payload.get("chunk_index", 0),
                        "metadata": meta,
                        "score": 1.0,
                        "source": "title_reference",
                    })
                if next_offset is None or scanned >= max_points_scanned:
                    break
                offset = next_offset

            all_chunks.sort(key=lambda c: (c["metadata"].get("chunk_index", 0), c["chunk_id"]))
            result = all_chunks[:top_k]
            logger.info(f"Title/reference search: {len(all_chunks)} matching chunks, returning top {len(result)}")
            self._title_ref_cache[cache_key] = all_chunks
            return result
        except Exception as e:
            logger.warning(f"Title/reference search failed: {e}")
            return []

    # ------------------------------------------------------------------
    # QUERY EXPANSION
    # ------------------------------------------------------------------

    def analyze_query(self, query: str) -> Dict:
        """
        Analyze a complex query to generate a multi-faceted search plan.
        Returns: {
            "sub_queries": list of simplified queries,
            "hyde_document": hypothetical excerpt,
            "keywords": list of keywords
        }
        """
        if not config.QUERY_EXPANSION_ENABLED:
            return {"sub_queries": [], "hyde_document": "", "keywords": [], "metadata_filters": {}}

        # Fast path: exact reference queries (e.g. CCJA N°173/2021) do NOT need
        # an LLM analysis. This saves an entire OpenAI round-trip and typically
        # improves precision by focusing retrieval on the requested decision.
        refs = self._extract_refs(query)
        if refs:
            ref = refs[0]
            # Provide metadata_filters in the same shape expected downstream.
            return {
                "sub_queries": [],
                "hyde_document": "",
                "keywords": [],
                "metadata_filters": {
                    "reference": ref,
                    "doc_title": ["CCJA", "arrêt", ref],
                },
            }
            
        # FIX 8: improved prompt to extract exact document references as metadata filters
        prompt = f"""Tu es un expert en droit OHADA, droit guinéen et jurisprudence CCJA.
Analyse la requête juridique ci-dessous pour optimiser la récupération de documents dans une base vectorielle.
Renvoie un JSON strict (sans markdown) avec exactement quatre champs :

- "sub_queries": Liste de 2 à 3 sous-questions reformulées. Cibler un aspect précis.
- "hyde_document": Un paragraphe hypothétique de 4 à 5 phrases au style juridique formel qui SERAIT la réponse parfaite.
- "keywords": Liste de 3 à 6 termes juridiques techniques extraits de la question.
- "metadata_filters": Un dictionnaire. Si l'utilisateur mentionne un ARRÊT ou JUGEMENT avec numéro (ex: "arrêt CCJA N°173/2021"), extraire: "reference": "173/2021" ET "doc_title": ["CCJA", "arrêt", "173/2021"]. Pour une Loi/Convention: "doc_title" en liste. S'il donne un numéro (N° 004, 173/2021), "reference" avec le numéro exact. Année: "year". Si général: {{}}.

Question: {query}
"""
        try:
            resp = self.client.chat.completions.create(
                model=config.SIMPLE_QUERY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            import json
            plan = json.loads(resp.choices[0].message.content)
            logger.debug(f"Query analysis plan: {plan}")
            return plan
        except Exception as e:
            logger.warning(f"Query analysis failed: {e}")
            return {"sub_queries": [], "hyde_document": "", "keywords": []}

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for query
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        if query in self._embedding_cache:
            return self._embedding_cache[query]
            
        response = self.client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=[query]
        )
        
        embedding = response.data[0].embedding
        self._embedding_cache[query] = embedding
        return embedding
    
    def vector_search(
        self,
        query: str,
        top_k: int = None,
        filters: Dict = None
    ) -> List[Dict]:
        """
        Perform vector search in Qdrant
        
        Args:
            query: Query text
            top_k: Number of results
            filters: Optional filters
            
        Returns:
            List of search results
        """
        top_k = top_k or config.DENSE_TOP_K
        
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Prepare filters
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if key == 'doc_title':
                    pass # Handled by local python filtering to avoid strict exact MatchValue failure
                elif value and not isinstance(value, (dict, list)):
                    conditions.append(
                        FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                    )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        # Search using query_points
        results = self.qdrant.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter
        ).points
        
        # Format results — expose all payload fields including new metadata
        formatted_results = []
        for result in results:
            formatted_results.append({
                'chunk_id':    result.payload['chunk_id'],
                'doc_id':      result.payload['doc_id'],
                'content':     result.payload['content'],
                'chunk_index': result.payload.get('chunk_index', 0),
                'metadata':    result.payload['metadata'],
                'score':       result.score,
                'source':      'vector'
            })
        
        logger.debug(f"Vector search returned {len(formatted_results)} results")
        
        return formatted_results
    
    def graph_expansion(self, doc_ids: List[str]) -> List[str]:
        """
        Expand document list using graph traversal.
        Returns docs ordered by proximity (closest graph neighbors first).
        Seeds (distance=0) come first, then depth-1 neighbors, then depth-2.

        Args:
            doc_ids: Initial document IDs (seed set)

        Returns:
            Expanded list of document IDs, sorted by minimum hop distance
        """
        if not config.GRAPH_EXPANSION_ENABLED:
            return doc_ids

        # Map: doc_id -> minimum hop distance (0 = seed)
        distance_map: Dict[str, int] = {d: 0 for d in doc_ids}

        def get_related(did):
            return self.graph_builder.get_related_documents(
                doc_id=did,
                relationship_types=config.GRAPH_RELATIONSHIP_TYPES,
                max_depth=config.GRAPH_MAX_DEPTH,
            )

        with ThreadPoolExecutor(max_workers=min(20, len(doc_ids) or 1)) as pool:
            futures = [pool.submit(get_related, did) for did in doc_ids]
            for future in as_completed(futures):
                try:
                    related = future.result()
                    for rel_doc in related:
                        rid  = rel_doc['doc_id']
                        dist = rel_doc.get('distance', 1)
                        # Keep minimum distance if already seen
                        if rid not in distance_map or dist < distance_map[rid]:
                            distance_map[rid] = dist
                except Exception as e:
                    logger.warning(f"Graph traversal failed for a doc: {e}")

        # Sort by distance so closest graph neighbors appear first
        ordered = sorted(distance_map.keys(), key=lambda d: distance_map[d])

        graph_only_count = sum(1 for d, dist in distance_map.items() if dist > 0)
        logger.debug(
            f"Graph expansion: {len(doc_ids)} seeds -> "
            f"{len(ordered)} total docs ({graph_only_count} via graph)"
        )
        return ordered
    
    def search_expanded_documents(self, query: str, doc_ids: List[str], top_k: int = 5) -> List[Dict]:
        """
        Search within expanded document set using a single batched Qdrant query
        (previously made one HTTP call per doc_id — O(N) round-trips).
        
        Args:
            query: Query text
            doc_ids: Document IDs to search within
            top_k: Number of results
            
        Returns:
            Search results
        """
        if not doc_ids:
            return []
        
        # Cap to avoid oversized Qdrant filters — but now the input is ordered
        # by graph proximity, so the first N docs are the most relevant.
        MAX_EXPANDED_DOCS = 60  # was 20; increased so closest neighbors fit
        if len(doc_ids) > MAX_EXPANDED_DOCS:
            logger.debug(f"Capping expanded doc_ids from {len(doc_ids)} to {MAX_EXPANDED_DOCS}")
            doc_ids = doc_ids[:MAX_EXPANDED_DOCS]
        
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Single batched query: match any of the expanded doc_ids using native MatchAny
        doc_filter = Filter(
            must=[
                FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))
            ]
        )
        
        raw_results = self.qdrant.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=query_embedding,
            limit=top_k,
            query_filter=doc_filter
        ).points
        
        results = [
            {
                'chunk_id':    r.payload['chunk_id'],
                'doc_id':      r.payload['doc_id'],
                'content':     r.payload['content'],
                'chunk_index': r.payload.get('chunk_index', 0),
                'metadata':    r.payload['metadata'],
                'score':       r.score,
                'source':      'graph_expansion'
            }
            for r in raw_results
        ]
        
        logger.debug(f"Graph expansion search returned {len(results)} results (1 batched query)")
        
        return results
    
    # ------------------------------------------------------------------
    # OPTION B: CROSS-REFERENCE RESOLVER
    # When a retrieved chunk cites "Art.229" (or similar), automatically
    # fetch the chunk from the SAME document that defines that article.
    # This fixes the common "voir article NNN" gap in legal retrieval.
    # ------------------------------------------------------------------

    # Patterns that detect intra-document legal article references.
    # Deliberately conservative: only match short article codes, not generic
    # "article X" mentions that appear in every document.
    _ART_REF_PATTERN = __import__('re').compile(
        r'(?:art(?:icle)?[.\s]*|l\'article\s+)'    # "art.", "Art.", "l'article"
        r'(\d{1,4}(?:\s*(?:bis|ter|quater|quinquies)?)?)'  # number + optional suffix
        r'(?!\d)',                                   # not followed by another digit
        __import__('re').IGNORECASE
    )

    def _resolve_cross_references(
        self,
        chunks: List[Dict],
        max_refs: int = 6,
    ) -> List[Dict]:
        """
        OPTION B — Article Cross-Reference Resolver.

        Scans each retrieved chunk for patterns like:
          "défini à l'article 229", "Art.243 ter", "voir art. 92"
        Then fetches the chunk(s) from the SAME document that contain
        the text "Art.NNN" so the LLM has the full definition available.

        Args:
            chunks:   Already-retrieved chunk list (post-RRF, pre-rerank)
            max_refs: Max number of cross-referenced chunks to add (cap to
                      avoid exploding the context)

        Returns:
            chunks + any newly resolved cross-reference chunks (deduplicated)
        """
        import re

        existing_ids = {c['chunk_id'] for c in chunks}
        new_chunks: List[Dict] = []

        for chunk in chunks:
            content = chunk['content']
            doc_id  = chunk['doc_id']

            # Extract all article numbers mentioned in this chunk
            matches = self._ART_REF_PATTERN.findall(content)
            if not matches:
                continue

            for art_num in set(matches):
                art_num = art_num.strip()
                if not art_num:
                    continue

                # Build search patterns for how this article appears at the
                # START of a chunk (definition boundary)
                search_patterns = [
                    f"Art.{art_num}.",
                    f"Art.{art_num} ",
                    f"Art. {art_num}",
                    f"Article {art_num}",
                    f"Art.{art_num}\n",
                ]

                # Scroll the same document's chunks for a match
                try:
                    from qdrant_client.models import Filter, FieldCondition, MatchValue
                    offset = None
                    found = False
                    while not found:
                        batch, next_offset = self.qdrant.scroll(
                            collection_name=config.QDRANT_COLLECTION,
                            scroll_filter=Filter(must=[
                                FieldCondition(key="doc_id",
                                               match=MatchValue(value=str(doc_id)))
                            ]),
                            limit=50,
                            offset=offset,
                            with_payload=True,
                            with_vectors=False,
                        )
                        for p in batch:
                            c_content = p.payload.get('content', '')
                            c_id      = p.payload.get('chunk_id', '')
                            if c_id in existing_ids:
                                continue
                            if any(pat in c_content for pat in search_patterns):
                                new_chunks.append({
                                    'chunk_id':    c_id,
                                    'doc_id':      doc_id,
                                    'content':     c_content,
                                    'chunk_index': p.payload.get('chunk_index', 0),
                                    'metadata':    p.payload.get('metadata', {}),
                                    'score':       0.0,
                                    'rrf_score':   0.0,
                                    'source':      'cross_ref',
                                })
                                existing_ids.add(c_id)
                                found = True
                                break
                        if next_offset is None or found:
                            break
                        offset = next_offset
                except Exception as e:
                    logger.debug(f"Cross-ref lookup failed for Art.{art_num}: {e}")

                if len(new_chunks) >= max_refs:
                    break
            if len(new_chunks) >= max_refs:
                break

        if new_chunks:
            logger.info(
                f"Cross-reference resolver added {len(new_chunks)} chunk(s): "
                + ", ".join(c['chunk_id'] for c in new_chunks)
            )

        return chunks + new_chunks

    def reciprocal_rank_fusion(self, result_lists: List[List[Dict]], k: int = None) -> List[Dict]:

        """
        Merge multiple result lists using Reciprocal Rank Fusion
        
        Args:
            result_lists: List of result lists
            k: RRF constant
            
        Returns:
            Fused results
        """
        k = k or config.RRF_K
        
        # Collect all unique chunks
        chunk_scores = {}
        
        for result_list in result_lists:
            for rank, result in enumerate(result_list, start=1):
                chunk_id = result['chunk_id']
                rrf_score = 1.0 / (k + rank)
                
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {
                        'result': result,
                        'rrf_score': 0.0
                    }
                
                chunk_scores[chunk_id]['rrf_score'] += rrf_score
        
        # Sort by RRF score
        fused_results = []
        for chunk_id, data in chunk_scores.items():
            result = data['result'].copy()
            result['rrf_score'] = data['rrf_score']
            fused_results.append(result)
        
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        logger.debug(f"RRF fusion: {sum(len(r) for r in result_lists)} -> {len(fused_results)} unique results")
        
        return fused_results
    
    def rerank_with_llm(self, query: str, results: List[Dict], top_k: int = None) -> List[Dict]:
        """
        Rerank results using LLM
        
        Args:
            query: Query text
            results: Search results
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        if not config.RERANKING_ENABLED:
            return results[:top_k or config.RERANKING_TOP_K]
        
        top_k = top_k or config.RERANKING_TOP_K
        
        # Prepare prompt — include doc_title/ref so LLM can prioritize exact match
        chunks_text = "\n\n".join([
            f"[{i+1}] Titre: {result.get('metadata', {}).get('doc_title', '?')} | Réf: {result.get('metadata', {}).get('reference', '?')}\n"
            f"{result['content'][:900]}{'...' if len(result['content']) > 900 else ''}"
            for i, result in enumerate(results[:15])
        ])
        
        prompt = f"""En tant qu'expert juridique, classe les extraits par pertinence.
PRIORITÉ ABSOLUE: Documents dont le titre ou la référence correspondent EXACTEMENT à la question (ex: arrêt CCJA N°173/2021). Exclure les documents non liés. Seuls les extraits du document demandé doivent figurer en tête.

Question : {query}

Extraits :
{chunks_text}

Format de r\u00e9ponse : Uniquement les num\u00e9ros des {top_k} extraits les plus pertinents, s\u00e9par\u00e9s par des virgules (ex: "3,1,7,2,5").
"""
        
        try:
            response = self.client.chat.completions.create(
                model=config.RERANKING_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )
            
            # Parse rankings
            rankings_text = response.choices[0].message.content.strip()
            # FIX (Finding 7): re.findall is robust to commas, newlines, dots,
            # and extra whitespace.  str.isdigit() was failing on "3." / " 3",
            # silently producing an empty rankings list → unsorted fallback.
            import re as _re
            raw_nums = _re.findall(r'\d+', rankings_text)
            rankings = []
            for s in raw_nums:
                idx = int(s) - 1   # LLM uses 1-based indices
                if 0 <= idx < len(results) and idx not in rankings:
                    rankings.append(idx)
            
            # Reorder results
            reranked = []
            for rank_idx in rankings[:top_k]:
                if 0 <= rank_idx < len(results):
                    reranked.append(results[rank_idx])
            
            # Fill remaining slots if needed using set-based dedup (O(1) lookup)
            reranked_ids = {r['chunk_id'] for r in reranked}
            for result in results:
                if len(reranked) >= top_k:
                    break
                if result['chunk_id'] not in reranked_ids:
                    reranked.append(result)
                    reranked_ids.add(result['chunk_id'])
            
            logger.debug(f"Reranked {len(results)} -> {len(reranked)} results")
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking error: {e}, returning original order")
            return results[:top_k]
    
    @staticmethod
    def build_metadata_filter(filters: Dict) -> Optional[Filter]:
        """
        Build a Qdrant Filter from a plain dict of metadata field names.
        Supports all fields from data_cleaning_1.py:
          doc_title, type, reference, year, institution, source_url, date, modified
        
        Example:
            build_metadata_filter({'type': 'loi', 'year': 2022})
        
        Args:
            filters: Dict mapping metadata field names to match values
            
        Returns:
            Qdrant Filter object or None
        """
        if not filters:
            return None
        conditions = [
            FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        return Filter(must=conditions) if conditions else None

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Dict = None,
        use_graph: bool = True
    ) -> List[Dict]:
        """
        Main retrieval method: advanced query plan + hybrid vector/BM25 + graph + RRF + reranking.
        """
        top_k = top_k or config.FINAL_TOP_K

        # ── Stage 0 & 1 & 2: Advanced query analysis, then vector search and BM25 ──
        vector_lists: List[List[Dict]] = []
        non_vector_lists: List[List[Dict]] = []

        # Wait for query analysis carefully so that if the LLM extracts
        # valuable metadata filters (like exact reference/year), we
        # apply it to the main vector search.
        try:
            plan = self.analyze_query(query)
        except Exception as e:
            logger.warning(f"Query analysis failed: {e}")
            plan = {"sub_queries": [], "hyde_document": "", "keywords": [], "metadata_filters": {}}

        # Merge extracted metadata filters into any passed filters
        extracted_metadata = plan.get("metadata_filters", {})
        if extracted_metadata and isinstance(extracted_metadata, dict):
            if filters is None:
                filters = {}
            # Allow LLM-extracted metadata to overwrite
            for k, v in extracted_metadata.items():
                filters[k] = v

        with ThreadPoolExecutor() as pool:
            # Title/reference search first — fetches docs matching query ref (e.g. arrêt 173/2021)
            title_ref_future = pool.submit(self.title_reference_search, query, top_k=15)
            # Submit original query searches
            bm25_future = pool.submit(self.bm25_search, query, top_k=config.BM25_TOP_K) if config.BM25_ENABLED else None
            exact_vec_future = pool.submit(self.vector_search, query, top_k=config.DENSE_TOP_K, filters=filters)
            
            # Submit variant searches based on the plan
            variant_futures = []
            bm25_variant_futures = []
            
            # Sub-queries: both Vector and BM25
            for sq in plan.get("sub_queries", []):
                if sq and isinstance(sq, str):
                    variant_futures.append(pool.submit(self.vector_search, sq, top_k=config.DENSE_TOP_K, filters=filters))
                    if config.BM25_ENABLED:
                        bm25_variant_futures.append(pool.submit(self.bm25_search, sq, top_k=config.BM25_TOP_K))
            
            # HyDE Document: Vector search only (hallucinated text confuses BM25)
            hyde = plan.get("hyde_document", "")
            if hyde and isinstance(hyde, str):
                variant_futures.append(pool.submit(self.vector_search, hyde, top_k=config.DENSE_TOP_K, filters=filters))
                
            # Keywords: BM25 search only (if available)
            keywords = plan.get("keywords", [])
            if keywords and config.BM25_ENABLED and isinstance(keywords, list):
                kw_str = " ".join(str(k) for k in keywords)
                if kw_str:
                    bm25_variant_futures.append(pool.submit(self.bm25_search, kw_str, top_k=config.BM25_TOP_K))
                
            # Collect Vector results
            try:
                vector_lists.append(exact_vec_future.result())
            except Exception as e:
                logger.warning(f"Exact vector search failed: {e}")
                
            for future in as_completed(variant_futures):
                try:
                    vector_lists.append(future.result())
                except Exception as e:
                    logger.warning(f"Variant vector search failed: {e}")
                    
            # Collect BM25 results
            if bm25_future:
                try:
                    bm25_res = bm25_future.result()
                    if bm25_res:
                        non_vector_lists.append(bm25_res)
                except Exception as e:
                    logger.warning(f"Exact BM25 search failed: {e}")
                    
            for future in as_completed(bm25_variant_futures):
                try:
                    bm25_res = future.result()
                    if bm25_res:
                        non_vector_lists.append(bm25_res)
                except Exception as e:
                    logger.warning(f"Variant BM25 search failed: {e}")

            # Collect title/reference search (high priority — exact doc match)
            try:
                title_ref_res = title_ref_future.result()
                if title_ref_res:
                    non_vector_lists.insert(0, title_ref_res)  # First = highest RRF rank
            except Exception as e:
                logger.warning(f"Title/reference search failed: {e}")

        # If metadata filters exist, aggressively filter BOTH BM25 and Vector results locally
        # This is especially important for 'doc_title' which uses substring matching.
        if extracted_metadata:
            def _is_title_match(query_title: str, doc_title: str) -> bool:
                import re
                stopwords = {"les", "des", "sur", "pour", "dans", "aux", "par", "une", "est", "qui", "que", "avec"}
                def get_words(text):
                    text = re.sub(r'[^\w\s\']', ' ', text.lower())
                    return [w for w in text.split() if len(w) > 2 and w not in stopwords]
                
                q_words = get_words(query_title)
                if not q_words:
                    return True
                    
                d_words = set(get_words(doc_title))
                
                if len(q_words) > 0 and q_words[0] not in d_words:
                    return False
                if len(q_words) > 1 and q_words[1] not in d_words:
                    return False
                    
                matches = sum(1 for w in q_words if w in d_words)
                return (matches / len(q_words)) >= 0.4
                
            def filter_results_list(results_lists):
                filtered_lists = []
                for lst in results_lists:
                    filtered_lst = []
                    for res in lst:
                        match = True
                        for k, v in extracted_metadata.items():
                            # Skip empty filters
                            if not v or v == {} or v == []:
                                continue
                                
                            val_flat = res.get(k, '')
                            val_nested = res.get('metadata', {}).get(k, '')
                            val = str(val_nested or val_flat).strip()
                            
                            if k == 'doc_title':
                                q_vals = v if isinstance(v, list) else [v]
                                matched_any = False
                                for qv in q_vals:
                                    if qv and _is_title_match(str(qv).strip(), val):
                                        matched_any = True
                                        break
                                if not matched_any:
                                    match = False
                                    break
                            else:
                                query_val = str(v).strip()
                                # For 'reference' we often have minor formatting
                                # differences (spaces, prefixes, etc.). Use
                                # substring match instead of strict equality
                                # so "N°025/HAC/P/25" still matches "Décision N°025/HAC/P/25".
                                if k == 'reference':
                                    if query_val.lower() not in val.lower():
                                        match = False
                                        break
                                else:
                                    if val.lower() != query_val.lower():
                                        match = False
                                        break
                        if match:
                            filtered_lst.append(res)
                    if filtered_lst:
                        filtered_lists.append(filtered_lst)
                return filtered_lists

            vector_lists = filter_results_list(vector_lists)
            non_vector_lists = filter_results_list(non_vector_lists)

        # Fuse vector results FIRST to prevent query variants from dominating RRF
        fused_vectors = self.reciprocal_rank_fusion(vector_lists) if vector_lists else []
        if fused_vectors:
            # Limit the fused vectors to dense_top_k so they don't bloat the final pool
            fused_vectors = fused_vectors[:config.DENSE_TOP_K]

        # ── Stage 3: Graph expansion (falls back gracefully on error) ─────────
        if use_graph and config.GRAPH_EXPANSION_ENABLED:
            try:
                # Collect doc_ids for graph expansion from both vector and BM25 results
                seed_docs = {r['doc_id'] for r in fused_vectors}
                for lst in non_vector_lists:
                    seed_docs.update(r['doc_id'] for r in lst)
                
                all_seed_doc_ids = list(seed_docs)
                expanded_doc_ids = self.graph_expansion(all_seed_doc_ids)

                if len(expanded_doc_ids) > len(all_seed_doc_ids):
                    new_doc_ids = [
                        d for d in expanded_doc_ids
                        if d not in set(all_seed_doc_ids)
                    ]
                    graph_results = self.search_expanded_documents(
                        query, new_doc_ids, top_k=5
                    )
                    if graph_results:
                        non_vector_lists.append(graph_results)
            except Exception as e:
                logger.warning(
                    f"Graph expansion failed (falling back to vector-only): {e}"
                )

        # ── Stage 4: Final RRF Fusion ─────────────────────────────────────────
        # Title/reference results (if any) are already first in non_vector_lists → highest RRF
        lists_to_fuse = []
        if fused_vectors:
            lists_to_fuse.append(fused_vectors)
        lists_to_fuse.extend(non_vector_lists)
        
        fused_results = self.reciprocal_rank_fusion(lists_to_fuse)

        # ── Stage 5: LLM Reranking ────────────────────────────────────────────
        final_results = self.rerank_with_llm(query, fused_results, top_k=top_k)

        # ── Stage 6: Cross-Reference Resolver (Option B) ──────────────────────
        # Resolve cross-references on the final ranked results so they don't get
        # dropped by the LLM reranker, then append them.
        try:
            final_results = self._resolve_cross_references(final_results)
        except Exception as e:
            logger.warning(f"Cross-reference resolution failed (non-fatal): {e}")


        logger.info(f"Retrieved {len(final_results)} chunks for query: {query[:50]}...")

        return final_results
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = None,
        filters: Dict = None,
        use_graph: bool = True
    ) -> Tuple[List[Dict], str]:
        """
        Retrieve chunks and format as context string.
        The context header now includes all metadata fields from data_cleaning_1.py:
        doc_title, type, reference, year, institution, source_url, date, modified.
        
        Args:
            query:     Query text
            top_k:     Number of results
            filters:   Optional metadata filters (e.g. {'type': 'loi', 'year': 2022})
            use_graph: Whether to use graph expansion (forwarded from pipeline)
            
        Returns:
            Tuple of (chunks, context_string)
        """
        chunks = self.retrieve(query, top_k=top_k, filters=filters, use_graph=use_graph)
        
        # Format context — include all metadata fields
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            m = chunk['metadata']

            # Build a compact header line with all available metadata.
            # IMPORTANT: Use 'year' as the legal temporal anchor; 'date' and
            # 'modified' come from WordPress and should not drive legal reasoning.
            header_parts = [m.get('doc_title', 'Sans titre')]
            if m.get('type'):        header_parts.append(f"type={m['type']}")
            if m.get('reference'):   header_parts.append(f"réf={m['reference']}")
            if m.get('year'):        header_parts.append(f"année={m['year']}")
            if m.get('institution'): header_parts.append(f"inst={m['institution']}")
            # Only fall back to raw date if no structured legal year exists.
            elif m.get('date'):      header_parts.append(f"date={m['date'][:10]}")

            context_parts.append(
                f"[Document {i}] {' | '.join(header_parts)}\n"
                f"{chunk['content']}\n"
                f"Source: {m.get('source_url', 'N/A')}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        return chunks, context
    
    def close(self):
        """Close connections"""
        self.graph_builder.close()


def main():
    """Test retrieval"""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT
    )
    
    retriever = HybridRetriever()
    
    try:
        # Test query
        query = "Quelles sont les obligations d'une SARL en Guinée?"
        
        print(f"\nQuery: {query}\n")
        
        chunks, context = retriever.retrieve_with_context(query, top_k=5)
        
        print(f"Retrieved {len(chunks)} chunks:\n")
        
        for i, chunk in enumerate(chunks, 1):
            m = chunk['metadata']
            print(f"{i}. {m.get('doc_title', 'N/A')}")
            print(f"   Type:        {m.get('type', 'N/A')}")
            print(f"   Reference:   {m.get('reference', 'N/A')}")
            print(f"   Year:        {m.get('year', 'N/A')}")
            print(f"   Institution: {m.get('institution', 'N/A')}")
            print(f"   Date:        {m.get('date', 'N/A')}")
            print(f"   Modified:    {m.get('modified', 'N/A')}")
            print(f"   Source URL:  {m.get('source_url', 'N/A')}")
            print(f"   Score:       {chunk.get('rrf_score', chunk.get('score', 0)):.4f}")
            print(f"   Retrieved via: {chunk['source']}")
            print(f"   Content:     {chunk['content'][:150]}...\n")
        
    finally:
        retriever.close()


if __name__ == "__main__":
    main()