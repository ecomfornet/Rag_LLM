"""
Main GraphRAG Pipeline
Complete end-to-end system with graph-enhanced retrieval
Supports both classic single-pipeline mode and LangGraph multi-agent mode.
"""
import logging
import argparse
from typing import Dict
import time
import re

from config import FINAL_TOP_K
from retrieval_4 import HybridRetriever
from generation_5 import LLMGenerator, ResponseFormatter
from caching_6 import CacheManager
import config

# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------
# Both vector chunks and BM25 chunks nest metadata under chunk['metadata'].
# chunking_2.py (post-fix) also writes fields flat.  Use this helper so
# main.py is resilient to either shape without hard-coding the nesting.
# ---------------------------------------------------------------------------
def _chunk_meta(chunk: dict, key: str, default=''):
    """Return chunk[key], or chunk['metadata'][key], or default."""
    if key in chunk:
        return chunk[key]
    return chunk.get('metadata', {}).get(key, default)


def _dedup_sources(chunks: list) -> list:
    """Build deduplicated source cards from chunks.

    Multiple chunks may originate from the same document — this helper keeps
    only the first (highest-scored) occurrence per doc_id, preventing duplicate
    source cards in the response.
    """
    seen_doc_ids: set = set()
    sources: list = []
    for chunk in chunks:
        doc_id = chunk.get('doc_id') or chunk.get('id', '')
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        sources.append({
            'doc_id':      doc_id,
            'title':       (_chunk_meta(chunk, 'doc_title')
                            or _chunk_meta(chunk, 'title')),
            'type':        _chunk_meta(chunk, 'type'),
            'reference':   _chunk_meta(chunk, 'reference'),
            'year':        _chunk_meta(chunk, 'year'),
            'institution': _chunk_meta(chunk, 'institution'),
            'date':        _chunk_meta(chunk, 'date'),
            'modified':    _chunk_meta(chunk, 'modified'),
            'source_url':  _chunk_meta(chunk, 'source_url'),
            'score':       chunk.get('rrf_score', chunk.get('score', 0)),
            'source':      chunk.get('source', 'vector'),
            'content_preview': (
                chunk.get('content', '')[:200] + '...'
                if len(chunk.get('content', '')) > 200
                else chunk.get('content', '')
            )
        })
    return sources

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """Complete GraphRAG pipeline"""
    
    def __init__(self):
        logger.info("Initializing GraphRAG Pipeline...")
        
        # Initialize components
        self.retriever = HybridRetriever()
        self.generator = LLMGenerator()
        self.formatter = ResponseFormatter()
        self.cache = CacheManager()
        
        logger.info("GraphRAG Pipeline initialized successfully")
    
    def query(
        self,
        question: str,
        top_k: int = FINAL_TOP_K,
        filters: Dict = None,
        use_cache: bool = True,
        use_graph: bool = True
    ) -> Dict:
        """
        Main query method.
        
        Args:
            question:  User question
            top_k:     Number of chunks to retrieve
            filters:   Optional metadata filters (e.g. {'type': 'loi', 'year': 2022})
                       Supported keys: doc_title, type, reference, year,
                                       institution, source_url, date, modified
            use_cache: Whether to use caching
            use_graph: Whether to use graph expansion
            
        Returns:
            Complete response with metadata
        """
        start_time = time.time()
        
        logger.info(f"Processing query: {question}")
        
        # Step 1: Check cache (filter-aware — different filters = different cache entry)
        # NOTE: We use "" as context key consistently at both get and set so the
        # MD5 hash matches. The semantic cache (L3) already works query-only.
        if use_cache:
            cached_response = self.cache.get_response(question, "", filters=filters)
            if cached_response:
                logger.info("Response found in cache")
                # Return a shallow copy — never mutate the cached dict in-place
                # (L1 memory cache stores a reference, mutating it corrupts the entry)
                return {
                    **cached_response,
                    'cached': True,
                    'latency_ms': (time.time() - start_time) * 1000,
                }
        
        # Step 2: Retrieval (hybrid vector + graph)
        retrieval_start = time.time()
        chunks, context = self.retriever.retrieve_with_context(
            query=question,
            top_k=top_k,
            filters=filters,
            use_graph=use_graph
        )
        retrieval_time = (time.time() - retrieval_start) * 1000

        logger.info(f"Retrieved {len(chunks)} chunks in {retrieval_time:.0f}ms")

        # Step 3: Generation
        generation_start = time.time()
        llm_result = self.generator.generate_response(question, context, chunks=chunks)
        generation_time = (time.time() - generation_start) * 1000

        # Step 4: Self-RAG — retry if confidence is below threshold
        if config.SELF_RAG_ENABLED:
            llm_result, chunks, context, generation_time = self._self_rag_loop(
                question=question,
                llm_result=llm_result,
                chunks=chunks,
                context=context,
                generation_time=generation_time,
                top_k=top_k,
                filters=filters,
                use_graph=use_graph,
            )

        logger.info(f"Generated response in {generation_time:.0f}ms")
        
        # Step 4: Format and validate
        sections = self.formatter.extract_sections(llm_result['answer'])
        is_valid, issues = self.formatter.validate_response(llm_result['answer'], context)
        
        # Step 5: Compute cost
        cost = self.generator.compute_cost(llm_result['usage'])
        
        # Build complete response — sources use safe _chunk_meta helper
        # so that both flat (chunking_2.py output) and nested (Qdrant payload)
        # chunk shapes are handled without a KeyError crash.
        response = {
            'query':   question,
            'answer':  llm_result['answer'],
            'sections': sections,
            'sources': _dedup_sources(chunks),
            'validation': {
                'is_valid': is_valid,
                'issues':   issues
            },
            'metadata': {
                'model':    llm_result['model'],
                'usage':    llm_result['usage'],
                'cost_usd': cost,
                'latency_ms': {
                    'retrieval':  retrieval_time,
                    'generation': generation_time,
                    'total':      (time.time() - start_time) * 1000
                },
                'num_chunks':    len(chunks),
                'graph_enabled': use_graph,
                'filters_used':  filters or {}
            },
            'cached': False
        }
        
        # Cache response — use "" for context key (same as lookup above)
        if use_cache:
            self.cache.set_response(question, "", response, filters=filters)
        
        return response
    
    # ----------------------------------------------------------------
    # Self-RAG helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _parse_confidence(answer: str) -> float:
        """
        Parse **CONFIANCE:** score from LLM answer.
        Returns float in [0, 1]; defaults to 1.0 if not found.
        """
        m = re.search(r'\*\*CONFIANCE:\*\*\s*([0-9.]+)', answer)
        if m:
            try:
                return min(1.0, max(0.0, float(m.group(1))))
            except ValueError:
                pass
        return 1.0  # assume confident if score is missing

    def _self_rag_loop(
        self,
        question: str,
        llm_result: Dict,
        chunks,
        context: str,
        generation_time: float,
        top_k: int,
        filters,
        use_graph: bool,
    ):
        """
        Iterative Self-RAG refinement.
        If CONFIANCE < threshold, re-retrieve with an expanded query and
        re-generate up to SELF_RAG_MAX_RETRIES times.

        Returns: (llm_result, chunks, context, total_generation_time)
        """
        for attempt in range(1, config.SELF_RAG_MAX_RETRIES + 1):
            confidence = self._parse_confidence(llm_result['answer'])
            if confidence >= config.SELF_RAG_CONFIDENCE_THRESHOLD:
                break

            logger.info(
                f"Self-RAG retry {attempt}/{config.SELF_RAG_MAX_RETRIES} "
                f"(confidence={confidence:.2f} < {config.SELF_RAG_CONFIDENCE_THRESHOLD})"
            )

            # Expand the query for retry
            expanded_question = (
                f"{question} — précisez en détail avec les références légales exactes."
            )

            # Re-retrieve
            try:
                new_chunks, new_context = self.retriever.retrieve_with_context(
                    query=expanded_question,
                    top_k=top_k,
                    filters=filters,
                    use_graph=use_graph,
                )
                chunks, context = new_chunks, new_context
            except Exception as e:
                logger.warning(f"Self-RAG re-retrieval failed: {e}")
                break

            # Re-generate
            gen_start = time.time()
            try:
                llm_result = self.generator.generate_response(
                    expanded_question, context, chunks=chunks
                )
                generation_time += (time.time() - gen_start) * 1000
            except Exception as e:
                logger.warning(f"Self-RAG re-generation failed: {e}")
                generation_time += (time.time() - gen_start) * 1000
                break

        return llm_result, chunks, context, generation_time

    def interactive_mode(self):
        """Interactive question-answering mode"""
        print("\n" + "="*70)
        print("[BOT] GraphRAG Assistant - Strategy 3 Expert")
        print("="*70)
        print("\nCommands:")
        print("  /stats  - Show cache statistics")
        print("  /clear  - Clear cache")
        print("  /quit   - Exit")
        print("\n" + "="*70 + "\n")
        
        while True:
            try:
                # Get user input
                question = input("\n>> Question: ").strip()
                
                if not question:
                    continue
                
                # Handle commands
                if question == '/quit':
                    print("\n-- Au revoir!")
                    break
                
                elif question == '/stats':
                    self._show_stats()
                    continue
                
                elif question == '/clear':
                    self.cache.clear_all()
                    print("✓ Cache cleared")
                    continue
                
                # Process query
                print("\n... Processing...")
                response = self.query(question)
                
                # Display response
                self._display_response(response)
                
            except KeyboardInterrupt:
                print("\n\n-- Au revoir!")
                break
            
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                print(f"\n[ERROR] Erreur: {str(e)}")
    
    def _display_response(self, response: Dict):
        """Display formatted response"""
        print("\n" + "="*70)
        print("REPONSE")
        print("="*70)
        
        print(response['answer'])
        
        print("\n" + "-"*70)
        print("METADONNEES")
        print("-"*70)
        
        metadata = response['metadata']
        print(f"Modèle: {metadata['model']}")
        print(f"Tokens: {metadata['usage']['total_tokens']} " +
              f"(prompt: {metadata['usage']['prompt_tokens']}, " +
              f"completion: {metadata['usage']['completion_tokens']})")
        print(f"Coût: ${metadata['cost_usd']:.6f}")
        latency = metadata['latency_ms']
        total_ms = latency.get('total', latency.get('retrieval', 0) + latency.get('generation', 0))
        print(f"Latence totale: {total_ms:.0f}ms")
        if 'retrieval' in latency:
            print(f"  retrieval:   {latency['retrieval']:.0f}ms")
        if 'generation' in latency:
            print(f"  generation:  {latency['generation']:.0f}ms")
        print(f"Documents: {metadata['num_chunks']}")
        print(f"Graph: {'[ON]' if metadata['graph_enabled'] else '[OFF]'}")
        print(f"Cache: {'[HIT]' if response['cached'] else '[MISS]'}")

        # Agent trace (only present in agentic mode)
        if response.get('agent_trace'):
            print("\n" + "-"*70)
            print("AGENT TRACE")
            print("-"*70)
            for step in response['agent_trace']:
                print(f"  {step['agent']:35s} {step['latency_ms']:7.1f} ms  {step.get('notes','')}")
            if 'complexity' in metadata:
                print(f"  Complexity: {metadata['complexity']} "
                      f"(score={metadata.get('complexity_score', 0):.2f})")
            if 'confidence' in metadata:
                print(f"  Confidence: {metadata['confidence']:.2f}")
            fc_ok = metadata.get('fact_check_ok', True)
            print(f"  Fact-check: {'[OK]' if fc_ok else '[WARN]'}")
        
        # Validation
        if not response['validation']['is_valid']:
            print("\n[!] Validation issues:")
            for issue in response['validation']['issues']:
                print(f"  - {issue}")
        
        # Sources
        print("\n" + "-"*70)
        print("SOURCES")
        print("-"*70)
        for i, source in enumerate(response['sources'], 1):
            print(f"\n{i}. {source['title']}")
            if source.get('type'):        print(f"   Type:        {source['type']}")
            if source.get('reference'):   print(f"   Référence:   {source['reference']}")
            if source.get('year'):        print(f"   Année:       {source['year']}")
            if source.get('institution'): print(f"   Institution: {source['institution']}")
            if source.get('date'):        print(f"   Date:        {source['date'][:10]}")
            print(f"   Score: {source['score']:.4f} | Via: {source['source']}")
            if source.get('source_url'):
                print(f"   URL: {source['source_url']}")
    
    def _show_stats(self):
        """Show cache statistics"""
        stats = self.cache.get_all_stats()
        
        print("\n" + "="*70)
        print("CACHE STATISTICS")
        print("="*70)
        
        print("\nL1 Memory Cache:")
        print(f"  Hit rate: {stats['l1_memory']['hit_rate']:.1f}%")
        print(f"  Hits: {stats['l1_memory']['hits']}")
        print(f"  Size: {stats['l1_memory']['size']}/{stats['l1_memory']['max_size']}")
        
        print("\nL2 Redis Cache:")
        print(f"  Hit rate: {stats['l2_redis']['hit_rate']:.1f}%")
        print(f"  Enabled: {stats['l2_redis']['enabled']}")
        
        print("\nL3 Semantic Cache:")
        print(f"  Hit rate: {stats['l3_semantic']['hit_rate']:.1f}%")
        print(f"  Size: {stats['l3_semantic']['size']}/{stats['l3_semantic']['max_size']}")
        print(f"  Enabled: {stats['l3_semantic']['enabled']}")
        
        print("\nGlobal:")
        print(f"  Total hit rate: {stats['global']['hit_rate']:.1f}%")
        print(f"  Total requests: {stats['global']['total_requests']}")
    
    def close(self):
        """Close all database connections (Neo4j + Qdrant)."""
        self.retriever.close()          # closes Qdrant client
        # Also close the graph_builder Neo4j driver inside the retriever
        if hasattr(self.retriever, 'graph_builder') and self.retriever.graph_builder:
            self.retriever.graph_builder.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='GraphRAG Pipeline - Strategy 3 Expert')
    parser.add_argument('--query',       '-q', type=str,            help='Single query to process')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--top-k',       '-k', type=int, default=FINAL_TOP_K, help='Number of chunks')
    parser.add_argument('--no-cache',          action='store_true', help='Disable caching')
    parser.add_argument('--no-graph',          action='store_true', help='Disable graph expansion')
    parser.add_argument('--agentic',     '-a', action='store_true',
                        help='Use LangGraph multi-agent pipeline (Strategy 3)')
    parser.add_argument(
        '--filter', '-f',
        nargs=2, metavar=('KEY', 'VALUE'), action='append',
        help='Metadata filter, e.g. --filter type loi --filter year 2022'
    )
    
    args = parser.parse_args()
    
    # Build filters dict from --filter KEY VALUE pairs
    filters = None
    if args.filter:
        filters = {}
        for key, value in args.filter:
            # Auto-cast year to int
            filters[key] = int(value) if key == 'year' and value.isdigit() else value
    
    # Initialize pipeline
    use_agentic = args.agentic or config.LANGGRAPH_ENABLED
    try:
        if use_agentic:
            from agents_langgraph import AgenticPipeline
            pipeline = AgenticPipeline()
            print("[BOT] Mode: LangGraph Multi-Agent Pipeline")
        else:
            pipeline = GraphRAGPipeline()
            print("[SYS] Mode: Classic Single-Pipeline")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        print(f"\n[ERROR] Erreur d'initialisation: {str(e)}")
        print("\nAssurez-vous que:")
        print("  1. Docker services are running (docker-compose up -d)")
        print("  2. Data is indexed (run data_cleaning_1.py, chunking_2.py, embedding_3.py)")
        print("  3. OPENAI_API_KEY is set in .env")
        if use_agentic:
            print("  4. langgraph is installed: pip install langgraph>=0.1.0")
        return
    
    # Single query mode
    if args.query:
        try:
            response = pipeline.query(
                question=args.query,
                top_k=args.top_k,
                filters=filters,
                use_cache=not args.no_cache,
                use_graph=not args.no_graph
            )
            if hasattr(pipeline, '_display_response'):
                pipeline._display_response(response)
            else:
                GraphRAGPipeline._display_response(None, response)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"\n[ERROR] Erreur: {str(e)}")
        finally:
            pipeline.close()
    
    # Interactive mode
    elif args.interactive:
        try:
            pipeline.interactive_mode()
        finally:
            pipeline.close()
    
    # Default: show help
    else:
        parser.print_help()
        print("\nExemples:")
        print("  python main.py -i                                    # Mode interactif")
        print("  python main.py -q \"Obligations SARL Guinee\"          # Query unique")
        print("  python main.py -q \"Capital minimum\" -k 10           # Avec top-k custom")
        print("  python main.py -q \"Test\" --no-graph                 # Sans graph expansion")
        print("  python main.py -q \"Test\" --agentic                  # LangGraph multi-agent")
        print("  python main.py -q \"Test\" -a --no-cache              # Agentic, no cache")


if __name__ == "__main__":
    import sys
    try:
        main()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[FATAL] {exc}")
        sys.exit(1)
