"""
Caching Module - Multi-Level with Redis and Semantic Cache
"""
import logging
import hashlib
import json
import time
from typing import Optional, Dict, List
import redis
import numpy as np
import faiss
from openai import OpenAI

import config

logger = logging.getLogger(__name__)


class MemoryCache:
    """L1: In-memory LRU cache"""
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or config.MEMORY_CACHE_MAX_SIZE
        self.cache = {}
        self.access_times = {}
        self.hits = 0
        self.misses = 0
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if len(self.cache) >= self.max_size:
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_key]
            del self.access_times[lru_key]
    
    def get(self, key: str) -> Optional[Dict]:
        """Get from cache"""
        if key in self.cache:
            self.access_times[key] = time.time()
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Dict):
        """Set in cache"""
        self._evict_lru()
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear cache and reset stats"""
        self.cache.clear()
        self.access_times.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size
        }


class RedisCache:
    """L2: Redis cache"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0

        if not config.REDIS_CACHE_ENABLED:
            self.client = None
            logger.info("Redis cache disabled")
            return
        
        try:
            self.client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                password=config.REDIS_PASSWORD,
                decode_responses=True
            )
            self.client.ping()
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.client = None
    
    def _make_key(self, query: str, context_hash: str, filters: Dict = None) -> str:
        """Create cache key, incorporating metadata filters when present."""
        filters_str = json.dumps(filters, sort_keys=True) if filters else ''
        filters_hash = hashlib.md5(filters_str.encode()).hexdigest()[:6] if filters_str else '000000'
        return f"graphrag:response:{hashlib.md5(query.encode()).hexdigest()}:{context_hash}:{filters_hash}"
    
    def get(self, query: str, context: str, filters: Dict = None) -> Optional[Dict]:
        """Get from Redis. filters (e.g. {'type': 'loi', 'year': 2022}) are
        incorporated into the cache key so filtered queries are cached separately."""
        if not self.client:
            return None
        
        context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
        key = self._make_key(query, context_hash, filters)
        
        try:
            value = self.client.get(key)
            if value:
                self.hits += 1
                return json.loads(value)
            
            self.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, query: str, context: str, value: Dict, filters: Dict = None):
        """Set in Redis. The full value dict (including 'sources' list) is serialised."""
        if not self.client:
            return
        
        context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
        key = self._make_key(query, context_hash, filters)
        
        try:
            self.client.setex(
                key,
                config.REDIS_CACHE_TTL,
                json.dumps(value, ensure_ascii=False)
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def clear(self):
        """Clear all cache keys and reset stats"""
        if not self.client:
            return
        
        try:
            for key in self.client.scan_iter("graphrag:*"):
                self.client.delete(key)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
        
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'enabled': self.client is not None
        }


class SemanticCache:
    """L3: Semantic cache using FAISS"""
    
    def __init__(self):
        if not config.SEMANTIC_CACHE_ENABLED:
            self.index = None
            logger.info("Semantic cache disabled")
            return
        
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.dimension = config.CACHE_EMBEDDING_DIMENSION
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.query_map = {}  # Maps index position to (query, filters_key)
        self.response_map = {}  # Maps (query, filters_key) to response
        self.hits = 0
        self.misses = 0
        
        logger.info("Semantic cache initialized")
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query"""
        response = self.client.embeddings.create(
            model=config.CACHE_EMBEDDING_MODEL,
            input=[query]
        )
        
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def get(self, query: str, filters: Dict = None) -> Optional[Dict]:
        """Get from semantic cache. filters are incorporated into the lookup key
        so that filtered queries cannot hit an unfiltered (or differently-filtered)
        cached response."""
        if not self.index or self.index.ntotal == 0:
            self.misses += 1
            return None
        
        filters_key = json.dumps(filters, sort_keys=True) if filters else ''
        composite_key = f"{query}||{filters_key}"
        
        try:
            # Embed only the query text (not the filters) for semantic similarity
            query_embedding = self._embed_query(query)
            
            # Search for similar queries
            k = min(5, self.index.ntotal)
            scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
            
            # Check if best match exceeds threshold AND has matching composite key
            if scores[0][0] >= config.SEMANTIC_CACHE_SIMILARITY_THRESHOLD:
                best_idx = indices[0][0]
                stored_composite_key = self.query_map.get(best_idx)
                
                if stored_composite_key == composite_key:
                    self.hits += 1
                    logger.debug(
                        f"Semantic cache hit: {query[:50]}... "
                        f"(score: {scores[0][0]:.3f}, filters: {filters_key or 'none'})"
                    )
                    return self.response_map[composite_key]
            
            self.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Semantic cache get error: {e}")
            self.misses += 1
            return None
    
    def set(self, query: str, response: Dict, filters: Dict = None):
        """Set in semantic cache. filters are incorporated into the storage key."""
        if not self.index:
            return
        
        filters_key = json.dumps(filters, sort_keys=True) if filters else ''
        composite_key = f"{query}||{filters_key}"
        
        try:
            # Check size limit
            if self.index.ntotal >= config.SEMANTIC_CACHE_MAX_SIZE:
                logger.warning("Semantic cache full, not adding new entry")
                return
            
            # Embed only the query text for semantic similarity search
            query_embedding = self._embed_query(query)
            
            # Add to index
            idx = self.index.ntotal
            self.index.add(query_embedding.reshape(1, -1))
            
            # Store mappings using composite key
            self.query_map[idx] = composite_key
            self.response_map[composite_key] = response
            
            logger.debug(
                f"Added to semantic cache: {query[:50]}... "
                f"(filters: {filters_key or 'none'})"
            )
            
        except Exception as e:
            logger.error(f"Semantic cache set error: {e}")
    
    def clear(self):
        """Clear semantic cache and reset stats"""
        if self.index:
            self.index.reset()
            self.query_map.clear()
            self.response_map.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': self.index.ntotal if self.index else 0,
            'max_size': config.SEMANTIC_CACHE_MAX_SIZE,
            'enabled': self.index is not None
        }


class CacheManager:
    """Manages multi-level caching.
    
    All public methods accept an optional `filters` dict that maps metadata
    field names to values (e.g. {'type': 'loi', 'year': 2022, 'institution': 'Présidence'}).
    This ensures that queries with different metadata filters are cached separately.
    Supported filter keys match data_cleaning_1.py metadata:
      doc_title, type, reference, year, institution, source_url, date, modified
    """
    
    def __init__(self):
        self.memory_cache = MemoryCache()
        self.redis_cache = RedisCache()
        self.semantic_cache = SemanticCache()
        
        logger.info("CacheManager initialized")
    
    @staticmethod
    def _filters_suffix(filters: Dict = None) -> str:
        """Stable string representation of filters for use in cache keys."""
        if not filters:
            return ''
        return json.dumps(filters, sort_keys=True)

    def get_response(self, query: str, context: str, filters: Dict = None) -> Optional[Dict]:
        """Get response from cache (L1 -> L2 -> L3).
        
        Args:
            query:   User query
            context: Retrieved context string
            filters: Optional metadata filters (e.g. {'type': 'loi', 'year': 2022})
            
        Returns:
            Cached response dict (includes 'answer', 'sources', 'model', 'usage') or None
        """
        # If the user disables L2 and L3 caches in config, also disable L1
        if not config.REDIS_CACHE_ENABLED and not config.SEMANTIC_CACHE_ENABLED:
            return None

        # L1: Memory cache — key includes filters
        cache_key = hashlib.md5(
            f"{query}{context}{self._filters_suffix(filters)}".encode()
        ).hexdigest()
        result = self.memory_cache.get(cache_key)
        if result:
            logger.debug("L1 cache hit (memory)")
            return result
        
        # L2: Redis cache
        result = self.redis_cache.get(query, context, filters)
        if result:
            logger.debug("L2 cache hit (Redis)")
            # Populate L1
            self.memory_cache.set(cache_key, result)
            return result
        
        # L3: Semantic cache — filters now incorporated to prevent cross-filter hits
        result = self.semantic_cache.get(query, filters)
        if result:
            logger.debug("L3 cache hit (semantic)")
            # Populate L1 and L2
            self.memory_cache.set(cache_key, result)
            self.redis_cache.set(query, context, result, filters)
            return result
        
        return None
    
    def set_response(self, query: str, context: str, response: Dict, filters: Dict = None):
        """Set response in all cache levels.
        
        The response dict should be the full dict returned by generation_5.py:
          {'answer': ..., 'model': ..., 'usage': ..., 'sources': [...]}
        
        Args:
            query:    User query
            context:  Retrieved context string
            response: Full response dict including 'sources' list
            filters:  Optional metadata filters used during retrieval
        """
        cache_key = hashlib.md5(
            f"{query}{context}{self._filters_suffix(filters)}".encode()
        ).hexdigest()
        
        # Set in all levels
        self.memory_cache.set(cache_key, response)
        self.redis_cache.set(query, context, response, filters)
        self.semantic_cache.set(query, response, filters)
    
    def get_cached_sources(self, query: str, context: str, filters: Dict = None) -> List[Dict]:
        """Convenience helper — return just the 'sources' list from a cached response.
        
        Returns an empty list if there is no cache hit or the cached entry has no sources.
        Each source dict contains: doc_id, doc_title, type, reference, year,
        institution, date, modified, source_url.
        """
        cached = self.get_response(query, context, filters)
        if cached:
            return cached.get('sources', [])
        return []
    
    def clear_all(self):
        """Clear all caches"""
        self.memory_cache.clear()
        self.redis_cache.clear()
        self.semantic_cache.clear()
        logger.info("All caches cleared")
    
    def get_all_stats(self) -> Dict:
        """Get statistics from all cache levels"""
        l1_stats = self.memory_cache.get_stats()
        l2_stats = self.redis_cache.get_stats()
        l3_stats = self.semantic_cache.get_stats()
        
        total_hits = l1_stats['hits'] + l2_stats['hits'] + l3_stats['hits']
        total_misses = l1_stats['misses'] + l2_stats['misses'] + l3_stats['misses']
        total_requests = total_hits + total_misses
        
        global_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'l1_memory':  l1_stats,
            'l2_redis':   l2_stats,
            'l3_semantic': l3_stats,
            'global': {
                'total_hits':     total_hits,
                'total_misses':   total_misses,
                'total_requests': total_requests,
                'hit_rate':       global_hit_rate
            }
        }


def main():
    """Test caching"""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT
    )
    
    cache = CacheManager()
    
    # Test data — response now includes 'sources' list from generation_5.py
    query   = "Quel est le capital minimum d'une SARL?"
    context = "[Document 1] Loi sur les SARL | type=loi | réf=L/2023/015 | année=2023 | inst=Présidence | date=2023-04-01\nArticle 15: Le capital minimum est 1.000.000 GNF\nSource: https://kytisa.com/loi-sarl-2023"
    filters = {'type': 'loi', 'year': 2023}  # metadata filters from retrieval_4.py
    
    response = {
        'answer':  'Le capital minimum est 1.000.000 GNF',
        'model':   'gpt-4o-mini',
        'usage':   {'prompt_tokens': 120, 'completion_tokens': 40, 'total_tokens': 160},
        'sources': [{
            'doc_id':      '999',
            'doc_title':   'Loi sur les SARL',
            'type':        'loi',
            'reference':   'L/2023/015',
            'year':        2023,
            'institution': 'Présidence',
            'date':        '2023-04-01T00:00:00',
            'modified':    '2025-01-10T00:00:00',
            'source_url':  'https://kytisa.com/loi-sarl-2023',
        }]
    }
    
    # Set with filters
    cache.set_response(query, context, response, filters=filters)
    print("✓ Response cached (with metadata filters)")
    
    # Get with filters
    cached = cache.get_response(query, context, filters=filters)
    print(f"✓ Retrieved from cache: {cached is not None}")
    if cached:
        print(f"  Answer:  {cached['answer']}")
        print(f"  Sources: {len(cached.get('sources', []))} document(s)")
        for src in cached.get('sources', []):
            print(f"    • {src['doc_title']} | {src['type']} | {src['reference']} | {src['year']} | {src['institution']}")
    
    # get_cached_sources convenience helper
    sources = cache.get_cached_sources(query, context, filters=filters)
    print(f"\n✓ get_cached_sources returned {len(sources)} source(s)")
    
    # Stats
    stats = cache.get_all_stats()
    print(f"\nCache Statistics:")
    print(f"  Global hit rate: {stats['global']['hit_rate']:.1f}%")
    print(f"  L1 (Memory):     {stats['l1_memory']['hit_rate']:.1f}%")
    print(f"  L2 (Redis):      {stats['l2_redis']['hit_rate']:.1f}%")
    print(f"  L3 (Semantic):   {stats['l3_semantic']['hit_rate']:.1f}%")


if __name__ == "__main__":
    main()
