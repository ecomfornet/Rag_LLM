import os
import redis
import logging
from pathlib import Path
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_redis_cache():
    """Clear all Redis keys starting with 'graphrag:'"""
    try:
        client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            password=config.REDIS_PASSWORD,
            decode_responses=True
        )
        client.ping()
        
        count = 0
        for key in client.scan_iter("graphrag:*"):
            client.delete(key)
            count += 1
            
        logger.info(f"[OK] Redis cache cleared: {count} keys deleted from 'graphrag:*'")
    except Exception as e:
        logger.error(f"[ERROR] Redis clear failed: {e}")

def clean_file_caches():
    """Delete local cache files like BM25 pickle"""
    cache_files = [
        Path(config.DATA_DIR) / "bm25_cache.pkl"
    ]
    
    for file_path in cache_files:
        if file_path.exists():
            try:
                os.remove(file_path)
                logger.info(f"[OK] Local cache removed: {file_path.name}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to remove {file_path.name}: {e}")
        else:
            logger.info(f"- No local cache found for: {file_path.name}")

def main():
    print("\n" + "="*50)
    print("GraphRAG Cache Cleaner")
    print("="*50 + "\n")
    
    # 1. Clear Redis (L2 Cache)
    clean_redis_cache()
    
    # 2. Clear BM25 Index (Retrieval Cache)
    clean_file_caches()
    
    print("\n" + "="*50)
    print("All caches cleared successfully!")
    print("Note: In-memory caches (L1/L3) will be cleared upon app restart.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
