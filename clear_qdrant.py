"""
Clear Qdrant Collection - Utility Script
Deletes and recreates the Qdrant collection so it can be re-populated with
full document metadata: title, type, reference, year, institution,
source_url, date, modified  (all fields from data_cleaning_1.py).
"""
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_qdrant_collection():
    """Delete and recreate Qdrant collection"""
    
    # Initialize Qdrant client
    qdrant = QdrantClient(
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT,
        https=config.QDRANT_USE_HTTPS,
        api_key=config.QDRANT_API_KEY
    )
    
    collection_name = config.QDRANT_COLLECTION
    
    try:
        # Get current collection info
        info = qdrant.get_collection(collection_name)
        points_count = info.points_count
        
        logger.info(f"Current collection '{collection_name}' has {points_count:,} points")
        
        # Ask for confirmation
        print(f"\n⚠️  WARNING: This will delete ALL {points_count:,} embeddings from '{collection_name}'")
        print("You will need to re-run embedding_3.py to regenerate embeddings with full metadata.")
        print("  Fields stored per point: title, type, reference, year,")
        print("                           institution, source_url, date, modified")
        
        response = input("\nAre you sure you want to continue? (yes/no): ").strip().lower()
        
        if response != 'yes':
            logger.info("Operation cancelled by user")
            print("\n❌ Operation cancelled")
            return
        
        # Delete collection
        logger.info(f"Deleting collection '{collection_name}'...")
        qdrant.delete_collection(collection_name)
        logger.info("✓ Collection deleted")
        
        # Recreate collection
        logger.info(f"Recreating collection '{collection_name}'...")
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )
        logger.info("✓ Collection recreated")
        
        print(f"\n✅ Success! Collection '{collection_name}' has been cleared")
        print("\nNext steps:")
        print("1. Run: python embedding_3.py")
        print("2. Wait for all embeddings to regenerate")
        print("   Each Qdrant point will now carry full metadata:")
        print("   title · type · reference · year · institution · source_url · date · modified")
        print("3. Test your queries to see improved metadata-aware search")
        
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    clear_qdrant_collection()
