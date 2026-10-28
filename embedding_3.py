"""
Embedding Module - OpenAI text-embedding-3-large
Generates embeddings and indexes in Qdrant with graph metadata
"""
import logging
import json
import time
import hashlib
from typing import List, Dict, Tuple as TypingTuple
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tenacity import retry, stop_after_attempt, wait_exponential

import config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using OpenAI and indexes in Qdrant"""
    
    def __init__(self):
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=30.0,   # prevent infinite hang on OpenAI network issues
        )
        
        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            https=config.QDRANT_USE_HTTPS,
            api_key=config.QDRANT_API_KEY
        )
        
        self.input_dir = config.DATA_DIR / "chunks"
        self._ensure_collection()
        
        logger.info("EmbeddingGenerator initialized")
    
    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        collections = self.qdrant.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if config.QDRANT_COLLECTION not in collection_names:
            self.qdrant.create_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {config.QDRANT_COLLECTION}")
        else:
            logger.info(f"Using existing collection: {config.QDRANT_COLLECTION}")
    
    @retry(
        stop=stop_after_attempt(config.EMBEDDING_MAX_RETRIES),
        wait=wait_exponential(multiplier=config.EMBEDDING_RETRY_DELAY, min=2, max=10)
    )
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI API with retry logic
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_chunks_batch(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for a batch of chunks
        
        Args:
            chunks: List of chunk dictionaries (already carry document metadata
                    from chunking_2.py, so no extra file lookup is needed)
            
        Returns:
            Chunks with embeddings added
        """
        # Prepare texts with titles prepended.
        # Chunks now carry 'title' directly (set by chunking_2.py).
        texts_with_titles = []
        for chunk in chunks:
            doc_title = chunk.get('title', '')
            if doc_title:
                text_with_title = f"[TITLE: {doc_title}]\n\n{chunk['content']}"
            else:
                text_with_title = chunk['content']
            texts_with_titles.append(text_with_title)
        
        # Generate embeddings in batches
        all_embeddings = []
        
        for i in range(0, len(texts_with_titles), config.EMBEDDING_BATCH_SIZE):
            batch_texts = texts_with_titles[i:i + config.EMBEDDING_BATCH_SIZE]
            batch_embeddings = self.generate_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Rate limiting
            time.sleep(0.1)
        
        # Add embeddings to chunks (keep original content)
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk['embedding'] = embedding
        
        return chunks
    
    def load_document_metadata(self, doc_id: str) -> Dict:
        """Load document metadata from cleaned data"""
        doc_file = config.DATA_DIR / "cleaned" / f"{doc_id}.json"
        
        if doc_file.exists():
            with open(doc_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {}
    
    @staticmethod
    def _get_meta(chunk: Dict, fallback: Dict, field: str, default='') -> str:
        """Read a metadata field from the chunk dict, falling back to a doc-level dict."""
        return chunk.get(field) or fallback.get(field, default)

    @staticmethod
    def _stable_point_id(chunk_id: str) -> int:
        """
        Convert a chunk_id string to a stable Qdrant point integer ID.
        Uses MD5 so the same string always maps to the same integer,
        regardless of Python process, version, or PYTHONHASHSEED.
        (Python's built-in hash() is randomised per-process.)
        """
        return int(hashlib.md5(chunk_id.encode()).hexdigest(), 16) % (2 ** 63)

    def prepare_qdrant_points(self, chunks: List[Dict]) -> List[PointStruct]:
        """
        Prepare Qdrant points with embeddings and full document metadata.
        Chunks already carry all metadata fields from data_cleaning_1.py
        (via chunking_2.py), so no extra file lookup is required.
        
        Args:
            chunks: Chunks with embeddings
            
        Returns:
            List of PointStruct objects
        """
        points = []
        
        for chunk in chunks:
            fallback = {}
            if not chunk.get('title'):  # legacy chunk without inline metadata
                fallback = self.load_document_metadata(chunk['doc_id'])
            
            payload = {
                'chunk_id':    chunk['chunk_id'],
                'doc_id':      chunk['doc_id'],
                'content':     chunk['content'],
                'chunk_index': chunk['chunk_index'],
                'metadata': {
                    'doc_title':   self._get_meta(chunk, fallback, 'title'),
                    'type':        self._get_meta(chunk, fallback, 'type'),
                    'reference':   self._get_meta(chunk, fallback, 'reference'),
                    'year':        chunk.get('year') or fallback.get('year'),
                    'institution': self._get_meta(chunk, fallback, 'institution'),
                    'source_url':  self._get_meta(chunk, fallback, 'source_url'),
                    'date':        self._get_meta(chunk, fallback, 'date'),
                    'modified':    self._get_meta(chunk, fallback, 'modified'),
                }
            }
            
            point = PointStruct(
                id=self._stable_point_id(chunk['chunk_id']),  # MD5-based stable ID
                vector=chunk['embedding'],
                payload=payload
            )
            points.append(point)
        
        return points
    
    def index_chunks(self, chunks: List[Dict]) -> int:
        """
        Index chunks in Qdrant with batching to avoid payload size limits
        
        Args:
            chunks: Chunks with embeddings
            
        Returns:
            Number of chunks indexed
        """
        # Prepare points
        points = self.prepare_qdrant_points(chunks)
        
        # Batch upsert to avoid Qdrant 32MB payload limit
        # Split into smaller batches (50 points at a time to be safe)
        batch_size = 50
        total_indexed = 0
        
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i + batch_size]
            
            self.qdrant.upsert(
                collection_name=config.QDRANT_COLLECTION,
                points=batch_points
            )
            
            total_indexed += len(batch_points)
            logger.debug(f"Indexed batch {i//batch_size + 1}: {len(batch_points)} chunks")
        
        logger.debug(f"Indexed {total_indexed} total chunks in Qdrant")
        
        return total_indexed
    
    def process_document_chunks(self, chunks_file: Path, force: bool = False) -> TypingTuple[int, int]:
        """
        Process chunks from a document: embed and index.
        Safe to resume: skips the file if its embeddings output already exists
        on disk OR if the first chunk is already present in Qdrant.
        
        Args:
            chunks_file: Path to chunks JSON file
            force:        If True, bypasses existing embedding checks
            
        Returns:
            Tuple of (num_indexed, token_estimate) — both 0 if skipped.
            token_estimate = sum of len(content)//4 for all chunks in file.
        """
        # ── Resume-safe: skip if embeddings file already written to disk ──
        output_file = config.DATA_DIR / "embeddings" / chunks_file.name
        
        if not force and output_file.exists():
            logger.debug(f"Embeddings file already exists, skipping: {chunks_file.name}")
            return 0, 0

        # Load chunks
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not chunks:
            return 0, 0
        
        if not force:
            # Check if chunks already exist in Qdrant (secondary guard)
            first_chunk_id = chunks[0]['chunk_id']
            point_id = self._stable_point_id(first_chunk_id)  # stable MD5-based ID
            
            try:
                result = self.qdrant.retrieve(
                    collection_name=config.QDRANT_COLLECTION,
                    ids=[point_id]
                )
                if result:
                    doc_id = chunks[0]['doc_id']
                    logger.info(f"Document {doc_id} chunks already embedded in Qdrant, skipping...")
                    return 0, 0
            except Exception as e:
                logger.debug(f"Chunk not found in Qdrant (will process): {e}")
        
        # Compute token estimate ONCE from the already-loaded chunks (no re-read needed)
        token_estimate = sum(len(c['content']) // 4 for c in chunks)

        # Generate embeddings
        chunks_with_embeddings = self.embed_chunks_batch(chunks)
        
        # Index in Qdrant
        indexed = self.index_chunks(chunks_with_embeddings)
        
        # Save embeddings to disk (marks this doc as done for future resumes)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_with_embeddings, f, ensure_ascii=False, indent=2)
        
        return indexed, token_estimate
    
    def process_all_chunks(self) -> Dict:
        """
        Process all chunk files. Logs progress every 100 documents so you can
        monitor the run. Safely resumes if interrupted (already-embedded docs
        are skipped via the on-disk embeddings file check).
        
        Returns:
            Statistics dictionary
        """
        chunk_files = list(self.input_dir.glob("*_chunks.json"))
        total_files = len(chunk_files)
        
        logger.info(f"Found {total_files} chunk files to process...")
        
        total_chunks = 0
        total_tokens = 0
        skipped_docs = 0
        processed_docs = 0
        
        for i, chunks_file in enumerate(tqdm(chunk_files, desc="Generating embeddings"), start=1):
            indexed, tokens = self.process_document_chunks(chunks_file)  # tuple now
            
            if indexed > 0:
                total_chunks += indexed
                total_tokens += tokens   # already computed inside process_document_chunks
                processed_docs += 1
            else:
                skipped_docs += 1
            
            # Periodic progress log every 100 documents
            if i % 100 == 0 or i == total_files:
                cost_so_far = (total_tokens / 1_000_000) * config.EMBEDDING_COST_PER_1M
                logger.info(
                    f"Progress: {i}/{total_files} files | "
                    f"embedded={processed_docs} skipped={skipped_docs} | "
                    f"chunks={total_chunks:,} | "
                    f"~${cost_so_far:.4f} so far"
                )
        
        # Final cost using config constant
        cost = (total_tokens / 1_000_000) * config.EMBEDDING_COST_PER_1M
        
        stats = {
            'total_documents':     total_files,
            'processed_documents': processed_docs,
            'skipped_documents':   skipped_docs,
            'total_chunks':        total_chunks,
            'total_tokens':        total_tokens,
            'estimated_cost_usd':  cost
        }
        
        logger.info(f"Embedding complete: {stats}")
        
        return stats
    
    def get_collection_info(self) -> Dict:
        """Get Qdrant collection information.
        Compatible with both old (vectors as object) and new (vectors as dict)
        qdrant-client API (>=1.9 changed the schema).
        """
        info = self.qdrant.get_collection(config.QDRANT_COLLECTION)
        vectors = info.config.params.vectors
        # Qdrant >=1.9: vectors is a dict of named vector configs
        # Older versions: vectors is a single VectorParams object with .size
        if hasattr(vectors, 'size'):
            vector_size = vectors.size
        else:
            # Named vectors — take the first one
            vector_size = list(vectors.values())[0].size
        
        return {
            'name':          config.QDRANT_COLLECTION,
            'vector_size':   vector_size,
            'vectors_count': info.vectors_count,
            'points_count':  info.points_count,
            'status':        info.status
        }


def main():
    """Main execution"""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT
    )
    
    generator = EmbeddingGenerator()
    
    stats = generator.process_all_chunks()
    
    print(f"\n✓ Embedding generation complete")
    print(f"  Documents processed: {stats['processed_documents']}/{stats['total_documents']}")
    print(f"  Documents skipped: {stats['skipped_documents']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Estimated cost: ${stats['estimated_cost_usd']:.4f}")
    
    # Collection info
    collection_info = generator.get_collection_info()
    print(f"\nQdrant Collection Info:")
    print(f"  Points: {collection_info['points_count']}")
    print(f"  Status: {collection_info['status']}")


if __name__ == "__main__":
    main()
