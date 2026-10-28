"""
Single URL Ingestion Module
Handles fetching, cleaning, chunking, and embedding of a single WordPress URL.
"""
import logging
import requests
import re
import json
import traceback
from pathlib import Path
from typing import Dict, Optional

import config
from data_cleaning_1 import DocumentCleaner
from chunking_2 import GraphAwareChunker
from embedding_3 import EmbeddingGenerator

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

class URLIngestor:
    """Ingests a single WordPress URL into the GraphRAG pipeline"""
    
    def __init__(self):
        self.cleaner = DocumentCleaner()
        self.chunker = GraphAwareChunker()
        self.embedder = EmbeddingGenerator()
        logger.info("URLIngestor initialized")

    def extract_slug(self, url: str) -> Optional[str]:
        """Extract post slug from WordPress URL (handles multi-segment paths)"""
        # Remove trailing slash and query params/fragments
        clean_url = url.split('?')[0].split('#')[0].rstrip('/')
        # Take the last part of the path
        parts = clean_url.split('/')
        if len(parts) > 3: # https, "", domain, ...
            return parts[-1]
        return None

    def get_api_url(self, url: str) -> str:
        """Construct WordPress API URL from post URL"""
        match = re.match(r'(https?://[^/]+)', url)
        if match:
            return f"{match.group(1)}/wp-json/wp/v2/posts"
        return config.WORDPRESS_API_URL

    def fetch_post_by_url(self, url: str) -> Dict:
        """Fetch a single WordPress post by discovering its API endpoint from HTML"""
        logger.info(f"Discovering API endpoint for: {url}")
        result = {"post": None, "error": ""}
        try:
            auth = None
            if config.WORDPRESS_USERNAME and config.WORDPRESS_PASSWORD:
                auth = (config.WORDPRESS_USERNAME, config.WORDPRESS_PASSWORD)

            html = ""
            try:
                # 1. Fetch the page HTML to find the API link
                response = requests.get(url, timeout=30, auth=auth)
                if response.status_code == 404:
                    result["error"] = "URL not found (404). Please check for typos."
                    return result
                response.raise_for_status()
                html = response.text
            except requests.exceptions.HTTPError as e:
                logger.warning(f"Failed to fetch HTML from frontend ({e}), falling back to direct API call...")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error when fetching HTML ({e}), falling back to direct API call...")
            
            post_api_match = None
            if html:
                # 2. Look for the specific post API link in HTML
                # <link rel="alternate" type="application/json" href="https://site.com/wp-json/wp/v2/posts/123" />
                post_api_match = re.search(r'type=["\']application/json["\'] href=["\'](https?://[^"\']+/wp-json/wp/v2/posts/\d+)["\']', html)
            
            if post_api_match:
                api_url = post_api_match.group(1)
                logger.info(f"Found specific post API link: {api_url}")
                
                # Fetch the post JSON — context=edit returns full content for authenticated users
                api_res = requests.get(api_url, params={'context': 'edit'}, timeout=30, auth=auth)
                api_res.raise_for_status()
                post = api_res.json()
                
                if isinstance(post, list):
                    result["post"] = post[0] if post else None
                else:
                    result["post"] = post
            else:
                # Fallback to slug search if ID not found or HTML fetch failed
                slug = self.extract_slug(url)
                if not slug:
                    result["error"] = "Could not extract slug from URL."
                    return result
                
                # Try to find the general API root
                api_root = None
                if html:
                    api_link_match = re.search(r'rel=["\']https?://api.w.org/["\'] href=["\'](https?://[^"\']+/wp-json/)["\']', html)
                    if api_link_match:
                        api_root = api_link_match.group(1)
                
                if not api_root:
                    api_root = self.get_api_url(url).replace("wp/v2/posts", "")
                
                api_url = f"{api_root}wp/v2/posts"
                logger.info(f"Searching by slug: {slug} at {api_url}")
                
                # IMPORTANT: context=edit forces WordPress to return FULL unfiltered content
                # for authenticated users (editor/admin). Without this, paywall plugins
                # strip content.rendered even for valid API credentials.
                res = requests.get(api_url, params={'slug': slug, 'context': 'edit'}, timeout=30, auth=auth)
                res.raise_for_status()
                posts = res.json()
                if posts:
                    result["post"] = posts[0]
                else:
                    result["error"] = f"No post found with slug: {slug}"
                    return result

            if not result["post"]:
                result["error"] = "Found API link but post content is empty."
            else:
                # If rendered content is empty but raw exists (context=edit gives both),
                # use raw as fallback so paywall-gated posts still get processed.
                post = result["post"]
                if not post.get('content', {}).get('rendered') and post.get('content', {}).get('raw'):
                    post['content']['rendered'] = post['content']['raw']
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching post: {e}")
            result["error"] = f"Connection error: {str(e)}"
            return result

    def is_document_healthy(self, doc_id: str) -> bool:
        """Check if document exists and has chunks and embeddings"""
        logger.info(f"Checking health of document: {doc_id}")
        try:
            # 1. Check Neo4j for chunks
            with self.cleaner.graph_builder.driver.session(database=config.NEO4J_DATABASE) as session:
                # Use BELONGS_TO which is the relationship used in GraphBuilder.create_chunk_node
                result = session.run(
                    "MATCH (d:Document {id: $doc_id})<-[:BELONGS_TO]-(c:Chunk) RETURN count(c) as count",
                    doc_id=doc_id
                )
                record = result.single()
                if not record or record["count"] == 0:
                    logger.info(f"Document {doc_id} has no chunks in Neo4j")
                    return False
                logger.info(f"Document {doc_id} has {record['count']} chunks in Neo4j")
            
            # 2. Check Qdrant for indexed chunks
            from qdrant_client.http import models
            res = self.embedder.qdrant.count(
                collection_name=config.QDRANT_COLLECTION,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            )
            if res.count == 0:
                logger.info(f"Document {doc_id} has no embeddings in Qdrant")
                return False
            logger.info(f"Document {doc_id} has {res.count} embeddings in Qdrant")
                
            return True
        except Exception as e:
            logger.warning(f"Error checking document health: {e}")
            return False

    def ingest_post(self, post: Dict, force_update: bool = False, force: bool = False) -> Dict:
        """
        Complete pipeline for a single WordPress post object:
        Clean -> Chunk -> Embed -> Graph
        """
        url = post.get('link', '')
        result = {
            "url": url,
            "status": "failed",
            "message": "",
            "doc_id": None,
            "chunks_count": 0
        }
        
        try:
            doc_id = str(post['id'])
            result["doc_id"] = doc_id
            
            # 1. Check if already exists and is healthy (has chunks/embeddings)
            if not force_update and not force and self.is_document_healthy(doc_id):
                result["status"] = "exists"
                result["message"] = "Document already exists and is healthy"
                return result

            # 2. Process Document (Clean + Graph Document Node)
            doc_data = self.cleaner.process_document(post, force=True)
            
            if not doc_data:
                result["message"] = "Document quality too low or skipped"
                return result

            # 3. Chunking
            chunks = self.chunker.chunk_text(
                doc_data['content'], 
                doc_id, 
                {k: v for k, v in doc_data.items() if k != 'content'}
            )
            
            if not chunks:
                result["message"] = "No chunks created (too short?)"
                return result
                
            # Create chunk nodes in Neo4j
            self.chunker.create_chunk_graph_nodes(chunks)
            result["chunks_count"] = len(chunks)
            
            # Save chunks locally
            chunks_file = config.DATA_DIR / "chunks" / f"{doc_id}_chunks.json"
            chunks_file.parent.mkdir(exist_ok=True, parents=True)
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            # 4. Embedding & Qdrant Indexing
            indexed, _ = self.embedder.process_document_chunks(chunks_file, force=True)
            
            if indexed > 0:
                result["status"] = "success"
                result["message"] = f"Successfully ingested {indexed} chunks"
            else:
                result["status"] = "already_indexed"
                result["message"] = "Document already indexed in Qdrant"
                
            return result
        except Exception as e:
            logger.error(f"Error during ingestion of post {post.get('id')}: {e}")
            logger.error(traceback.format_exc())
            result["message"] = f"Internal processing error: {str(e)}"
            return result

    def ingest_url(self, url: str, force_update: bool = False, force: bool = False) -> Dict:
        """
        Complete pipeline for a single URL:
        Fetch -> Clean -> Chunk -> Embed -> Graph
        """
        try:
            # 1. Fetch
            fetch_result = self.fetch_post_by_url(url)
            post = fetch_result["post"]
            if not post:
                return {
                    "url": url,
                    "status": "failed",
                    "message": fetch_result["error"] or "Post not found or invalid URL",
                    "doc_id": None,
                    "chunks_count": 0
                }
                
            return self.ingest_post(post, force_update=force_update or force)
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            logger.error(traceback.format_exc())
            return {
                "url": url,
                "status": "failed",
                "message": f"Internal processing error: {str(e)}",
                "doc_id": None,
                "chunks_count": 0
            }

    def close(self):
        """Close connections"""
        try:
            self.cleaner.close()
            self.chunker.close()
        except:
            pass

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    else:
        test_url = "https://kytisa.com/cour-dappel-de-kankan-chambre-du-premier-president-arret-n-004-du-11-janvier-2017/"
        
    ingestor = URLIngestor()
    try:
        print(f"Ingesting: {test_url}")
        res = ingestor.ingest_url(test_url)
        print(json.dumps(res, indent=2))
    finally:
        ingestor.close()
