"""
Flask Application for GraphRAG Pipeline (API Only)
"""
from flask import Flask, request, jsonify
import logging
import sys
import os
import atexit
import traceback
import threading
import requests

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from main import GraphRAGPipeline

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('flask_app')

# Global pipeline and lock for thread-safe lazy initialization
rag_pipeline = None
url_ingestor = None
_pipeline_lock = threading.Lock()

def get_pipeline():
    global rag_pipeline
    if rag_pipeline is None:
        with _pipeline_lock:
            if rag_pipeline is None:
                if config.LANGGRAPH_ENABLED:
                    logger.info("Initializing AgenticPipeline (LangGraph)...")
                    try:
                        from agents_langgraph import AgenticPipeline
                        rag_pipeline = AgenticPipeline()
                        logger.info("AgenticPipeline initialized successfully")
                    except Exception as e:
                        logger.error(f"Failed to initialize AgenticPipeline: {e}")
                        logger.error(traceback.format_exc())
                        raise
                else:
                    logger.info("Initializing GraphRAGPipeline (classic)...")
                    try:
                        rag_pipeline = GraphRAGPipeline()
                        logger.info("GraphRAGPipeline initialized successfully")
                    except Exception as e:
                        logger.error(f"Failed to initialize pipeline: {e}")
                        logger.error(traceback.format_exc())
                        raise
    return rag_pipeline

def get_url_ingestor():
    global url_ingestor
    if url_ingestor is None:
        with _pipeline_lock:
            if url_ingestor is None:
                try:
                    from ingestion_url import URLIngestor
                    url_ingestor = URLIngestor()
                    logger.info("URLIngestor initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize URLIngestor: {e}")
                    logger.error(traceback.format_exc())
                    raise
    return url_ingestor

def _shutdown():
    """Called automatically on process exit — close Neo4j and Qdrant connections."""
    global rag_pipeline, url_ingestor
    if rag_pipeline is not None:
        logger.info("Shutting down GraphRAG Pipeline...")
        rag_pipeline.close()
        rag_pipeline = None
    if url_ingestor is not None:
        logger.info("Shutting down URL Ingestor...")
        url_ingestor.close()
        url_ingestor = None

# Register shutdown handler so connections are closed on Ctrl-C or SIGTERM
atexit.register(_shutdown)

def _run_query(query):
    # Prevent token-cost abuse: very long queries would send huge prompts to OpenAI
    MAX_QUERY_LENGTH = 2000
    if len(query) > MAX_QUERY_LENGTH:
        return jsonify({
            "error": f"Query too long ({len(query)} chars). Maximum is {MAX_QUERY_LENGTH} chars."
        }), 400

    try:
        logger.info(f"Processing query: {query}")
        pipeline = get_pipeline()
        response = pipeline.query(query)
        
        logger.info(f"Query completed successfully")
        
        return jsonify({
            "answer": response['answer'],
            "sources": response['sources'],
            "metadata": response['metadata'],
            "cached": response.get('cached', False),
            "agent_trace": response.get('agent_trace', []),
        })

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def root():
    """
    If ?query= is present, answer it directly as JSON.
    If ?url= is present, ingest that URL and return status.
    If ?upurl= is present, force-update that URL.
    Otherwise return usage instructions.
    """
    query = request.args.get("query", "").strip()
    url = request.args.get("url", "").strip()
    upurl = request.args.get("upurl", "").strip()

    if query:
        return _run_query(query)
    
    if upurl:
        return _run_url_ingest(upurl, force_update=True)
    
    if url:
        return _run_url_ingest(url, force_update=False)
        
    return jsonify({
        
    })

@app.route("/url", methods=["GET", "POST"])
def api_url():
    """Ingest a specific WordPress URL into the database."""
    force_update = False
    
    if request.method == 'POST':
        body = request.get_json(force=True, silent=True) or {}
        url = body.get('url')
        if not url:
            url = body.get('upurl')
            if url:
                force_update = True
        elif body.get('force_update'):
            force_update = True
    else:
        url = request.args.get('url')
        upurl = request.args.get('upurl')
        if upurl:
            url = upurl
            force_update = True
    
    if not url:
        return jsonify({"error": "No url or upurl parameter provided"}), 400
        
    return _run_url_ingest(url, force_update=force_update)

def _run_url_ingest(url, force_update=False):
    try:
        logger.info(f"Starting ingestion for URL: {url} (force_update={force_update})")
        ingestor = get_url_ingestor()
        
        # Extract slug and use WordPress API config to fetch the post directly
        slug = url.split('?')[0].split('#')[0].rstrip('/').split('/')[-1]
        result = None
        if slug:
            auth = None
            if config.WORDPRESS_USERNAME and config.WORDPRESS_PASSWORD:
                auth = (config.WORDPRESS_USERNAME, config.WORDPRESS_PASSWORD)
                
            api_url = config.WORDPRESS_API_URL
            logger.info(f"Using WordPress API Configuration to fetch post by slug: {slug}")
            
            try:
                # IMPORTANT: context=edit forces WordPress to return the FULL unfiltered
                # content for authenticated users (editor/admin), bypassing paywall plugins
                # that strip content.rendered for unauthenticated or subscriber-level users.
                params = {
                    'slug': slug,
                    'context': 'edit',
                    '_fields': 'id,link,title,content,excerpt,date,modified,author,categories,tags,status'
                }
                res = requests.get(api_url, params=params, timeout=30, auth=auth)
                if res.status_code == 200:
                    posts = res.json()
                    if posts and len(posts) > 0:
                        post = posts[0]
                        # Log content length to confirm paywall bypass worked
                        content_len = len(post.get('content', {}).get('rendered', '') or
                                         post.get('content', {}).get('raw', ''))
                        logger.info(f"Found post via API (content_len={content_len}), ingesting (force_update={force_update})")
                        # If 'rendered' is empty but 'raw' exists, use raw as rendered
                        if not post.get('content', {}).get('rendered') and post.get('content', {}).get('raw'):
                            post['content']['rendered'] = post['content']['raw']
                        result = ingestor.ingest_post(post, force_update=force_update, force=force_update)
                    else:
                        logger.warning(f"No post found for slug '{slug}' via API. Falling back to URL ingestion.")
                elif res.status_code == 401:
                    logger.warning(f"WordPress API auth failed (401). Check WORDPRESS_USERNAME / WORDPRESS_PASSWORD in .env")
                else:
                    logger.warning(f"API returned {res.status_code}. Falling back to URL ingestion.")
            except Exception as e:
                logger.warning(f"Error calling WP API directly: {e}. Falling back to URL ingestion.")
        
        if result is None:
            # Fallback if API lookup fails or no slug extracted
            result = ingestor.ingest_url(url, force_update=force_update, force=force_update)
        
        status_code = 200
        if result["status"] == "failed":
            status_code = 500
        elif result["status"] == "exists":
            status_code = 200
            
        return jsonify(result), status_code

    except Exception as e:
        logger.error(f"Error processing ingestion: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        pipeline = get_pipeline()
        return jsonify({
            "status": "healthy",
            "pipeline_initialized": pipeline is not None
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Get cache statistics"""
    try:
        pipeline = get_pipeline()
        cache_stats = pipeline.cache.get_all_stats()
        return jsonify(cache_stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_cache():
    """Clear all cache levels (L1 memory, L2 Redis, L3 semantic).
    Useful after ingesting new documents or for debugging.
    """
    try:
        pipeline = get_pipeline()
        pipeline.cache.clear_all()
        logger.info("All caches cleared via /api/clear")
        return jsonify({"status": "cleared", "message": "All cache levels cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize pipeline on startup
    try:
        get_pipeline()
        logger.info("Starting Flask API application...")
        # threaded=True: allows concurrent requests so a slow OpenAI call
        # doesn't block all other users waiting in queue.
        # For production, prefer: gunicorn -w 4 -b 0.0.0.0:5000 ask:app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
