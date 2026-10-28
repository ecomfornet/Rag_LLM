"""
Flask Application for GraphRAG Pipeline with Enhanced HTML Interface
Includes graph relationship visualization
"""
from flask import Flask, request, jsonify, render_template_string
import logging
import sys
import os
import atexit
import traceback
import threading

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from main import GraphRAGPipeline

app = Flask(__name__)
# Avoid accidental 404s when a reverse proxy adds/removes trailing slashes
app.url_map.strict_slashes = False

# Configure logging
log_file = os.path.join(config.LOGS_DIR, "app.log")
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('flask_app')

# Global pipeline and lock for thread-safe lazy initialization
rag_pipeline = None
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

def _shutdown():
    """Called automatically on process exit — close Neo4j and Qdrant connections."""
    global rag_pipeline
    if rag_pipeline is not None:
        logger.info("Shutting down GraphRAG Pipeline...")
        rag_pipeline.close()
        rag_pipeline = None

# Register shutdown handler so connections are closed on Ctrl-C or SIGTERM
atexit.register(_shutdown)

# --- HTML TEMPLATE (Enhanced with Graph Features) ---
HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GraphRAG Legal Assistant</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #333; margin: 0; padding: 40px 20px; min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        h1 { text-align: center; color: #2c3e50; margin-bottom: 10px; font-size: 2.5em; }
        .subtitle { text-align: center; color: #7f8c8d; margin-bottom: 30px; font-size: 14px; }
        .badge { display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-left: 10px; }
        
        .search-box { display: flex; gap: 10px; margin-bottom: 30px; }
        input[type="text"] { flex-grow: 1; padding: 15px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; outline: none; transition: border-color 0.3s; }
        input[type="text"]:focus { border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1); }
        button { padding: 15px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; transition: all 0.3s; font-weight: 600; }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
        button:disabled { background: #bdc3c7; cursor: not-allowed; transform: none; }
        
        /* Results Section */
        #result-area { display: none; margin-top: 30px; border-top: 2px solid #eee; padding-top: 20px; }
        .answer-box { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 25px; border-radius: 12px; border-left: 5px solid #667eea; line-height: 1.8; white-space: pre-wrap; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        .meta-info { font-size: 12px; color: #7f8c8d; margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 8px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
        .meta-item { display: flex; align-items: center; gap: 5px; }
        .meta-label { font-weight: 600; color: #2c3e50; }
        
        .sources-section { margin-top: 25px; }
        .sources-section h4 { color: #2c3e50; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
        .source-card { background: #fff; border: 2px solid #e0e0e0; padding: 18px; margin-bottom: 15px; border-radius: 10px; font-size: 14px; transition: all 0.3s; cursor: pointer; position: relative; }
        .source-card:hover { box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2); border-color: #667eea; transform: translateX(5px); }
        .source-card.graph-source { border-left: 5px solid #27ae60; }
        .source-card.vector-source { border-left: 5px solid #3498db; }
        .source-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .source-title { font-weight: bold; color: #2c3e50; font-size: 15px; }
        .source-meta { color: #7f8c8d; font-size: 12px; margin-bottom: 8px; display: flex; gap: 15px; flex-wrap: wrap; }
        .source-score { color: #27ae60; font-size: 13px; font-weight: 600; background: #e8f5e9; padding: 4px 10px; border-radius: 12px; }
        .source-type-badge { font-size: 11px; padding: 3px 8px; border-radius: 10px; font-weight: 600; }
        .source-type-badge.graph { background: #e8f5e9; color: #27ae60; }
        .source-type-badge.vector { background: #e3f2fd; color: #2196f3; }
        .source-link { color: #667eea; font-size: 12px; text-decoration: none; display: inline-block; margin-top: 8px; font-weight: 600; }
        .source-link:hover { text-decoration: underline; }
        .source-content { 
            margin-top: 12px; 
            padding: 12px; 
            background: #f9f9f9; 
            border-left: 3px solid #667eea; 
            border-radius: 6px;
            font-size: 13px;
            line-height: 1.6;
            color: #555;
            max-height: 100px;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }
        .source-content.expanded { max-height: 500px; overflow-y: auto; }

        .loader { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 30px auto; display: none; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .error-box { background: #fee; border-left: 5px solid #e74c3c; padding: 20px; border-radius: 10px; margin-top: 20px; display: none; }
        .error-box h4 { color: #e74c3c; margin: 0 0 10px 0; }
        
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 20px; }
        .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; margin-bottom: 5px; }
        .stat-label { font-size: 12px; opacity: 0.9; }
        
        .cache-indicator { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-left: 10px; }
        .cache-indicator.hit { background: #e8f5e9; color: #27ae60; }
        .cache-indicator.miss { background: #fff3e0; color: #f57c00; }

        /* Agent Trace Panel */
        .agent-trace { margin-top: 20px; background: #1e1e2e; border-radius: 10px; padding: 16px; font-family: monospace; }
        .agent-trace h4 { color: #cdd6f4; margin: 0 0 12px 0; font-size: 13px; letter-spacing: 1px; }
        .agent-step { display: flex; align-items: center; gap: 10px; padding: 6px 0; border-bottom: 1px solid #313244; font-size: 12px; }
        .agent-step:last-child { border-bottom: none; }
        .agent-name { color: #89dceb; width: 200px; flex-shrink: 0; font-weight: 600; }
        .agent-ms { color: #a6e3a1; width: 80px; text-align: right; }
        .agent-notes { color: #cdd6f4; opacity: 0.8; }
        .complexity-badge { display: inline-block; padding: 3px 8px; border-radius: 8px; font-size: 11px; font-weight: 600; margin-left: 6px; }
        .complexity-simple { background: #a6e3a1; color: #1e1e2e; }
        .complexity-complex { background: #f38ba8; color: #1e1e2e; }
        .complexity-multi { background: #fab387; color: #1e1e2e; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚖️ GraphRAG Legal Assistant <span class="badge">Strategy 3 Expert</span></h1>
        <div class="subtitle">Powered by Graph-Enhanced RAG | Neo4j + OpenAI | OHADA & Guinée Legal Database</div>
        
        <div class="search-box">
            <input type="text" id="queryInput" placeholder="Posez une question juridique (ex: 'Quel est le taux d\'imposition sur les plus-values selon le CGI 2024?')..." onkeypress="handleEnter(event)">
            <button onclick="sendQuery()" id="searchBtn">🔍 Search</button>
        </div>

        <div class="loader" id="loader"></div>
        
        <div class="error-box" id="errorBox">
            <h4>❌ Error</h4>
            <p id="errorMessage"></p>
        </div>

        <div id="result-area">
            <h3>📄 Answer:</h3>
            <div class="answer-box" id="answerText"></div>
            
            <div class="sources-section">
                <h4>📚 Sources Used <span id="sourceCount"></span></h4>
                <div id="sourcesList"></div>
            </div>

            <div id="agentTracePanel" style="display:none">
            </div>

            <div class="stats-grid" id="statsGrid"></div>
            <div class="meta-info" id="metaInfo"></div>
        </div>
    </div>

    <script>
        function handleEnter(e) {
            if (e.key === 'Enter') sendQuery();
        }

        async function sendQuery() {
            const query = document.getElementById('queryInput').value.trim();
            if (!query) {
                alert('Please enter a question');
                return;
            }

            // UI Updates
            document.getElementById('searchBtn').disabled = true;
            document.getElementById('searchBtn').textContent = '⏳ Searching...';
            document.getElementById('loader').style.display = 'block';
            document.getElementById('result-area').style.display = 'none';
            document.getElementById('errorBox').style.display = 'none';

            try {
                console.log('Sending query:', query);
                
                const response = await fetch('/api/query?query=' + encodeURIComponent(query));
                
                console.log('Response status:', response.status);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                console.log('Response data:', data);

                if (data.error) {
                    showError(data.error);
                    return;
                }

                // Render Answer
                const answerText = data.answer || 'No answer generated';
                document.getElementById('answerText').textContent = answerText;

                // Render Sources
                const sourcesDiv = document.getElementById('sourcesList');
                sourcesDiv.innerHTML = '';
                
                if (data.sources && data.sources.length > 0) {
                    // Count source types
                    const graphCount = data.sources.filter(s => s.source === 'graph_expansion').length;
                    const vectorCount = data.sources.filter(s => s.source === 'vector').length;
                    
                    document.getElementById('sourceCount').innerHTML = `
                        <span class="source-type-badge vector">${vectorCount} Vector</span>
                        <span class="source-type-badge graph">${graphCount} Graph</span>
                    `;
                    
                    data.sources.forEach((src, index) => {
                        const contentId = `content-${index}`;
                        const isGraph = src.source === 'graph_expansion';
                        const sourceClass = isGraph ? 'graph-source' : 'vector-source';
                        const typeBadge = isGraph 
                            ? '<span class="source-type-badge graph">🔗 Graph</span>' 
                            : '<span class="source-type-badge vector">🎯 Vector</span>';
                        
                        const sourceUrlHtml = src.source_url 
                            ? `<br><a href="${safeUrl(src.source_url)}" target="_blank" class="source-link" onclick="event.stopPropagation()">🔗 View Source</a>` 
                            : '';
                        
                        const score = (src.rrf_score != null ? src.rrf_score : (src.score || 0));
                        const scoreDisplay = score < 1 ? (score * 100).toFixed(1) + '%' : score.toFixed(4);
                        
                        // Build meta items for all available fields
                        let metaItems = `${typeBadge}`;
                        if (src.type)        metaItems += `<span>Type: ${escapeHtml(src.type)}</span>`;
                        if (src.reference)   metaItems += `<span>Réf: ${escapeHtml(src.reference)}</span>`;
                        if (src.year)        metaItems += `<span>Année: ${src.year}</span>`;
                        if (src.institution) metaItems += `<span>Inst: ${escapeHtml(src.institution)}</span>`;
                        if (src.date)        metaItems += `<span>Date: ${escapeHtml(src.date.substring(0,10))}</span>`;
                        
                        sourcesDiv.innerHTML += `
                            <div class="source-card ${sourceClass}" onclick="toggleContent('${contentId}')">
                                <div class="source-header">
                                    <span class="source-title">${index + 1}. ${escapeHtml(src.title || src.doc_title || 'Sans titre')}</span>
                                    <span class="source-score">Score: ${scoreDisplay}</span>
                                </div>
                                <div class="source-meta">${metaItems}</div>
                                ${sourceUrlHtml}

                                <div class="source-content" id="${contentId}">
                                    ${escapeHtml(src.content_preview || '')}
                                </div>
                            </div>
                        `;
                    });
                } else {
                    sourcesDiv.innerHTML = '<p style="color: #999;">No sources available</p>';
                }

                // Render Stats
                if (data.metadata) {
                    const meta = data.metadata;
                    const cacheIndicator = data.cached 
                        ? '<span class="cache-indicator hit">&#10003; Cache Hit</span>'
                        : '<span class="cache-indicator miss">&#10007; Cache Miss</span>';

                    // latency_ms: AgenticPipeline returns {total}, classic returns {total, retrieval, generation}
                    const lms = meta.latency_ms || {};
                    const totalMs  = (lms.total != null ? lms.total : (lms.retrieval||0) + (lms.generation||0));
                    const retrMs   = lms.retrieval != null ? lms.retrieval.toFixed(0) + 'ms' : 'n/a';
                    const genMs    = lms.generation != null ? lms.generation.toFixed(0) + 'ms' : 'n/a';
                    
                    document.getElementById('statsGrid').innerHTML = `
                        <div class="stat-card">
                            <div class="stat-value">${totalMs.toFixed(0)}ms</div>
                            <div class="stat-label">Total Time</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${meta.num_chunks}</div>
                            <div class="stat-label">Chunks Retrieved</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">$${meta.cost_usd.toFixed(6)}</div>
                            <div class="stat-label">Cost</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${meta.usage.total_tokens}</div>
                            <div class="stat-label">Tokens Used</div>
                        </div>
                    `;
                    
                    // Complexity badge (agentic mode only)
                    const cx = meta.complexity || '';
                    const cxClass = cx === 'simple' ? 'complexity-simple' : cx === 'multi-doc' ? 'complexity-multi' : 'complexity-complex';
                    const cxBadge = cx ? `<span class="complexity-badge ${cxClass}">${cx}</span>` : '';

                    document.getElementById('metaInfo').innerHTML = `
                        <div class="meta-item">
                            <span class="meta-label">Model:</span> ${meta.model} ${cxBadge}
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Retrieval:</span> ${retrMs}
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Generation:</span> ${genMs}
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Graph:</span> ${meta.graph_enabled ? '&#10003; Enabled' : '&#10007; Disabled'}
                        </div>
                        <div class="meta-item">
                            ${cacheIndicator}
                        </div>
                    `;

                    // Agent Trace (only for agentic pipeline)
                    const tracePanel = document.getElementById('agentTracePanel');
                    if (data.agent_trace && data.agent_trace.length > 0) {
                        const steps = data.agent_trace.map(s => `
                            <div class="agent-step">
                                <span class="agent-name">${escapeHtml(s.agent)}</span>
                                <span class="agent-ms">${s.latency_ms.toFixed(0)} ms</span>
                                <span class="agent-notes">${escapeHtml(s.notes || '')}</span>
                            </div>
                        `).join('');
                        tracePanel.innerHTML = `
                            <div class="agent-trace">
                                <h4>&#129302; AGENT TRACE</h4>
                                ${steps}
                            </div>
                        `;
                        tracePanel.style.display = 'block';
                    } else {
                        tracePanel.style.display = 'none';
                        tracePanel.innerHTML = '';
                    }
                } else {
                    document.getElementById('metaInfo').textContent = 'Metadata not available';
                }

                // Show Results
                document.getElementById('result-area').style.display = 'block';

            } catch (err) {
                console.error('Error:', err);
                showError('Failed to get response: ' + err.message);
            } finally {
                document.getElementById('searchBtn').disabled = false;
                document.getElementById('searchBtn').textContent = '🔍 Search';
                document.getElementById('loader').style.display = 'none';
            }
        }
        
        function toggleContent(contentId) {
            const content = document.getElementById(contentId);
            content.classList.toggle('expanded');
        }
        
        function showError(message) {
            document.getElementById('errorMessage').textContent = message;
            document.getElementById('errorBox').style.display = 'block';
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Prevent XSS via javascript: or data: URLs in href attributes
        function safeUrl(url) {
            try {
                const u = new URL(url);
                return (u.protocol === 'http:' || u.protocol === 'https:') ? url : '#';
            } catch (e) {
                return '#';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    """Show HTML Interface"""
    return render_template_string(HTML_INTERFACE)

@app.route('/api/query', methods=['GET', 'POST'])
def api_query():
    """API endpoint for queries — supports GET (?query=...) and POST (JSON body).
    POST is preferred for long legal questions that would exceed URL length limits.
    """
    if request.method == 'POST':
        body = request.get_json(force=True, silent=True) or {}
        query = body.get('query')
    else:
        query = request.args.get('query')
    
    if not query:
        return jsonify({"error": "No query parameter provided"}), 400
    
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
            "sources": response['sources'][:5],
            "metadata": response['metadata'],
            "cached": response.get('cached', False),
            "agent_trace": response.get('agent_trace', []),
        })

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# Ensure API paths always return JSON (not HTML), so clients don't fail with
# "Unexpected token '<' ... is not valid JSON" when a 404/500 occurs.
@app.errorhandler(404)
def _handle_404(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not Found", "path": request.path}), 404
    return e, 404


@app.errorhandler(405)
def _handle_405(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Method Not Allowed", "path": request.path}), 405
    return e, 405


@app.errorhandler(500)
def _handle_500(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal Server Error"}), 500
    return e, 500

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


@app.route('/api/diagnose', methods=['GET'])
def api_diagnose():
    """
    Full diagnostic: verify Neo4j, Qdrant, relations, and optionally run a test query.
    Use ?query=... to run a custom test query; otherwise uses a default.
    """
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable
    from qdrant_client import QdrantClient

    result = {
        "neo4j": {"ok": False, "docs": 0, "chunks": 0, "relationships": [], "error": None},
        "qdrant": {"ok": False, "collection": config.QDRANT_COLLECTION, "vectors": 0, "error": None},
        "query_test": {"run": False, "answer_preview": None, "sources_count": 0, "error": None},
        "issues": [],
    }

    # 1. Neo4j check
    try:
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        with driver.session(database=config.NEO4J_DATABASE) as session:
            session.run("RETURN 1")
            doc_count = session.run("MATCH (d:Document) RETURN count(d) as c").single()["c"]
            chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) as c").single()["c"]
            rels = session.run("""
                MATCH ()-[r]->() RETURN type(r) as t, count(r) as c ORDER BY c DESC
            """)
            rel_list = [{"type": r["t"], "count": r["c"]} for r in rels]
        driver.close()
        result["neo4j"] = {"ok": True, "docs": doc_count, "chunks": chunk_count, "relationships": rel_list}
        if doc_count == 0:
            result["issues"].append("Neo4j: No documents")
        if chunk_count == 0:
            result["issues"].append("Neo4j: No chunks")
    except ServiceUnavailable as e:
        result["neo4j"]["error"] = str(e)
        result["issues"].append(f"Neo4j: Connection failed - {e}")
    except Exception as e:
        result["neo4j"]["error"] = str(e)
        result["issues"].append(f"Neo4j: {e}")

    # 2. Qdrant check
    try:
        client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            https=config.QDRANT_USE_HTTPS,
            api_key=config.QDRANT_API_KEY or None,
            timeout=10
        )
        if config.QDRANT_COLLECTION in [c.name for c in client.get_collections().collections]:
            info = client.get_collection(config.QDRANT_COLLECTION)
            result["qdrant"] = {"ok": True, "collection": config.QDRANT_COLLECTION, "vectors": info.points_count}
            if info.points_count == 0:
                result["issues"].append("Qdrant: Collection empty")
        else:
            result["qdrant"]["error"] = f"Collection '{config.QDRANT_COLLECTION}' not found"
            result["issues"].append(result["qdrant"]["error"])
    except Exception as e:
        result["qdrant"]["error"] = str(e)
        result["issues"].append(f"Qdrant: {e}")

    # 3. Optional query test
    query = request.args.get("query") or "Quel est le taux d'imposition sur les plus-values selon le CGI?"
    run_query = request.args.get("run_query", "true").lower() in ("1", "true", "yes")

    if run_query:
        try:
            pipeline = get_pipeline()
            response = pipeline.query(query)
            result["query_test"] = {
                "run": True,
                "answer_preview": (response.get("answer") or "")[:300],
                "sources_count": len(response.get("sources", [])),
                "metadata": response.get("metadata"),
            }
            if not response.get("answer"):
                result["issues"].append("Query: Empty answer")
            if not response.get("sources"):
                result["issues"].append("Query: No sources")
        except Exception as e:
            result["query_test"]["error"] = str(e)
            result["issues"].append(f"Query: {e}")

    result["status"] = "ok" if not result["issues"] else "issues_found"
    return jsonify(result)

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
        logger.info("Starting Flask application...")
        # threaded=True: allows concurrent requests so a slow OpenAI call
        # doesn't block all other users waiting in queue.
        # For production, prefer: gunicorn -w 4 -b 0.0.0.0:5000 app:app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)