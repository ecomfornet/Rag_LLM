"""
Configuration for GraphRAG System (Strategy 3 Expert)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
for dir_path in [DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# ============================================================================
# API CONFIGURATION
# ============================================================================

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Please add it to your .env file."
    )

# WordPress API (strip whitespace to avoid auth failures from .env formatting)
WORDPRESS_API_URL = os.getenv("WORDPRESS_API_URL", "https://kytisa.com/wp-json/wp/v2/posts").strip()
WORDPRESS_USERNAME = os.getenv("WORDPRESS_USERNAME", "").strip()
WORDPRESS_PASSWORD = os.getenv("WORDPRESS_PASSWORD", "").strip()

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================

# OpenAI text-embedding-3-large (3072 dimensions)
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072
EMBEDDING_BATCH_SIZE = 128  # Max texts per API call
EMBEDDING_MAX_RETRIES = 3
EMBEDDING_RETRY_DELAY = 2  # seconds

# For semantic cache (cheaper model)
CACHE_EMBEDDING_MODEL = "text-embedding-3-small"
CACHE_EMBEDDING_DIMENSION = 1536

# ============================================================================
# NEO4J CONFIGURATION
# ============================================================================

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7688")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
# WARNING: do not hardcode a real password here — use .env file
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag123")   # dev default only
NEO4J_DATABASE = "neo4j"

# Graph extraction settings
CITATION_PATTERNS = [
    r"article\s+(\d+)",
    r"loi\s+([A-Z0-9/\-]+)",
    r"décret\s+([A-Z0-9/\-]+)",
    r"code\s+([A-Z0-9/\-]+)",
]

# Relationship types
RELATIONSHIP_TYPES = {
    "CITES":      "Document cites another document",
    "AMENDS":     "Document amends another document",
    "REPLACES":   "Document replaces another document",
    "BELONGS_TO": "Chunk belongs to document",
    "MENTIONS":   "Chunk mentions document",
    "CONTRADICTS":"Document contradicts another",
    "INTERPRETS": "Document interprets/applies another (jurisprudence)",
}

# ============================================================================
# QDRANT CONFIGURATION
# ============================================================================

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6340"))
QDRANT_COLLECTION = "legal_docs_graphrag"
QDRANT_USE_HTTPS = os.getenv("QDRANT_USE_HTTPS", "false").lower() == "true"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# ============================================================================
# REDIS CONFIGURATION
# ============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6380"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = [
    "\n## ",      # Markdown titles
    "\n### ",     # Subtitles
    "\nArticle ", # Legal articles
    "\n\n",       # Paragraphs
    "\n",         # Lines
    ". ",         # Sentences
    " ",          # Words
    ""            # Characters
]

# Citation preservation
PRESERVE_CITATIONS = True
CITATION_EXTEND_CHARS = 300  # Extend chunk if citation is truncated

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

# Vector search
DENSE_TOP_K = 20  # FIX Option A: increased 15→20 so cross-referenced articles are fetched
BM25_TOP_K = 20   # FIX Option A: mirrors DENSE_TOP_K
FINAL_TOP_K = 8  # Increased from 5 to 8 for better context depth
RRF_K = 60        # Reciprocal Rank Fusion constant

# Graph expansion
GRAPH_EXPANSION_ENABLED = True
GRAPH_MAX_DEPTH = 2  # FIX: increased to 2 to reach laws cited at 2 hop distance
GRAPH_RELATIONSHIP_TYPES = ["CITES", "AMENDS", "REPLACES", "INTERPRETS", "COMPLEMENTE_PAR", "APPLIQUE", "MODIFIE_PAR", "PRIME_SUR"]

# Advanced Query Analyzer (HyDE & Sub-queries)
QUERY_EXPANSION_ENABLED = True
QUERY_EXPANSION_COUNT = 3  # (Not used in new logic, but kept for compatibility)

# Reranking
RERANKING_ENABLED = True   # FIX: enabled — LLM reranking critical for legal text precision
RERANKING_MODEL = "gpt-4o-mini"  # FIX: use mini to reduce cost while keeping quality
RERANKING_TOP_K = 12       # Increased from 10 to 12

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Primary LLM (OpenAI GPT-4o-mini)
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 4000
LLM_TOP_P = 0.9

# Routing configuration
SIMPLE_QUERY_MODEL = "gpt-4o-mini"
COMPLEX_QUERY_MODEL = "gpt-4o"  # More powerful model for complex queries

# Query complexity thresholds
COMPLEXITY_THRESHOLD_SIMPLE = 0.3
COMPLEXITY_THRESHOLD_COMPLEX = 0.7

# ============================================================================
# CACHING CONFIGURATION
# ============================================================================

# L1: Memory cache
MEMORY_CACHE_MAX_SIZE = 100
# NOTE: MEMORY_CACHE_TTL is defined for reference; MemoryCache uses LRU eviction
# and does not currently enforce TTL expiry — items live until evicted by size.
MEMORY_CACHE_TTL = 3600  # 1 hour

# L2: Redis cache
REDIS_CACHE_TTL = 86400  # 24 hours
REDIS_CACHE_ENABLED = True

# L3: Semantic cache
SEMANTIC_CACHE_ENABLED = True
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = 0.97
SEMANTIC_CACHE_MAX_SIZE = 10000

# Cache warming
CACHE_WARMING_ENABLED = False
CACHE_TOP_QUERIES = 50

# ============================================================================
# AGENTIC WORKFLOW CONFIGURATION
# ============================================================================

# NOTE: AGENT_MAX_ITERATIONS is defined in the LANGGRAPH section below.
# Do not add a second definition here.
AGENT_TIMEOUT = 30  # seconds

# Self-RAG
SELF_RAG_ENABLED = True   # Enabled: retry on low-confidence answers
SELF_RAG_CONFIDENCE_THRESHOLD = 0.70
SELF_RAG_MAX_RETRIES = 2

# BM25 sparse retrieval (rank_bm25 already installed)
BM25_ENABLED = True
BM25_TOP_K = 20  # FIX Option A: mirrors new DENSE_TOP_K=20

# Query expansion (GPT-4o-mini generates sub-queries, HyDE, and keywords)
QUERY_EXPANSION_ENABLED = True   # FIX: re-enabled — essential for semantic coverage
QUERY_EXPANSION_COUNT = 2        # number of additional query variants

# ============================================================================
# LANGGRAPH MULTI-AGENT CONFIGURATION
# ============================================================================

# Master switch — set False to fall back to the classic single-pipeline path
LANGGRAPH_ENABLED = True

# Query complexity routing thresholds (Router Agent)
AGENT_SIMPLE_THRESHOLD  = 0.35   # below → fast path (cache + direct generation)
AGENT_COMPLEX_THRESHOLD = 0.70   # above → full 6-agent pipeline

# Timeouts + retries
AGENT_MAX_ITERATIONS    = 3      # max Self-RAG retries inside agentic loop
AGENT_TIMEOUT_SECONDS   = 45     # hard timeout for the whole agent graph

# Temporal analyst
AGENT_TEMPORAL_KEYWORDS = [
    "2020", "2021", "2022", "2023", "2024", "2025",
    "récent", "actuel", "en vigueur", "depuis", "avant",
    "modifié", "abrogé", "promulgué",
]

# Fact-checker: minimum citation match ratio to pass without warning
AGENT_FACTCHECK_MIN_RATIO = 0.50

# ============================================================================
# DATA QUALITY THRESHOLDS
# ============================================================================

MIN_DOCUMENT_LENGTH = 200  # characters
MIN_CHUNK_LENGTH = 100     # characters
MIN_QUALITY_SCORE = 0.6    # 0-1 scale

# Graph quality
MIN_CITATION_RESOLUTION = 0.80  # % of citations resolved
MIN_GRAPH_CONNECTIVITY = 0.70   # % of nodes with edges

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "graphrag.log"

# ============================================================================
# PERFORMANCE TARGETS
# ============================================================================

TARGET_RECALL_AT_10 = 0.94
TARGET_PRECISION_AT_5 = 0.95
TARGET_LATENCY_MS = 500  # Improved target (was 1200)
TARGET_CACHE_HIT_RATE = 0.88

# ============================================================================
# LLM COST CONFIGURATION (USD per 1M tokens)
# ============================================================================
# GPT-4o pricing (update here if model/pricing changes)
LLM_INPUT_COST_PER_1M = 2.50
LLM_OUTPUT_COST_PER_1M = 10.00

# Embedding pricing (text-embedding-3-large)
EMBEDDING_COST_PER_1M = 0.13  # USD per 1M tokens
