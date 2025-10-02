#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================================================
# PART 1: GLOBAL CONFIGURATION AND IMPORTS
# =====================================================================

# --- API Key ---
# Add your OpenRouter API key here to enable the AI model
OPENROUTER_API_KEY = "sk-or-v1-"

# --- WordPress Site Configuration ---
WORDPRESS_URL = "https://yourwebsite.com/"
WORDPRESS_USER = "user"
WORDPRESS_APP_PASSWORD = "Key"

# --- RAG System Optimization ---
CONTEXT_FOR_LLM_COUNT = 10      # Number of parent documents to send to the LLM
EMBEDDING_BATCH_SIZE = 32       # Reduced batch size to save RAM during embedding
CHROMA_ADD_BATCH_SIZE = 2000    # Batch size for adding documents to ChromaDB
WP_INGESTION_BATCH_SIZE = 50    # Number of WordPress posts to fetch per API call

# --- Force pysqlite3 for ChromaDB ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- Standard Library Imports ---
import os
import uuid
import sqlite3
import threading
from io import BytesIO
import re
import html
from typing import List, Dict, Any, Tuple

# --- Flask Imports for the Web API ---
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# --- Core RAG Library Imports ---
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

# --- Resilient HTTP Session with Retries ---
def make_session():
    """Creates a robust requests session with automatic retries."""
    s = requests.Session()
    retries = Retry(
        total=3, connect=3, read=3, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

http = make_session()

# =====================================================================
# PART 2: UTILITY FUNCTIONS
# =====================================================================

def clean_html(raw_html: str) -> str:
    """Nettoie une chaine HTML en supprimant les balises et en normalisant les espaces."""
    clean_text = re.sub('<[^<]+?>', '', raw_html)
    clean_text = html.unescape(clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

# =====================================================================
# PART 3: UPGRADED SQLITE FTS5 FOR KEYWORD SEARCH
# =====================================================================
class SQLiteFTS5Search:
    """
    Manages a disk-based SQLite FTS5 index for fast keyword search.
    This version returns scores for better result fusion.
    """
    def __init__(self, db_path: str = "./fts5_search.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initializes the FTS5 database and table if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
                USING fts5(
                    doc_id, content, post_id UNINDEXED,
                    tokenize = 'porter unicode61 remove_diacritics 1'
                )
            """)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Adds a list of documents to the FTS5 index."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                data_to_insert = [
                    (doc['id'], doc['content'], doc.get('post_id', ''))
                    for doc in documents
                ]
                conn.executemany(
                    "INSERT OR REPLACE INTO documents_fts (doc_id, content, post_id) VALUES (?, ?, ?)",
                    data_to_insert
                )

    def search(self, query: str, limit: int = 50, post_id: str = None) -> List[Tuple[str, float]]:
        """
        Searches the FTS5 index and returns a list of (doc_id, score) tuples.
        The score is the BM25 rank from FTS5. Lower is better.
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                # A more robust query syntax: treats user input as a phrase/set of terms.
                # FTS5 will handle the matching of individual words.
                clean_query = f'"{query}"'

                sql = "SELECT doc_id, rank FROM documents_fts WHERE documents_fts MATCH ? "
                params = [clean_query]
                if post_id:
                    sql += "AND post_id = ? "
                    params.append(post_id)
                
                sql += "ORDER BY rank LIMIT ?"
                params.append(limit)

                cursor = conn.execute(sql, tuple(params))
                # Returns a list of tuples: [('doc_id_1', 1.23), ('doc_id_2', 1.45), ...]
                return cursor.fetchall()

# =====================================================================
# PART 4: THE OPTIMIZED RAG SYSTEM
# =====================================================================

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]], k: int = 60
) -> List[Tuple[str, float]]:
    """
    Performs Reciprocal Rank Fusion on multiple ranked lists of documents.
    Args:
        ranked_lists: A list of lists, where each inner list contains (doc_id, score) tuples.
        k: A constant used in the RRF formula, defaults to 60.
    Returns:
        A single fused and sorted list of (doc_id, score) tuples.
    """
    fused_scores = {}
    for rank_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(rank_list):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            # The core RRF formula
            fused_scores[doc_id] += 1 / (k + rank + 1)

    # Sort by the fused score in descending order
    reranked_results = sorted(
        fused_scores.items(), key=lambda x: x[1], reverse=True
    )
    return reranked_results


class OptimizedRAGSystem:
    def __init__(self):
        print("Initializing Optimized RAG System...")
        if not OPENROUTER_API_KEY or "sk-or-v1" not in OPENROUTER_API_KEY:
            sys.exit("ERREUR: La variable OPENROUTER_API_KEY n'est pas definie correctement.")
        self.api_key = OPENROUTER_API_KEY

        print("Loading lightweight models...")
        self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-base', device='cpu')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device='cpu')

        separators = ["\n\n", "\n", ". ", " ", ""]
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=300, separators=separators)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=75, separators=separators)
        self.micro_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=25, separators=separators)

        print("Configuring persistent ChromaDB client...")
        self.client = chromadb.Client(
            Settings(is_persistent=True, persist_directory="./rag_chroma_db", anonymized_telemetry=False)
        )
        self.collection_name = "documents_micro_chunks"
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})

        self.fts_search = SQLiteFTS5Search()
        print("? RAG System is ready.")

    def generate_llm_response(self, query: str, context_documents: List[str]) -> str:
        """
        Sends the query and context to the LLM with a powerful, strict prompt
        that understands how to use the "exact_match" tag.
        """
        if not context_documents:
            return "Aucune information pertinente n'a ete trouvee dans la base de donnees pour repondre a cette question."

        context_str = ""
        for i, doc in enumerate(context_documents, 1):
            context_str += f"<document index='{i}'>\n{doc}\n</document>\n\n"

        prompt = f"""
        Vous etes un assistant de recherche juridique expert et meticuleux.
        Votre mission est de repondre a la question de l'utilisateur en vous basant EXCLUSIVEMENT sur les documents fournis.

        REGLES STRICTES:
        1.  Votre reponse DOIT etre une synthese directe des informations presentes dans les documents.
        2.  Portez une attention PARTICULIERE au contenu a l'interieur des balises <exact_match>. Ce passage est le plus pertinent. Basez votre reponse principalement sur ce dernier.
        3.  NE PAS utiliser de connaissances externes. Ne faites aucune supposition.
        4.  NE JAMAIS mentionner les documents, leur numero, ou les balises <exact_match>. La reponse doit etre fluide et directe.
        5.  Repondez en francais de maniere claire et professionnelle.
        6.  Si, et seulement si, les documents ne contiennent AUCUNE information permettant de repondre, repondez UNIQUEMENT par la phrase exacte : "Aucune information pertinente n'a ete trouvee dans la base de donnees pour repondre a cette question."

        CONTEXTE FOURNI:
        {context_str}
        QUESTION DE L'UTILISATEUR:
        {query}

        REPONSE SYNTHETIQUE ET DIRECTE:
        """

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": "meta-llama/llama-3-8b-instruct",
            "messages": [{"role": "user", "content": prompt.strip()}],
            "max_tokens": 4820,
            "temperature": 0.03,
            "top_p": 0.9,
        }
        try:
            r = http.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=45)
            r.raise_for_status()
            response_text = r.json()["choices"][0]["message"]["content"]
            return response_text.strip()
        except requests.exceptions.RequestException as e:
            print(f"Erreur API: {e}")
            return "Erreur lors de la communication avec le modele de langage."

    def add_document(self, text: str, filename: str, post_id: str = None, category: str = None) -> int:
        parent_chunks = self.parent_splitter.split_text(text)
        if not parent_chunks: return 0

        all_micro_chunks, metadatas, ids, fts_docs = [], [], [], []
        for i, p_chunk in enumerate(parent_chunks):
            child_chunks = self.child_splitter.split_text(p_chunk)
            for j, c_chunk in enumerate(child_chunks):
                micro_chunks = self.micro_splitter.split_text(c_chunk)
                for m_chunk in micro_chunks:
                    new_id = f"{post_id or 'file'}_{uuid.uuid4()}"
                    ids.append(new_id)
                    all_micro_chunks.append(m_chunk)
                    metadata = {"filename": filename, "parent_chunk": p_chunk}
                    if post_id: metadata["post_id"] = post_id
                    if category: metadata["category"] = category
                    metadatas.append(metadata)
                    fts_docs.append({'id': new_id, 'content': m_chunk, 'post_id': post_id})
        
        if not all_micro_chunks: return 0

        print(f"  Creating embeddings for {len(all_micro_chunks)} micro-chunks...")
        embeddings = self.embedding_model.encode(all_micro_chunks, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=False).tolist()

        print(f"  Adding {len(all_micro_chunks)} documents to ChromaDB and FTS5 index...")
        for i in range(0, len(ids), CHROMA_ADD_BATCH_SIZE):
            end_i = i + CHROMA_ADD_BATCH_SIZE
            self.collection.add(ids=ids[i:end_i], embeddings=embeddings[i:end_i], documents=all_micro_chunks[i:end_i], metadatas=metadatas[i:end_i])
        self.fts_search.add_documents(fts_docs)
        return len(all_micro_chunks)

    def is_post_id_ingested(self, post_id: str) -> bool:
        if not post_id: return False
        try:
            return len(self.collection.get(where={"post_id": post_id}, limit=1)['ids']) > 0
        except Exception as e:
            print(f"  Error checking post_id {post_id}: {e}", file=sys.stderr)
            return False

    def rerank_with_cross_encoder(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents: return []
        print(f"  Reranking {len(documents)} documents...")
        pairs = [[query, doc['document']] for doc in documents]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = score
        return sorted(documents, key=lambda x: x['rerank_score'], reverse=True)


# =====================================================================
# PART 5: FLASK WEB APPLICATION
# =====================================================================
app = Flask(__name__)
app.json.ensure_ascii = False
print("Initializing Flask application...")
rag_system = OptimizedRAGSystem()

@app.route('/query', methods=['GET'])
def handle_query():
    query_text = request.args.get('text')
    post_id = request.args.get('post_id')
    if not query_text:
        return render_template('response.html', error="Le parametre 'text' est requis.")
    
    # 1. Hybrid Search Phase
    # -----------------------
    if rag_system.collection.count() == 0:
        return render_template('response.html', error="La base de donnees est vide.")

    # Vector Search
    q_emb = rag_system.embedding_model.encode([query_text]).tolist()
    vector_results = rag_system.collection.query(
        query_embeddings=q_emb, 
        n_results=50, 
        where={"post_id": post_id} if post_id else None
    )
    # ChromaDB distances are similarity scores where lower is better, like FTS5 rank
    vector_ranked_list = list(zip(vector_results['ids'][0], vector_results['distances'][0]))

    # Keyword Search (FTS5)
    fts_ranked_list = rag_system.fts_search.search(query_text, limit=50, post_id=post_id)

    # 2. Reciprocal Rank Fusion (RRF)
    # --------------------------------
    fused_results = reciprocal_rank_fusion([vector_ranked_list, fts_ranked_list])
    if not fused_results:
        return render_template('response.html', error="Aucune information pertinente n'a ete trouvee.")
        
    # Get top 50 candidate IDs from the superior fused ranking
    candidate_ids = [doc_id for doc_id, score in fused_results[:50]]
    candidate_docs_data = rag_system.collection.get(ids=candidate_ids, include=["documents", "metadatas"])
    candidates = [{'id': id_val, 'document': doc, 'metadata': meta} for id_val, doc, meta in zip(candidate_docs_data['ids'], candidate_docs_data['documents'], candidate_docs_data['metadatas'])]

    # 3. Reranking Phase
    # -------------------
    reranked_micro_chunks = rag_system.rerank_with_cross_encoder(query_text, candidates)
    
    # 4. Context Assembly with Highlighting
    # -------------------------------------
    unique_parent_chunks = {}
    # The top result after reranking is our most important chunk
    best_micro_chunk = reranked_micro_chunks[0] if reranked_micro_chunks else None

    for chunk in reranked_micro_chunks:
        if len(unique_parent_chunks) >= CONTEXT_FOR_LLM_COUNT: break
        parent = chunk.get('metadata', {}).get('parent_chunk')
        if parent and parent not in unique_parent_chunks:
            # Spotlight the best micro-chunk within its parent context
            if best_micro_chunk and parent == best_micro_chunk['metadata']['parent_chunk']:
                highlighted_parent = parent.replace(
                    best_micro_chunk['document'], 
                    f"<exact_match>{best_micro_chunk['document']}</exact_match>"
                )
                unique_parent_chunks[parent] = {'text': highlighted_parent, 'meta': chunk['metadata']}
            else:
                unique_parent_chunks[parent] = {'text': parent, 'meta': chunk['metadata']}
    
    context_docs = [item['text'] for item in unique_parent_chunks.values()]
    
    # 5. Generation Phase
    # -------------------
    answer = rag_system.generate_llm_response(query_text, context_docs)
    
    first_meta = list(unique_parent_chunks.values())[0]['meta'] if unique_parent_chunks else {}
    filename = first_meta.get('filename', 'N/A')
    category = first_meta.get('category', 'Non specifiee')
    answer_html = html.escape(answer.strip()).replace('\n', '<br>')
    
    return render_template('response.html', answer_html=answer_html, filename=filename, category=category)


@app.route('/stats', methods=['GET'])
def handle_stats():
    return jsonify({"total_micro_chunks": rag_system.collection.count()})

@app.route('/', methods=['GET'])
def handle_root():
    return jsonify({
        "message": "Bienvenue sur l'application RAG optimisee pour serveur",
        "endpoints": {"query": "/query?text=<votre_question>", "stats": "/stats"}
    })

# =====================================================================
# PART 6: WORDPRESS INGESTION SCRIPT
# =====================================================================
def run_wordpress_ingestion():
    print("\n--- Starting WordPress Post Ingestion (Optimized Mode) ---")
    api_endpoint = f"{WORDPRESS_URL}/wp-json/wp/v2/posts"
    page, total_posts, total_chunks, skipped = 1, 0, 0, 0
    while True:
        print(f"\n--- Fetching page {page} of articles... ---")
        params = {'page': page, 'per_page': WP_INGESTION_BATCH_SIZE, 'status': 'publish', '_embed': 'wp:term'}
        try:
            response = http.get(api_endpoint, auth=(WORDPRESS_USER, WORDPRESS_APP_PASSWORD), params=params, timeout=45)
            response.raise_for_status()
            posts_on_page = response.json()
            if not posts_on_page:
                print("--- No more posts found. Ingestion finished. ---")
                break
            
            print(f"Processing {len(posts_on_page)} posts from page {page}...")
            for post in posts_on_page:
                total_posts += 1
                post_id = str(post.get('id', ''))
                title = post.get('title', {}).get('rendered', '').strip()
                print(f"({total_posts}) Checking: '{title}' (ID: {post_id})")
                
                if rag_system.is_post_id_ingested(post_id):
                    print("  -> Already ingested. Skipping.")
                    skipped += 1
                    continue
                
                raw_content = post.get('content', {}).get('rendered', '')
                cleaned_content = clean_html(raw_content)
                if not cleaned_content and not title:
                    print("  -> No content or title. Skipping.")
                    continue
                
                source_url = post.get('link', '')
                categories = [term['name'] for term_list in post.get('_embedded', {}).get('wp:term', []) for term in term_list if term.get('taxonomy') == 'category']
                category_str = ", ".join(categories) or "Uncategorized"
                
                full_text = f"Titre: {title}\nURL: {source_url}\nCategories: {category_str}\n\n{cleaned_content}"
                chunks_added = rag_system.add_document(full_text, source_url, post_id, category_str)
                total_chunks += chunks_added
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"  -> WordPress API Error: {e}. Stopping ingestion.")
            break
    
    print("\n--- INGESTION SUMMARY ---")
    print(f"Total posts processed: {total_posts}")
    print(f"Posts skipped (already present): {skipped}")
    print(f"New micro-chunks added: {total_chunks}")
    print(f"Total micro-chunks in database: {rag_system.collection.count()}")

def ingest_single_wordpress_post(post_id: str):
    """Fetches, processes, and ingests a single WordPress post by its ID."""
    print(f"\n--- Starting single post ingestion for ID: {post_id} ---")

    if rag_system.is_post_id_ingested(post_id):
        print(f"-> Post ID {post_id} is already ingested. Skipping.")
        return

    api_endpoint = f"{WORDPRESS_URL}/wp-json/wp/v2/posts/{post_id}"
    params = {'_embed': 'wp:term'}

    try:
        response = http.get(api_endpoint, auth=(WORDPRESS_USER, WORDPRESS_APP_PASSWORD), params=params, timeout=30)
        response.raise_for_status()
        post = response.json()

        title = post.get('title', {}).get('rendered', '').strip()
        print(f"Processing: '{title}' (ID: {post_id})")

        raw_content = post.get('content', {}).get('rendered', '')
        cleaned_content = clean_html(raw_content)

        if not cleaned_content and not title:
            print("-> No content or title found for this post. Skipping.")
            return

        source_url = post.get('link', '')
        categories = [
            term['name'] for term_list in post.get('_embedded', {}).get('wp:term', [])
            for term in term_list if term.get('taxonomy') == 'category'
        ]
        category_str = ", ".join(categories) or "Uncategorized"

        full_text = f"Titre: {title}\nURL: {source_url}\nCategories: {category_str}\n\n{cleaned_content}"
        chunks_added = rag_system.add_document(full_text, source_url, str(post_id), category_str)

        print(f"-> Successfully added {chunks_added} micro-chunks for post ID {post_id}.")
        print(f"Total micro-chunks in database: {rag_system.collection.count()}")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"-> Error: Post with ID {post_id} not found.")
        else:
            print(f"-> WordPress API HTTP Error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"-> WordPress API Connection Error: {e}")


# =====================================================================
# PART 7: MAIN ENTRY POINT
# =====================================================================
if __name__ == '__main__':
    script_name = sys.argv[0]
    if len(sys.argv) > 1 and sys.argv[1] == 'wp':
        run_wordpress_ingestion()
    elif len(sys.argv) > 2 and sys.argv[1] == 'wp-single':
        post_id_to_ingest = sys.argv[2]
        ingest_single_wordpress_post(post_id_to_ingest)
    else:
        print(f"Starting Flask server at http://0.0.0.0:5011")
        print(f"To start full WordPress ingestion, run: python3 {script_name} wp")
        print(f"To ingest a single post by ID, run: python3 {script_name} wp-single <POST_ID>")

        app.run(host='0.0.0.0', port=5011, debug=False)
