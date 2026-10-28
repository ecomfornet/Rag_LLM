# GraphRAG Legal Assistant — Setup Guide

> **Goal**: Re-import this exported project on any machine and run it with the
> exact same configuration and port numbers as the original system.

---

## Architecture Overview

| Component | Role | Host Port | Container Port |
|-----------|------|-----------|----------------|
| **Neo4j** | Graph database (legal relationships) | `7475` (HTTP UI) / `7688` (Bolt) | 7474 / 7687 |
| **Qdrant** | Vector database (embeddings) | `6340` (HTTP) / `6341` (gRPC) | 6333 / 6334 |
| **Redis** | Multi-level cache (L2) | `6380` | 6379 |
| **Flask API** | GraphRAG backend | `5000` | — |

---

## Prerequisites

Before you start, make sure the following are installed on your machine:

| Tool | Version | Check command |
|------|---------|---------------|
| Python | ≥ 3.10 | `python --version` |
| Docker Desktop | latest | `docker --version` |
| Docker Compose | v2+ (bundled with Docker Desktop) | `docker compose version` |
| Git | any | `git --version` |

---

## Step 1 — Copy the Project Files

Place all exported files into a single directory. The structure must be:

```
graphrag_export/
├── .env
├── app.py
├── main.py
├── config.py
├── requirements.txt
├── docker-compose.yml
├── data_cleaning_1.py
├── chunking_2.py
├── embedding_3.py
├── retrieval_4.py
├── generation_5.py
├── caching_6.py
├── graph_builder.py
├── neo4j.dump                          ← Neo4j database backup
├── qdrant_legal_docs_graphrag.snapshot ← Qdrant snapshot
├── redis_dump.rdb                      ← Redis cache dump
└── data/                               ← Raw document corpus
```

---

## Step 2 — Configure the `.env` File

Open `.env` and update **only** the values marked with `← EDIT`:

```ini
# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-proj-YOUR_OPENAI_KEY_HERE    # ← EDIT: your real API key

# ── WordPress (optional, only needed for content publishing) ──────────────────
WORDPRESS_API_URL=https://kytisa.com/wp-json/wp/v2/posts
WORDPRESS_USERNAME=agent
WORDPRESS_PASSWORD=ntbq v9FI kZu5 5wmT VUKC CTI5

# ── Neo4j (keep exactly as-is to match docker-compose ports) ─────────────────
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphrag123

# ── Qdrant ───────────────────────────────────────────────────────────────────
QDRANT_HOST=localhost
QDRANT_PORT=6340
QDRANT_USE_HTTPS=false
QDRANT_API_KEY=

# ── Redis ────────────────────────────────────────────────────────────────────
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_DB=0
REDIS_PASSWORD=

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
```

> **Important**: Do **not** change the port numbers unless you also update
> `docker-compose.yml` — they must stay in sync.

---

## Step 3 — Start Docker Services

Open a terminal in the project directory and run:

```bash
docker compose up -d
```

Verify all three containers are running:

```bash
docker compose ps
```

Expected output:

```
NAME                STATUS
graphrag_neo4j      Up
graphrag_qdrant     Up
graphrag_redis      Up
```

Wait ~15 seconds for Neo4j to finish its startup sequence.

---

## Step 4 — Restore the Neo4j Database

The project ships with `neo4j.dump` (full backup of the legal knowledge graph).

> **Important**: You must stop the **entire container** before loading — not
> just the Neo4j server process. Running `docker exec ... neo4j stop` is not
> enough; neo4j-admin will still report "The database is in use". The correct
> approach is to stop the container and load via a temporary offline container.

**Windows (PowerShell) — run from the project folder:**

```powershell
# 1. Check your exact volume name (look for the one containing "neo4j_data")
docker volume ls | findstr neo4j

# 2. Stop the container completely
docker compose stop neo4j

# 3. Load the dump using a temporary offline container
#    Replace 'graphrag_export_neo4j_data' below with the name from step 1
docker run --rm `
  -v graphrag_export_neo4j_data:/data `
  -v "${PWD}\neo4j.dump:/import/neo4j.dump" `
  neo4j:5.14-community `
  neo4j-admin database load neo4j --from-path=/import --overwrite-destination=true

# 4. Start Neo4j again
docker compose start neo4j
```

**Linux / macOS / Git Bash — same logic:**

```bash
# 1. Check the volume name
docker volume ls | grep neo4j

# 2. Stop container
docker compose stop neo4j

# 3. Load dump offline
docker run --rm \
  -v graphrag_export_neo4j_data:/data \
  -v "$(pwd)/neo4j.dump:/import/neo4j.dump" \
  neo4j:5.14-community \
  neo4j-admin database load neo4j --from-path=/import --overwrite-destination=true

# 4. Start again
docker compose start neo4j
```

**Verify restore** — open the Neo4j browser at http://localhost:7475
- Username: `neo4j`  |  Password: `graphrag123`
- Run: `MATCH (n) RETURN count(n)` — should return > 0

---

## Step 5 — Restore the Qdrant Snapshot

The project ships with `qdrant_legal_docs_graphrag.snapshot` (~1.4 GB).
This restores all document embeddings (text-embedding-3-large, 3072 dims).

```bash
# Upload the snapshot file to Qdrant
curl -X POST "http://localhost:6340/collections/legal_docs_graphrag/snapshots/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@qdrant_legal_docs_graphrag.snapshot"
```

**Verify restore** — check collection info:

```bash
curl http://localhost:6340/collections/legal_docs_graphrag
```

The response should include `"status": "green"` and a non-zero `vectors_count`.

> **On Windows (PowerShell)** use:
> ```powershell
> Invoke-RestMethod -Method Post `
>   -Uri "http://localhost:6340/collections/legal_docs_graphrag/snapshots/upload" `
>   -Form @{ snapshot = Get-Item ".\qdrant_legal_docs_graphrag.snapshot" }
> ```

---

## Step 6 — Restore Redis Cache (Optional)

Redis cache speeds up repeated queries. The dump is small and optional —
the system works without it (cache will rebuild automatically).

```bash
# Copy the RDB file into the container's data volume
docker cp redis_dump.rdb graphrag_redis:/data/dump.rdb

# Restart Redis to load the dump
docker compose restart redis
```

---

## Step 7 — Install Python Dependencies

Create and activate a virtual environment, then install all packages:

```bash
# Create a virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

> **Note**: `faiss-cpu` requires Microsoft Visual C++ Build Tools on Windows.  
> Install from https://visualstudio.microsoft.com/visual-cpp-build-tools/ if
> the installation fails.

---

## Step 8 — Run the Flask Application

With the virtual environment active and all Docker services running:

```bash
python app.py
```

You should see:

```
INFO  Initializing GraphRAG Pipeline...
INFO  GraphRAG Pipeline initialized successfully
INFO  Starting Flask application...
 * Running on http://0.0.0.0:5000
```

Open your browser at **http://localhost:5000**

---

## Step 9 — Test the System

### Web UI
Go to http://localhost:5000 and type a legal question, for example:
> *Quelles sont les obligations d'une SARL en Guinée?*

### REST API
```bash
curl "http://localhost:5000/api/query?query=Quelles+sont+les+obligations+d%27une+SARL"
```

### Health Check
```bash
curl http://localhost:5000/health
```
Expected: `{"status": "healthy", "pipeline_initialized": true}`

### Cache Statistics
```bash
curl http://localhost:5000/stats
```

---

## (Optional) Re-index From Scratch

If you want to rebuild the indexes from the raw documents in `data/` instead
of restoring the snapshots, run the pipeline steps in order:

```bash
python data_cleaning_1.py   # Clean and normalize raw documents
python chunking_2.py        # Split documents into chunks
python embedding_3.py       # Generate embeddings → upload to Qdrant
python graph_builder.py     # Extract citations → build Neo4j graph
```

> This will consume OpenAI API credits (embedding + LLM calls).
> Estimated cost: depends on corpus size. Check `benchmark.py` first.

---

## Port Reference Summary

| Service | UI / API URL | Credentials |
|---------|-------------|-------------|
| Flask Web App | http://localhost:5000 | — |
| Neo4j Browser | http://localhost:7475 | neo4j / graphrag123 |
| Qdrant Dashboard | http://localhost:6340/dashboard | — |
| Redis | localhost:6380 | no password |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `OPENAI_API_KEY is not set` | Edit `.env` and set your real key, then restart flask |
| Neo4j connection refused | Wait 30s after `docker compose up -d` (slow startup) |
| Qdrant snapshot upload fails | Make sure the container is fully up; retry the `curl` command |
| `faiss-cpu` install error on Windows | Install Visual C++ Build Tools (see Step 7) |
| Port already in use | Stop conflicting services or change the host port in `docker-compose.yml` AND `.env` |
| Empty Neo4j graph after restore | Re-run Step 4; check dump file exists and is ~138 MB: `dir neo4j.dump` (Windows) or `ls -lh neo4j.dump` (Linux/macOS) |
| Neo4j restore "database is in use" | Use the **full container stop + offline container** method in Step 4 — do not use `docker exec ... neo4j stop` |

---

## Stopping the System

```bash
# Stop Flask: Ctrl+C in the terminal where app.py is running

# Stop and remove Docker containers (data volumes are preserved)
docker compose down

# Stop AND delete all data volumes (full reset)
docker compose down -v
```


stop the server 
Stop-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess -Force

