# GraphRAG Project Export & Import Guide (Native Migration)

This guide describes how to migrate your entire GraphRAG project from one environment to another using native database snapshots and dumps.

## Overview

The migration process uses a unified export script that packages:
1.  **Source Code**: All `.py` files and configuration.
2.  **Data**: The entire `data/` directory (cleaned files, chunks, embeddings).
3.  **Databases**: 
    - **Qdrant**: Native `.snapshot` (via REST API).
    - **Neo4j**: Native `.dump` (via `neo4j-admin`).
    - **Redis**: JSON dump of all keys.

---

## 🚀 Part 1: Exporting the Project

Run the unified export script from the root of your project:

```bash
python export_project.py
```

### What happens:
- A new folder `exports/graphrag_full_migration_YYYYMMDD_HHMMSS` is created.
- Native snapshots of Qdrant and Neo4j are triggered and downloaded into `db/`.
- All source files and the `data/` directory are copied.
- A final `.zip` file is created for easy transfer.

---

## 📥 Part 2: Importing into a New Environment

### 1. Preparation
Transfer the `.zip` file to the new machine and extract it.

### 2. Infrastructure Setup
Ensure Docker is running and start the database containers. 

> [!IMPORTANT]
> Verify that the container names in `docker-compose.yml` exactly match those expected by the import script (`graphrag_neo4j`, `graphrag_qdrant`, `graphrag_redis`).

```bash
docker-compose up -d
```

### 3. Environment Configuration
Create a `.env` file from the example or migration defaults. You **must** provide your own `OPENAI_API_KEY` for the system to function.

### 4. Restore Databases
Use the native restore script:

```bash
python import_databases.py --input ./
```

### 5. Verification
Run the status check and diagnostic tools to ensure everything is connected and data is present:

```bash
python check_status.py
python test_system.py
```

---

## 🛠 Troubleshooting

### Neo4j Restore Issues
If `import_databases.py` fails to load the Neo4j dump:
- Ensure the container named `graphrag_neo4j` is running and healthy.
- Check for name conflicts: `docker rm -f graphrag_neo4j` if an old container is blocking the name.

### Qdrant Connection Errors
If you see `ConnectionRefusedError` (WinError 10061):
- Verify Docker containers are actually running: `docker ps`.
- Check that `QDRANT_PORT` in `.env` (default 6340) matches the host port in `docker-compose.yml`.

### Windows Console Encoding
If scripts like `check_status.py` crash with `UnicodeEncodeError`:
- Use `chcp 65001` in your powershell/terminal to enable UTF-8 support.
- The latest version of the scripts has been optimized to use plain ASCII markers for better compatibility.
