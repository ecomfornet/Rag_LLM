# GraphRAG Project Migration Guide

This package contains a full snapshot of the GraphRAG project, including source code, processed data, and native database dumps.

## Restoration Steps

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   # Update .env with your keys
   ```

2. **Start Infrastructure**:
   ```bash
   docker-compose up -d
   ```

3. **Restore Databases**:
   The easiest way is to use the updated `import_databases.py`:
   ```bash
   python import_databases.py --input ./
   ```
   
   This script will:
   - Upload the `legal_docs_graphrag.snapshot` to your new Qdrant instance.
   - Use `neo4j-admin database load` to restore the Neo4j dump.

4. **Verify & Run**:
   - Check status: `python check_status.py`
   - Start UI: `streamlit run app.py`
