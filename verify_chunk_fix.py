
import json
from pathlib import Path
from chunking_2 import GraphAwareChunker

def verify_fix():
    chunker = GraphAwareChunker()
    doc_path = Path("data/cleaned/10188.json")
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    
    print(f"Document Length: {len(doc['content'])}")
    
    chunks = chunker.chunk_text(doc['content'], doc['id'], doc)
    
    print(f"Total Chunks Generated: {len(chunks)}")
    
    for i, c in enumerate(chunks):
        print(f"Chunk {i}: Start={c['start_char']}, End={c['end_char']}, Length={len(c['content'])}")
        # Check for mid-word split (heuristic)
        snippet = c['content'][:20]
        print(f"  Snippet: {snippet}...")

if __name__ == "__main__":
    verify_fix()
