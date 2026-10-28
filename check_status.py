"""
Status Check Script for GraphRAG Pipeline
Run this to check the progress of data processing
"""
import os
from pathlib import Path
import json

def check_data_status():
    """Check the status of data processing"""
    
    # Use __file__ so the script works when run from any directory,
    # not just from the project root.
    data_dir = Path(__file__).parent / "data"
    
    # Check cleaned documents
    cleaned_dir = data_dir / "cleaned"
    cleaned_count = len(list(cleaned_dir.glob("*.json"))) if cleaned_dir.exists() else 0
    
    # Check chunks
    chunks_dir = data_dir / "chunks"
    chunk_files = list(chunks_dir.glob("*_chunks.json")) if chunks_dir.exists() else []
    chunk_count = len(chunk_files)
    
    # Count total chunks
    total_chunks = 0
    if chunk_files:
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    total_chunks += len(chunks)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  Warning: could not read {chunk_file.name}: {e}")
    
    # Check embeddings
    embeddings_dir = data_dir / "embeddings"
    embedding_count = len(list(embeddings_dir.glob("*.json"))) if embeddings_dir.exists() else 0
    
    print("\n" + "="*60)
    print("GraphRAG Pipeline Status")
    print("="*60)
    
    print(f"\nPhase 1 - Data Cleaning:")
    print(f"  Documents cleaned: {cleaned_count}")
    
    print(f"\nPhase 2 - Chunking:")
    print(f"  Documents chunked: {chunk_count}/{cleaned_count}")
    print(f"  Total chunks created: {total_chunks:,}")
    if cleaned_count > 0:
        progress = (chunk_count / cleaned_count) * 100
        print(f"  Progress: {progress:.1f}%")
    
    print(f"\nPhase 3 - Embeddings:")
    print(f"  Documents embedded: {embedding_count}/{chunk_count}")
    if chunk_count > 0:
        progress = (embedding_count / chunk_count) * 100
        print(f"  Progress: {progress:.1f}%")
    
    print("\n" + "="*60)
    
    # Determine next step
    if chunk_count < cleaned_count:
        print("\nNext: Wait for chunking to complete")
        print(f"   Estimated: {cleaned_count - chunk_count} documents remaining")
    elif embedding_count < chunk_count:
        print("\nNext: Run embedding generation")
        print("   Command: python embedding_3.py")
    else:
        print("\nAll phases complete! System ready for queries.")
        print("   Start web interface: python app.py")
        print("   Or use CLI: python main.py -i")
    
    print()

if __name__ == "__main__":
    check_data_status()
