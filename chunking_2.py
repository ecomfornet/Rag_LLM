"""
Chunking Module - Graph-Aware
Preserves citations and creates chunk nodes in Neo4j
"""
import logging
import re
import json
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

import config
from graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


class GraphAwareChunker:
    """Chunks documents while preserving citations and graph structure"""
    
    def __init__(self):
        self.graph_builder = GraphBuilder()
        self.input_dir = config.DATA_DIR / "cleaned"
        self.output_dir = config.DATA_DIR / "chunks"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info("GraphAwareChunker initialized")
    
    def detect_citation_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """
        Detect citation spans in text to avoid truncation
        
        Args:
            text: Document text
            
        Returns:
            List of (start, end) positions for citations
        """
        boundaries = []
        
        # FIX: Expanded patterns — previous set missed the most common corpus
        # forms like bare "Article 15" or "Loi L/2023/015", so citations was
        # almost always [], is_citation_truncated() always returned False, and
        # citations were silently sliced at chunk boundaries.
        citation_patterns = [
            # ── Bare references (most frequent in corpus) ─────────────────────
            r'article\s+\d+[\w\-\.]*(?:\s+(?:al\.?|alinéa)\s+\d+)?',
            r'(?:loi|décret|ordonnance|arrêté)\s+(?:n[°o\.]*\s*)?[A-Z0-9][A-Z0-9/\-\.]+',
            r'N°\s*\d+[/\-]\d+[A-Z0-9/\-\.]*',
            # ── Sentence forms ────────────────────────────────────────────────
            r'selon\s+l[\'"\u2019]article\s+\d+[^.]*\.',
            r'conform[eé]ment\s+à\s+la\s+loi\s+[A-Z0-9/\-]+[^.]*\.',
            r'en\s+application\s+de\s+l[\'"\u2019]article\s+\d+[^.]*\.',
            r'article\s+\d+\s+de\s+la\s+loi\s+[A-Z0-9/\-]+',
        ]
        
        for pattern in citation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                boundaries.append((match.start(), match.end()))
        
        # Merge overlapping/adjacent spans so one citation isn't counted twice
        boundaries.sort(key=lambda x: x[0])
        merged: List[Tuple[int, int]] = []
        for start, end in boundaries:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        return merged
    
    def is_citation_truncated(self, chunk_start: int, chunk_end: int, citations: List[Tuple[int, int]]) -> bool:
        """
        Check if a chunk boundary would truncate a citation
        
        Args:
            chunk_start: Chunk start position
            chunk_end: Chunk end position
            citations: List of citation boundaries
            
        Returns:
            True if citation would be truncated
        """
        for cite_start, cite_end in citations:
            # Citation starts in chunk but doesn't end in chunk
            if chunk_start <= cite_start < chunk_end < cite_end:
                return True
            # Citation ends in chunk but doesn't start in chunk
            if cite_start < chunk_start < cite_end <= chunk_end:
                return True
        
        return False
    
    def extend_chunk_for_citation(
        self,
        chunk_end: int,
        text: str,
        citations: List[Tuple[int, int]]
    ) -> int:
        """
        Extend chunk boundary to include complete citation
        
        Args:
            chunk_end: Current chunk end position
            text: Full document text
            citations: List of citation boundaries
            
        Returns:
            New chunk end position
        """
        max_extend = min(chunk_end + config.CITATION_EXTEND_CHARS, len(text))
        
        for cite_start, cite_end in citations:
            # If citation starts before chunk_end and extends beyond
            if cite_start < chunk_end < cite_end <= max_extend:
                chunk_end = cite_end
        
        return chunk_end
    
    def chunk_text(self, text: str, doc_id: str, doc_metadata: Dict = None) -> List[Dict]:
        """
        Chunk text while preserving citations
        
        Args:
            text: Document text
            doc_id: Document ID
            doc_metadata: Document metadata from data_cleaning_1.py
                          (title, type, reference, year, institution,
                           source_url, date, modified)
            
        Returns:
            List of chunk dictionaries, each carrying the document metadata
        """
        chunks = []
        
        # Normalise metadata so we always have a dict to spread
        if doc_metadata is None:
            doc_metadata = {}
        
        # Fields propagated from data_cleaning_1.py → extract_metadata()
        # FIX: use 'doc_title' (not 'title') so the key matches what
        # generation_5.py and app.py read when building the sources list.
        inherited_meta = {
            'doc_title':   doc_metadata.get('title', '') or doc_metadata.get('doc_title', ''),
            'title':       doc_metadata.get('title', '') or doc_metadata.get('doc_title', ''),
            'type':        doc_metadata.get('type', ''),
            'reference':   doc_metadata.get('reference', ''),
            'year':        doc_metadata.get('year', None),
            'institution': doc_metadata.get('institution', ''),
            'source_url':  doc_metadata.get('source_url', ''),
            'date':        doc_metadata.get('date', ''),
            'modified':    doc_metadata.get('modified', ''),
        }
        
        # Detect citations
        citations = self.detect_citation_boundaries(text) if config.PRESERVE_CITATIONS else []
        
        # Simple recursive character splitter
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(text):
            # 1. Determine initial chunk end based on size
            chunk_end = min(current_pos + config.CHUNK_SIZE, len(text))
            
            # 2. Try to break at a logical separator (backwards from chunk_end)
            if chunk_end < len(text):
                search_back_limit = max(current_pos, chunk_end - int(config.CHUNK_SIZE * 0.3))
                search_text = text[search_back_limit:chunk_end]
                
                best_sep_pos = -1
                for separator in config.CHUNK_SEPARATORS:
                    if not separator: continue
                    sep_pos = search_text.rfind(separator)
                    if sep_pos != -1:
                        best_sep_pos = search_back_limit + sep_pos + len(separator)
                        break
                
                if best_sep_pos != -1:
                    chunk_end = best_sep_pos
                else:
                    # Fallback: break at last whitespace to avoid mid-word split
                    last_space = text[current_pos:chunk_end].rfind(' ')
                    if last_space != -1 and current_pos + last_space > current_pos:
                        chunk_end = current_pos + last_space + 1

            # 3. Check for citation truncation (forward extension)
            if config.PRESERVE_CITATIONS and self.is_citation_truncated(current_pos, chunk_end, citations):
                chunk_end = self.extend_chunk_for_citation(chunk_end, text, citations)
            
            # 4. Extract and store chunk
            chunk_text = text[current_pos:chunk_end].strip()
            
            if len(chunk_text) >= config.MIN_CHUNK_LENGTH:
                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                chunks.append({
                    'chunk_id':    chunk_id,
                    'doc_id':      doc_id,
                    'content':     chunk_text,
                    'chunk_index': chunk_index,
                    'start_char':  current_pos,
                    'end_char':    chunk_end,
                    **inherited_meta,
                })
                chunk_index += 1
            
            # 5. Move to next position with overlap logic
            if chunk_end >= len(text):
                # We've reached the end! Breaking prevents the degenerate loop.
                break
                
            # Attempt to start the next chunk at an overlap point, 
            # aligned to the nearest preceding whitespace
            next_pos_ideal = chunk_end - config.CHUNK_OVERLAP
            search_start = max(current_pos + 1, next_pos_ideal - 50)
            search_text = text[search_start:chunk_end]
            
            last_space_in_overlap = search_text.rfind(' ')
            if last_space_in_overlap != -1:
                current_pos = search_start + last_space_in_overlap + 1
            else:
                current_pos = max(current_pos + 1, next_pos_ideal)
        
        logger.debug(f"Created {len(chunks)} chunks for document {doc_id}")
        return chunks
    
    def extract_chunk_citations(self, chunk_text: str) -> List[str]:
        """
        Extract citation references from chunk text
        
        Args:
            chunk_text: Chunk content
            
        Returns:
            List of citation references
        """
        citations = []
        
        for pattern in config.CITATION_PATTERNS:
            matches = re.finditer(pattern, chunk_text, re.IGNORECASE)
            for match in matches:
                citations.append(match.group(0))
        
        return citations
    
    def create_chunk_graph_nodes(self, chunks: List[Dict]) -> int:
        """
        Create chunk nodes in Neo4j and link to documents.
        MENTIONS relationships are written in a single batched UNWIND query
        instead of one session per citation (previously O(citations) round-trips).
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Number of chunks created
        """
        created = 0
        # Collect all resolved MENTIONS relationships first, then write in one batch
        mentions_batch = []  # List of {chunk_id, target_doc_id, citation_ref}

        for chunk in chunks:
            # Create chunk node
            self.graph_builder.create_chunk_node(chunk)

            # Extract citations and resolve them to document IDs
            citations = self.extract_chunk_citations(chunk['content'])
            for citation_ref in citations:
                target_doc_id = self.graph_builder.resolve_citation_to_doc_id(citation_ref)
                if target_doc_id:
                    mentions_batch.append({
                        'chunk_id':     chunk['chunk_id'],
                        'target_doc_id': target_doc_id,
                        'citation_ref': citation_ref,
                    })

            created += 1

        # Batch-write all MENTIONS relationships in a single Cypher UNWIND call
        if mentions_batch:
            batch_query = """
            UNWIND $rows AS row
            MATCH (c:Chunk    {id: row.chunk_id})
            MATCH (d:Document {id: row.target_doc_id})
            MERGE (c)-[:MENTIONS {reference: row.citation_ref}]->(d)
            """
            try:
                with self.graph_builder.driver.session(database=config.NEO4J_DATABASE) as session:
                    session.run(batch_query, rows=mentions_batch)
                logger.debug(
                    f"Batch-created {len(mentions_batch)} MENTIONS relationships "
                    f"across {created} chunks"
                )
            except Exception as e:
                logger.error(f"Error batch-creating MENTIONS relationships: {e}")

        return created
    
    def process_document(self, doc_file: Path) -> List[Dict]:
        """
        Process a single document: chunk and create graph nodes
        
        Args:
            doc_file: Path to document JSON file
            
        Returns:
            List of chunks or empty list if already processed
        """
        # Load document
        with open(doc_file, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        
        doc_id = doc['id']
        
        # Check if chunks already exist in Neo4j
        existing_chunks_count = self.graph_builder.get_document_chunks_count(doc_id)
        output_file = self.output_dir / f"{doc_id}_chunks.json"
        
        if existing_chunks_count > 0 and output_file.exists():
            logger.info(f"Document {doc_id} already has {existing_chunks_count} chunks, skipping...")
            return []
        
        content = doc['content']
        
        # Build metadata dict matching data_cleaning_1.py → extract_metadata() output
        doc_metadata = {
            'title':       doc.get('title', ''),
            'type':        doc.get('type', ''),
            'reference':   doc.get('reference', ''),
            'year':        doc.get('year', None),
            'institution': doc.get('institution', ''),
            'source_url':  doc.get('source_url', ''),
            'date':        doc.get('date', ''),
            'modified':    doc.get('modified', ''),
        }
        
        # Chunk text, propagating document metadata into every chunk
        chunks = self.chunk_text(content, doc_id, doc_metadata)
        
        # Create graph nodes
        self.create_chunk_graph_nodes(chunks)
        
        # Save chunks
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Processed {len(chunks)} chunks for {doc_id}")
        
        return chunks
    
    def process_all_documents(self) -> Dict:
        """
        Process all cleaned documents
        
        Returns:
            Statistics dictionary
        """
        doc_files = list(self.input_dir.glob("*.json"))
        
        logger.info(f"Processing {len(doc_files)} documents...")
        
        total_chunks = 0
        skipped_docs = 0
        processed_docs = 0
        
        for doc_file in tqdm(doc_files, desc="Chunking documents"):
            chunks = self.process_document(doc_file)
            if chunks:
                total_chunks += len(chunks)
                processed_docs += 1
            else:
                skipped_docs += 1
        
        stats = {
            'total_documents': len(doc_files),
            'processed_documents': processed_docs,
            'skipped_documents': skipped_docs,
            'total_chunks': total_chunks,
            'avg_chunks_per_doc': total_chunks / processed_docs if processed_docs else 0
        }
        
        logger.info(f"Chunking complete: {stats}")
        
        return stats
    
    def close(self):
        """Close connections"""
        self.graph_builder.close()


def main():
    """Main execution"""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT
    )
    
    chunker = GraphAwareChunker()
    
    try:
        stats = chunker.process_all_documents()
        
        print(f"\n✓ Chunking complete")
        print(f"  Documents processed: {stats['processed_documents']}/{stats['total_documents']}")
        print(f"  Documents skipped: {stats['skipped_documents']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Avg chunks/doc: {stats['avg_chunks_per_doc']:.1f}")
        print(f"  Output directory: {chunker.output_dir}")
        
    finally:
        chunker.close()


if __name__ == "__main__":
    main()
