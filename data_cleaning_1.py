"""
Data Cleaning Module - GraphRAG Enhanced
Extracts documents from WordPress and builds knowledge graph
"""
import logging
import re
import json
from typing import List, Dict, Optional
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import config
from graph_builder import GraphBuilder
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class DocumentCleaner:
    """Cleans and processes documents from WordPress"""
    
    def __init__(self):
        self.graph_builder = GraphBuilder()
        self.output_dir = config.DATA_DIR / "cleaned"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info("DocumentCleaner initialized")
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True
    )
    def _fetch_page(self, page: int, per_page: int) -> requests.Response:
        """Fetch a single page with retries on network errors"""
        return requests.get(
            config.WORDPRESS_API_URL,
            params={'page': page, 'per_page': per_page},
            timeout=30
        )

    def fetch_wordpress_posts(self, max_posts: int = None) -> List[Dict]:
        """
        Fetch posts from WordPress API
        
        Args:
            max_posts: Maximum number of posts to fetch (None for all)
            
        Returns:
            List of post dictionaries
        """
        posts = []
        page = 1
        per_page = 100
        
        logger.info(f"Fetching posts from {config.WORDPRESS_API_URL} (max_posts={max_posts or 'all'})")
        
        while True:
            try:
                response = self._fetch_page(page, per_page)
                response.raise_for_status()
                
                batch = response.json()
                if not batch:
                    break
                
                posts.extend(batch)
                logger.info(f"Fetched page {page}: {len(batch)} posts")
                
                if max_posts and len(posts) >= max_posts:
                    posts = posts[:max_posts]
                    break
                
                page += 1
                
            except requests.exceptions.HTTPError as e:
                # Handle 400 Bad Request specifically (WordPress uses this for end of pagination)
                if e.response is not None and e.response.status_code == 400:
                    try:
                        error_data = e.response.json()
                        if error_data.get('code') == 'rest_post_invalid_page_number':
                            logger.info(f"End of pagination reached at page {page}")
                            break
                    except Exception:
                        pass
                
                logger.error(f"HTTP Error fetching page {page}: {e}")
                break  # Stop on hard HTTP errors other than invalid page
                
            except Exception as e:
                logger.error(f"Unexpected error fetching page {page}: {e}")
                break
        
        logger.info(f"Total posts fetched: {len(posts)}")
        return posts
    
    def clean_html(self, html_content: str) -> str:
        """
        Clean HTML content and extract text
        
        Args:
            html_content: Raw HTML
            
        Returns:
            Cleaned text
        """
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines()]
        text = '\n'.join(line for line in lines if line)
        
        return text
    
    def extract_metadata(self, post: Dict) -> Dict:
        """
        Extract metadata from WordPress post
        
        Args:
            post: WordPress post dictionary
            
        Returns:
            Metadata dictionary
        """
        # Determine document type from title/content
        title = post.get('title', {}).get('rendered', '')
        cleaned_title = self.clean_html(title)
        doc_type = self._classify_document_type(cleaned_title)
        
        # Extract reference from title
        reference = self._extract_reference(cleaned_title)
        
        # Extract year from title (prioritize title over WordPress date)
        year = self._extract_year_from_title(cleaned_title, reference)
        if not year:  # Fallback to WordPress post date only if not found in title
            if 'date' in post:
                try:
                    year = int(post['date'][:4])
                except (ValueError, TypeError):
                    pass
        
        # Extract institution — pass doc_type to avoid running _classify_document_type() twice
        institution = self._extract_institution(cleaned_title, doc_type=doc_type)
        
        metadata = {
            'id': str(post['id']),
            'title': cleaned_title,
            'type': doc_type,
            'reference': reference,
            'year': year,
            'institution': institution,
            'source_url': post.get('link', ''),
            'date': post.get('date', ''),
            'modified': post.get('modified', ''),
        }
        
        return metadata
    
    def _classify_document_type(self, title: str) -> str:
        """Classify document type from title with enhanced pattern matching"""
        title_lower = title.lower()
        
        # Court decisions - check first as they're most specific
        # CCJA: matches both 'ccja' shorthand and full 'Cour Commune de Justice et d'Arbitrage'
        if ('ccja' in title_lower or 'cour commune de justice' in title_lower) and (
                'arrêt' in title_lower or 'arret' in title_lower):
            return 'ccja_arret'
        elif ('cour suprême' in title_lower or 'cour supreme' in title_lower) and (
                'arrêt' in title_lower or 'arret' in title_lower):
            return 'cour_supreme'
        elif ("cour d'appel" in title_lower or "cour d appel" in title_lower) and (
                'arrêt' in title_lower or 'arret' in title_lower):
            return 'cour_appel'
        elif 'tribunal' in title_lower and 'jugement' in title_lower:
            return 'jugement_tribunal'
        elif 'jugement' in title_lower or 'judgment' in title_lower:
            return 'jurisprudence'
        
        # Administrative documents
        # Check keyword first so 'Décret D/2012/041' is not mistaken for 'decision'
        elif 'décret' in title_lower or 'decree' in title_lower:
            return 'décret'
        elif 'arrêté conjoint' in title_lower or re.search(r'\bAC/\d{4}/\d+', title):
            return 'arrete_conjoint'
        elif 'arrêté' in title_lower or re.search(r'\bA/\d{4}/\d+', title):
            return 'arrêté'
        elif re.search(r'\bD/\d{4}/\d+', title) or ('décision' in title_lower and re.search(r'D/\d{4}', title)):
            return 'decision'
        
        # Legislative documents
        elif 'loi' in title_lower or 'law' in title_lower or re.search(r'\bL/\d{4}/\d+', title):
            return 'loi'
        elif 'ordonnance' in title_lower:
            return 'ordonnance'
        elif 'code' in title_lower and any(w in title_lower for w in [
                'douanes', 'civil', 'pénal', 'penal', 'commerce', 'travail',
                'minier', 'investissement', 'procédure', 'procedure', 'militaire',
                'pétrolier', 'petrolier', 'justice']):
            return 'code'
        
        # Agreements and contracts
        elif 'convention' in title_lower:
            return 'convention'
        elif 'accord' in title_lower and 'entre' in title_lower:
            return 'accord'
        elif 'contrat' in title_lower or 'contract' in title_lower:
            return 'contrat'
        
        # Avis (opinions/advice)
        elif 'avis' in title_lower and ('consultatif' in title_lower or 'n°' in title_lower):
            return 'avis'
        
        else:
            return 'autre'
    
    def _extract_reference(self, title: str) -> str:
        """Extract legal reference from title with comprehensive patterns"""
        # Try to find patterns in order of specificity
        patterns = [
            r'[A-Z]{1,2}/\d{4}/\d+(?:/[A-Z]+(?:/[A-Z]+)*)?',  # A/2022/3354/MTFP/CAB or D/2024/219/PRG/CNRD/SGG
            r'[A-Z]/\d{4}/\d+',                                # L/2023/015
            r'N°\s*\d+(?:/\d{4})?',                             # N° 343 or N°033/2011
            r'n°\s*\d+(?:/\d{4})?',                             # n° 343 or n°033/2011
            r'NA°\s*\d+(?:/\d{4})?',                            # NA° 264 /2020 (encoded A)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title)
            if match:
                return match.group(0).strip()
        
        return ''
    
    def _extract_year_from_title(self, title: str, reference: str):
        """Extract year from document title or reference"""
        # NOTE: 're' is already imported at module level — no local import needed
        # Try to extract year from reference first (most reliable)
        if reference:
            # Look for 4-digit year in reference patterns like A/2022/3354 or N°055/2020
            year_match = re.search(r'/(\d{4})(?:/|$|\s)', reference)
            if year_match:
                year = int(year_match.group(1))
                # Sanity check: year should be between 1950-2030
                if 1950 <= year <= 2030:
                    return year
        
        # Try to extract from dates like "du 27 février 2020" or "du 08 décembre 2011"
        french_months = r'(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)'
        date_patterns = [
            rf'du\s+\d{{1,2}}\s+{french_months}\s+(\d{{4}})',  # du 27 février 2020
            r'du\s+(\d{4})',                                     # du 2020
            rf'\d{{1,2}}\s+{french_months}\s+(\d{{4}})',        # 27 février 2020
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                year = int(match.group(match.lastindex))
                if 1950 <= year <= 2030:
                    return year
        
        # Last resort: look for any 4-digit number that could be a year
        year_pattern = r'\b(19\d{2}|20[0-3]\d)\b'
        matches = re.findall(year_pattern, title)
        if matches:
            # Return the first reasonable year found
            return int(matches[0])
        
        return None
    
    def _extract_institution(self, title: str, doc_type: str = None) -> str:
        """Extract issuing institution from title.
        
        Args:
            title:    Document title
            doc_type: Pre-computed document type (avoids re-running classification)
        """
        title_lower = title.lower()
        
        # Courts — check full name first, then abbreviation
        if 'cour commune de justice' in title_lower or 'ccja' in title_lower:
            return 'CCJA'
        elif 'cour suprême' in title_lower or 'cour supreme' in title_lower:
            return 'Cour Suprême'
        elif "cour d'appel" in title_lower or "cour d appel" in title_lower:
            return "Cour d'Appel"
        elif 'tribunal de commerce' in title_lower:
            return 'Tribunal de Commerce'
        elif 'tribunal de première instance' in title_lower or 'tribunal de premi' in title_lower:
            return 'Tribunal de Première Instance'
        elif 'tribunal' in title_lower:
            return 'Tribunal'
        
        # Presidential/Government — check FIRST because these appear at the END of
        # reference paths (e.g. /PRG/CNRD/SGG) and are the actual issuing authority.
        # PRG/CNRD before SGG: Présidence issues Décrets; SGG just countersigns.
        if 'prg' in title_lower or 'cnrd' in title_lower:
            return 'Présidence'
        elif 'sgg' in title_lower:
            return 'Secrétariat Général du Gouvernement'
        
        # Ministries - scan all codes in the title path; return the LAST match
        # (the last code in a path like /MEEF/MMG is the issuing authority)
        ministry_codes = {
            'MMG':    'Ministère des Mines',
            'MTFP':   'Ministère du Travail',
            'MEEF':   'Ministère de l\'Environnement',
            'MEF':    'Ministère de l\'Économie et des Finances',
            'MVAT':   'Ministère de la Ville',
            'MUHAT':  'Ministère de l\'Urbanisme',
            'MTNTI':  'Ministère des Télécommunications',
            'MJDH':   'Ministère de la Justice',
            'MEHH':   'Ministère de l\'Énergie',
            'MSPC':   'Ministère de la Santé Publique',
            'MC':     'Ministère du Commerce',
        }
        
        # Find all ministry codes present in the title (as /CODE/ or /CODE at end)
        found_inst = ''
        for code, name in ministry_codes.items():
            if re.search(rf'/{re.escape(code)}(?:/|\s|$)', title):
                found_inst = name  # keep last match (rightmost = issuing authority)
        if found_inst:
            return found_inst
        
        # Laws and codes issued by the National Assembly
        # Use pre-computed doc_type if available to avoid calling _classify_document_type() again
        title_type = doc_type if doc_type is not None else self._classify_document_type(title)
        if title_type in ('loi', 'ordonnance', 'code'):
            return 'Assemblée Nationale'
        
        return ''
    
    def calculate_quality_score(self, content: str, metadata: Dict) -> float:
        """
        Calculate document quality score
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            Quality score (0-1)
        """
        score = 0.0
        
        # Length check
        if len(content) >= config.MIN_DOCUMENT_LENGTH:
            score += 0.3
        
        # Has reference
        if metadata.get('reference'):
            score += 0.2
        
        # Has year
        if metadata.get('year'):
            score += 0.2
        
        # Has clear type
        if metadata.get('type') != 'autre':
            score += 0.2
        
        # Content quality (no excessive special characters)
        clean_ratio = len(re.sub(r'[^\w\s]', '', content)) / max(len(content), 1)
        score += 0.1 * clean_ratio
        
        return min(score, 1.0)
    
    def process_document(self, post: Dict, metadata: Dict = None, force: bool = False) -> Optional[Dict]:
        """
        Process a single document: clean, extract metadata, build graph.
        
        Args:
            post:     WordPress post dict from the API
            metadata: Pre-computed metadata dict (optional, avoids double extraction
                      when called from process_all_documents which already extracted it)
            force:    If True, bypasses skip guards and force processes the document
            
        Returns:
            Processed document dictionary or None if skipped / quality too low
        """
        # Accept pre-computed metadata to avoid calling extract_metadata() twice
        metadata   = metadata or self.extract_metadata(post)
        doc_id     = metadata['id']
        source_url = metadata.get('source_url', '')

        # Ensure output_file is always defined
        output_file = self.output_dir / f"{doc_id}.json"

        if not force:
            # ── Skip guard 1: doc_id already in Neo4j ──
            if self.graph_builder.document_exists(doc_id):
                logger.info(f"Document {doc_id} already exists in Neo4j (by id), skipping...")
                return None
    
            # ── Skip guard 2: source_url already in Neo4j ──
            if source_url and self.graph_builder.document_exists_by_url(source_url):
                logger.info(f"Document {doc_id} already exists in Neo4j (by url), skipping...")
                return None
    
            # ── Skip guard 3: cleaned JSON file already on disk ──
            if output_file.exists():
                logger.info(f"Document {doc_id} already cleaned on disk, skipping...")
                return None

        # Clean content
        content = self.clean_html(post.get('content', {}).get('rendered', ''))

        
        # Calculate quality
        quality_score = self.calculate_quality_score(content, metadata)
        
        if quality_score < config.MIN_QUALITY_SCORE:
            logger.warning(f"Document {doc_id} quality too low: {quality_score:.2f}")
            if not force:
                return None
            else:
                logger.info(f"Bypassing quality check for Document {doc_id} due to force=True")
        
        # Extract citations
        citations = self.graph_builder.extract_citations(content)
        
        # Prepare document data — all 8 metadata fields included
        doc_data = {
            **metadata,          # id, title, type, reference, year, institution, source_url, date, modified
            'content':       content,
            'quality_score': quality_score,
        }
        
        # Create document node in Neo4j (stores all 8 metadata fields)
        doc_id = self.graph_builder.create_document_node(doc_data)
        
        # Create citation relationships
        if citations:
            self.graph_builder.create_citation_relationships(doc_id, citations)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Processed document {doc_id}: {metadata['title'][:50]}...")
        
        return doc_data
    
    def process_all_documents(self, max_docs: int = None) -> List[Dict]:
        """
        Process all documents from WordPress.
        
        Args:
            max_docs: Maximum number of documents to process
            
        Returns:
            List of processed documents
        """
        # Fetch posts
        posts = self.fetch_wordpress_posts(max_posts=max_docs)
        
        # Process each post
        processed_docs = []
        skipped_docs   = 0   # already in DB / on disk
        rejected_docs  = 0   # quality too low
        
        logger.info(f"Processing {len(posts)} documents...")
        
        for post in tqdm(posts, desc="Processing documents"):
            # Quick pre-check: if doc already exists, process_document returns None
            metadata = self.extract_metadata(post)
            doc_id   = metadata['id']
            
            if (self.graph_builder.document_exists(doc_id)
                    or self.graph_builder.document_exists_by_url(metadata.get('source_url', ''))
                    or (self.output_dir / f"{doc_id}.json").exists()):
                skipped_docs += 1
                continue
            
            doc = self.process_document(post, metadata=metadata)  # reuse pre-computed metadata
            if doc:
                processed_docs.append(doc)
            else:
                # process_document returned None → quality too low
                rejected_docs += 1
        
        logger.info(
            f"Done: processed={len(processed_docs)}, "
            f"skipped(existing)={skipped_docs}, "
            f"rejected(quality)={rejected_docs}, "
            f"total={len(posts)}"
        )
        
        # Get graph stats
        stats = self.graph_builder.get_graph_stats()
        logger.info(f"Graph statistics: {stats}")
        
        return processed_docs
    
    def close(self):
        """Close connections"""
        self.graph_builder.close()


def main():
    """Main execution"""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT
    )
    
    cleaner = DocumentCleaner()
    
    try:
        # Process all documents
        docs = cleaner.process_all_documents()
        
        print(f"\n✓ Processed {len(docs)} documents")
        print(f"✓ Output directory: {cleaner.output_dir}")
        
        # Show graph stats
        stats = cleaner.graph_builder.get_graph_stats()
        print("\nGraph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    finally:
        cleaner.close()


if __name__ == "__main__":
    main()
