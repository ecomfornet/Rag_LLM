"""
Graph Builder Module - Neo4j Integration
Handles graph construction, relationship extraction, and Neo4j operations
"""
import os
import logging
import re
from typing import List, Dict, Tuple, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, ConnectionAcquisitionTimeoutError, DriverError
import config

logger = logging.getLogger(__name__)

# Immutable allowlist of relationship types used in f-string Cypher queries.
# Relationship types cannot be parameterized in Cypher, so we use a strict
# allowlist to prevent Cypher injection via the rel_type argument.
_ALLOWED_REL_TYPES: frozenset = frozenset({
    # Core relationships
    "CITES", "AMENDS", "REPLACES", "BELONGS_TO", "MENTIONS",
    "CONTRADICTS", "INTERPRETS",
    # FIX: Extended legal relationship types present in the DB but missing from
    # the allowlist — they were silently skipped on every query, making graph
    # expansion nearly useless. Added here to match config.GRAPH_RELATIONSHIP_TYPES.
    "PRIME_SUR", "COMPLEMENTE_PAR", "APPLIQUE", "MODIFIE_PAR",
})


class GraphBuilder:
    """Manages Neo4j graph database operations"""
    
    def __init__(self):
        """Initialize Neo4j connection with configured connection pool"""
        # Connection pool tuning with robust fallback (Requirement R3)
        try:
            raw_pool = getattr(config, "NEO4J_MAX_CONNECTION_POOL_SIZE", None)
            if raw_pool is None:
                raw_pool = os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "50")
            max_pool_size = int(raw_pool)
            if max_pool_size <= 0:
                logger.warning(f"Invalid NEO4J_MAX_CONNECTION_POOL_SIZE ({max_pool_size}), falling back to 50")
                max_pool_size = 50
        except (ValueError, TypeError):
            logger.warning("Unparseable NEO4J_MAX_CONNECTION_POOL_SIZE, falling back to 50")
            max_pool_size = 50

        try:
            raw_timeout = getattr(config, "NEO4J_CONNECTION_ACQUISITION_TIMEOUT", None)
            if raw_timeout is None:
                raw_timeout = os.getenv("NEO4J_CONNECTION_ACQUISITION_TIMEOUT", "15.0")
            acq_timeout = float(raw_timeout)
            if acq_timeout <= 0:
                logger.warning(f"Invalid NEO4J_CONNECTION_ACQUISITION_TIMEOUT ({acq_timeout}), falling back to 15.0")
                acq_timeout = 15.0
        except (ValueError, TypeError):
            logger.warning("Unparseable NEO4J_CONNECTION_ACQUISITION_TIMEOUT, falling back to 15.0")
            acq_timeout = 15.0

        self.max_connection_pool_size = max_pool_size
        self.connection_acquisition_timeout = acq_timeout
        self._is_closed = False

        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            max_connection_pool_size=self.max_connection_pool_size,
            connection_acquisition_timeout=self.connection_acquisition_timeout,
        )
        self._verify_connectivity()
        self._create_constraints()
        # Cache of relationship types that actually exist in the DB.
        # Populated lazily on first call to get_related_documents().
        self._existing_rel_types: Optional[frozenset] = None
        logger.info("GraphBuilder initialized successfully")
    
    def _verify_connectivity(self):
        """Verify Neo4j connection"""
        try:
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection verified")
        except (ServiceUnavailable, ConnectionAcquisitionTimeoutError, DriverError) as e:
            logger.error(f"Cannot connect to Neo4j ({type(e).__name__}): {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Neo4j connectivity check: {e}")
            raise
    
    def _create_constraints(self):
        """Create database constraints and indexes"""
        constraints = [
            "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE INDEX doc_reference  IF NOT EXISTS FOR (d:Document) ON (d.reference)",
            "CREATE INDEX doc_type       IF NOT EXISTS FOR (d:Document) ON (d.type)",
            "CREATE INDEX doc_year       IF NOT EXISTS FOR (d:Document) ON (d.year)",
            "CREATE INDEX doc_source_url IF NOT EXISTS FOR (d:Document) ON (d.source_url)",
            "CREATE INDEX doc_institution IF NOT EXISTS FOR (d:Document) ON (d.institution)",
        ]
        
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")
        
        logger.info("Database constraints created")

    def _get_existing_relationship_types(self) -> frozenset:
        """
        Query Neo4j for the relationship types that actually exist in the database.
        Result is cached after the first call so we only pay the round-trip once
        per process lifetime.

        Returns:
            frozenset of relationship type name strings (e.g. {'CITES', 'BELONGS_TO'})
        """
        if self._existing_rel_types is not None:
            return self._existing_rel_types

        query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        try:
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                result = session.run(query)
                types = frozenset(record["relationshipType"] for record in result)
            logger.info(f"Relationship types found in Neo4j: {sorted(types)}")
            self._existing_rel_types = types
        except Exception as e:
            logger.warning(f"Could not fetch relationship types from Neo4j: {e}. "
                           "Will use only CITES as safe fallback.")
            self._existing_rel_types = frozenset({"CITES"})

        return self._existing_rel_types

    def _filter_relationship_types(self, requested: List[str]) -> List[str]:
        """
        Return only the relationship types from *requested* that exist in the DB
        AND are in the security allowlist.  Logs a warning for any that are dropped
        so operators are aware of the mismatch without spamming Neo4j warnings on
        every single query.

        Args:
            requested: Relationship type names requested by the caller

        Returns:
            Filtered list safe to embed in a Cypher query
        """
        existing = self._get_existing_relationship_types()
        filtered = []
        for rt in requested:
            if rt not in _ALLOWED_REL_TYPES:
                logger.warning(f"Relationship type {rt!r} is not in the security allowlist — skipping.")
            elif rt not in existing:
                logger.debug(f"Relationship type {rt!r} does not exist in the database — skipping.")
            else:
                filtered.append(rt)

        if not filtered:
            logger.warning(
                f"None of the requested relationship types {requested} exist in the DB. "
                "Graph expansion will return no results."
            )

        return filtered

    def create_document_node(self, doc_data: Dict) -> str:
        """
        Create or update a document node with all metadata fields.
        Stores: id, title, type, reference, year, institution,
                source_url, date, modified, content, quality_score.
        
        Args:
            doc_data: Document metadata dict from data_cleaning_1.py
            
        Returns:
            Document ID
        """
        query = """
        MERGE (d:Document {id: $id})
        SET d.title         = $title,
            d.type          = $type,
            d.reference     = $reference,
            d.year          = $year,
            d.institution   = $institution,
            d.source_url    = $source_url,
            d.date          = $date,
            d.modified      = $modified,
            d.content       = $content,
            d.created_at    = datetime(),
            d.quality_score = $quality_score
        RETURN d.id as doc_id
        """
        
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            result = session.run(
                query,
                id=doc_data['id'],
                title=doc_data.get('title', ''),
                type=doc_data.get('type', ''),
                reference=doc_data.get('reference', ''),
                year=doc_data.get('year'),
                institution=doc_data.get('institution', ''),
                source_url=doc_data.get('source_url', ''),
                date=doc_data.get('date', ''),
                modified=doc_data.get('modified', ''),
                content=doc_data.get('content', ''),
                quality_score=doc_data.get('quality_score', 0.0),
            )
            doc_id = result.single()["doc_id"]
            logger.debug(f"Created/updated document node: {doc_id}")
            return doc_id
    
    def create_chunk_node(self, chunk_data: Dict) -> str:
        """
        Create a chunk node and link to document
        
        Args:
            chunk_data: Chunk metadata
            
        Returns:
            Chunk ID
        """
        query = """
        MERGE (c:Chunk {id: $chunk_id})
        SET c.content = $content,
            c.doc_id = $doc_id,
            c.chunk_index = $chunk_index,
            c.start_char = $start_char,
            c.end_char = $end_char
        WITH c
        MATCH (d:Document {id: $doc_id})
        MERGE (c)-[:BELONGS_TO]->(d)
        RETURN c.id as chunk_id
        """
        
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            result = session.run(query, **chunk_data)
            chunk_id = result.single()["chunk_id"]
            logger.debug(f"Created chunk node: {chunk_id}")
            return chunk_id
    
    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict = None
    ) -> bool:
        """
        Create a relationship between two documents
        
        Args:
            source_id: Source document ID
            target_id: Target document ID
            rel_type: Relationship type (CITES, AMENDS, etc.)
            properties: Additional relationship properties
            
        Returns:
            Success status
        """
        # Guard against Cypher injection: rel_type is interpolated directly
        # into the query string (Cypher cannot parameterize relationship types).
        # Raise immediately so injection attempts are visible in logs.
        if rel_type not in _ALLOWED_REL_TYPES:
            raise ValueError(
                f"Disallowed relationship type {rel_type!r}. "
                f"Must be one of: {sorted(_ALLOWED_REL_TYPES)}"
            )
        
        properties = properties or {}
        
        query = f"""
        MATCH (source:Document {{id: $source_id}})
        MATCH (target:Document {{id: $target_id}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += $properties
        RETURN r
        """
        
        try:
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    properties=properties
                )
                logger.debug(f"Created {rel_type} relationship: {source_id} -> {target_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False
    
    def extract_citations(self, text: str) -> List[Dict]:
        """
        Extract legal citations from text using regex patterns
        
        Args:
            text: Document text
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        for pattern in config.CITATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citations.append({
                    'reference': match.group(0),
                    'position': match.start(),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
        
        logger.debug(f"Extracted {len(citations)} citations")
        return citations
    
    def resolve_citation_to_doc_id(self, citation_ref: str) -> Optional[str]:
        """
        Resolve a citation reference to a document ID
        
        Args:
            citation_ref: Citation reference string
            
        Returns:
            Document ID if found, None otherwise
        """
        query = """
        MATCH (d:Document)
        WHERE d.reference CONTAINS $ref OR d.title CONTAINS $ref
        RETURN d.id as doc_id
        LIMIT 1
        """
        
        try:
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                result = session.run(query, ref=citation_ref)
                record = result.single()
                if record:
                    return record["doc_id"]
        except Exception as e:
            logger.error(f"Error resolving citation: {e}")
        
        return None
    
    def create_citation_relationships(self, doc_id: str, citations: List[Dict]) -> int:
        """
        Create CITES relationships for extracted citations.
        Uses a single batched UNWIND query instead of one Neo4j session per citation.
        
        Args:
            doc_id:    Source document ID
            citations: List of citation dictionaries
            
        Returns:
            Number of relationships created
        """
        if not citations:
            return 0
        
        # Resolve all citations first, collecting valid (source, target) pairs
        rows = []
        for citation in citations:
            target_id = self.resolve_citation_to_doc_id(citation['reference'])
            if target_id:
                rows.append({
                    'source_id':  doc_id,
                    'target_id':  target_id,
                    'reference':  citation['reference'],
                    'context':    citation.get('context', ''),
                })
        
        if not rows:
            logger.info(f"No resolvable citations for {doc_id}")
            return 0
        
        # Batch-write all CITES relationships in a single UNWIND call
        batch_query = """
        UNWIND $rows AS row
        MATCH (source:Document {id: row.source_id})
        MATCH (target:Document {id: row.target_id})
        MERGE (source)-[r:CITES]->(target)
        SET r.reference = row.reference,
            r.context   = row.context
        """
        try:
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                session.run(batch_query, rows=rows)
            created = len(rows)
            logger.info(f"Batch-created {created}/{len(citations)} CITES relationships for {doc_id}")
            return created
        except Exception as e:
            logger.error(f"Error batch-creating CITES relationships: {e}")
            return 0
    
    def create_interprets_relationship(
        self,
        source_id: str,
        target_id: str,
        notes: str = ""
    ) -> bool:
        """
        Create an INTERPRETS relationship: source_id (jurisprudence) interprets
        target_id (law/article).

        Args:
            source_id: Document ID of the interpreting document (case law)
            target_id: Document ID of the interpreted document (law/article)
            notes:     Optional annotation (e.g. article number, ruling date)

        Returns:
            True on success
        """
        return self.create_relationship(
            source_id, target_id, "INTERPRETS",
            properties={"notes": notes} if notes else {}
        )

    def get_related_documents(
        self,
        doc_id: str,
        relationship_types: List[str] = None,
        max_depth: int = 2
    ) -> List[Dict]:
        """
        Get documents related to a given document via graph traversal.
        Only queries relationship types that actually exist in the database,
        preventing Neo4j UnknownRelationshipTypeWarning spam in the logs.
        
        Args:
            doc_id: Source document ID
            relationship_types: Types of relationships to traverse
            max_depth: Maximum traversal depth
            
        Returns:
            List of related document dictionaries
        """
        relationship_types = relationship_types or config.GRAPH_RELATIONSHIP_TYPES

        # FIX: Filter to only relationship types that exist in the DB before
        # building the Cypher query — eliminates AMENDS/REPLACES warning spam.
        valid_types = self._filter_relationship_types(relationship_types)

        if not valid_types:
            logger.debug(f"No valid relationship types to traverse for doc {doc_id}")
            return []

        rel_pattern = "|".join(valid_types)
        
        query = f"""
        MATCH path = (d:Document {{id: $doc_id}})-[r:{rel_pattern}*1..{max_depth}]-(related:Document)
        RETURN DISTINCT related.id as doc_id,
               related.title as title,
               related.type as type,
               length(path) as distance
        ORDER BY distance
        """
        
        related_docs = []
        
        try:
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                result = session.run(query, doc_id=doc_id)
                for record in result:
                    related_docs.append({
                        'doc_id': record['doc_id'],
                        'title': record['title'],
                        'type': record['type'],
                        'distance': record['distance']
                    })
        except Exception as e:
            logger.error(f"Error getting related documents: {e}")
        
        logger.debug(f"Found {len(related_docs)} related documents for {doc_id}")
        return related_docs

    def get_related_documents_batch(
        self,
        doc_ids: List[str],
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 2
    ) -> Dict[str, int]:
        """
        Batch traverse graph for multiple seed documents using a single UNWIND Cypher query.
        Returns a dict mapping doc_id -> minimum_hop_distance across all seeds.
        Implements Contract A (graph_builder.py -> retrieval_4.py) and Requirement R3.

        Args:
            doc_ids: List of seed document IDs to expand from
            relationship_types: Types of relationships to traverse (defaults to config.GRAPH_RELATIONSHIP_TYPES)
            max_depth: Maximum traversal depth (clamped between 1 and 3)

        Returns:
            Dict[str, int]: Mapping of document ID to minimum hop distance (e.g. {'doc_123': 1, 'doc_456': 2})
        """
        if not doc_ids:
            return {}

        # 1. Clean, deduplicate, and validate seed document IDs
        unique_doc_ids = list(dict.fromkeys(
            str(d).strip() for d in doc_ids if d is not None and str(d).strip()
        ))
        if not unique_doc_ids:
            return {}

        # 2. Resolve and filter relationship types against allowlist and database
        relationship_types = relationship_types or getattr(config, "GRAPH_RELATIONSHIP_TYPES", ["CITES"])
        valid_types = self._filter_relationship_types(relationship_types)

        if not valid_types:
            logger.debug(f"No valid relationship types to traverse for doc_ids {unique_doc_ids}")
            return {}

        # 3. Sanitize max_depth
        try:
            clean_depth = int(max_depth)
        except (ValueError, TypeError):
            clean_depth = 2
        clean_depth = max(1, min(3, clean_depth))

        rel_pattern = "|".join(valid_types)

        # 4. Construct single UNWIND Cypher query with min(length(path)) aggregation
        query = f"""
        UNWIND $doc_ids AS seed_id
        MATCH path = (d:Document {{id: seed_id}})-[r:{rel_pattern}*1..{clean_depth}]-(related:Document)
        RETURN related.id AS doc_id, min(length(path)) AS min_distance
        """

        distance_map: Dict[str, int] = {}
        try:
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                result = session.run(query, doc_ids=unique_doc_ids)
                for record in result:
                    distance_map[record["doc_id"]] = int(record["min_distance"])
        except (ConnectionAcquisitionTimeoutError, ServiceUnavailable) as e:
            logger.error(f"Neo4j connection error during batch graph traversal: {e}")
        except Exception as e:
            logger.error(f"Error getting related documents batch: {e}")

        logger.debug(f"Found {len(distance_map)} related documents across {len(unique_doc_ids)} seed docs")
        return distance_map
    
    def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document already exists in Neo4j by its ID.
        
        Args:
            doc_id: Document ID to check
            
        Returns:
            True if document exists, False otherwise
        """
        query = """
        MATCH (d:Document {id: $doc_id})
        RETURN count(d) > 0 as exists
        """
        
        try:
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                result = session.run(query, doc_id=doc_id)
                record = result.single()
                return record["exists"] if record else False
        except Exception as e:
            logger.error(f"Error checking document existence: {e}")
            return False

    def document_exists_by_url(self, source_url: str) -> bool:
        """
        Check if a document already exists in Neo4j by its source_url.
        
        Args:
            source_url: The WordPress post URL
            
        Returns:
            True if a document with this URL already exists
        """
        if not source_url:
            return False
        query = """
        MATCH (d:Document {source_url: $source_url})
        RETURN count(d) > 0 as exists
        """
        try:
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                result = session.run(query, source_url=source_url)
                record = result.single()
                return record["exists"] if record else False
        except Exception as e:
            logger.error(f"Error checking document existence by URL: {e}")
            return False
    
    def chunk_exists(self, chunk_id: str) -> bool:
        """
        Check if a chunk already exists in Neo4j
        
        Args:
            chunk_id: Chunk ID to check
            
        Returns:
            True if chunk exists, False otherwise
        """
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        RETURN count(c) > 0 as exists
        """
        
        try:
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                result = session.run(query, chunk_id=chunk_id)
                record = result.single()
                return record["exists"] if record else False
        except Exception as e:
            logger.error(f"Error checking chunk existence: {e}")
            return False
    
    def get_document_chunks_count(self, doc_id: str) -> int:
        """
        Get the number of chunks for a document
        
        Args:
            doc_id: Document ID
            
        Returns:
            Number of chunks for this document
        """
        query = """
        MATCH (c:Chunk)-[:BELONGS_TO]->(d:Document {id: $doc_id})
        RETURN count(c) as chunk_count
        """
        
        try:
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                result = session.run(query, doc_id=doc_id)
                record = result.single()
                return record["chunk_count"] if record else 0
        except Exception as e:
            logger.error(f"Error getting chunk count: {e}")
            return 0
    
    def get_graph_stats(self) -> Dict:
        """Get graph statistics"""
        queries = {
            'total_documents':          ("MATCH (d:Document) RETURN count(d) as count",   'count'),
            'total_chunks':             ("MATCH (c:Chunk) RETURN count(c) as count",        'count'),
            'total_relationships':      ("MATCH ()-[r]->() RETURN count(r) as count",       'count'),
            'documents_with_citations': (
                """
                MATCH (d:Document)-[:CITES]->()
                RETURN count(DISTINCT d) as count
                """,
                'count'
            ),
            'avg_citations_per_doc': (
                """
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[r:CITES]->()
                WITH d, count(r) as citation_count
                RETURN avg(citation_count) as avg
                """,
                'avg'
            ),
        }

        stats = {}

        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            for key, (query, col) in queries.items():
                try:
                    result = session.run(query)
                    record = result.single()
                    stats[key] = record[col] if record else 0
                except Exception as e:
                    logger.error(f"Error getting stat {key}: {e}")
                    stats[key] = 0

        return stats
    
    def invalidate_relationship_type_cache(self):
        """
        Clear the cached relationship types so they are re-fetched from Neo4j
        on the next call to get_related_documents().
        Call this after ingesting new documents that may introduce new
        relationship types (e.g. AMENDS, REPLACES).
        """
        self._existing_rel_types = None
        logger.info("Relationship type cache invalidated")

    def __enter__(self):
        """Support context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager exit by cleanly closing connection pool."""
        self.close()

    def close(self):
        """Close Neo4j connection and release pool resources (idempotent)."""
        if getattr(self, "_is_closed", False):
            logger.debug("GraphBuilder is already closed")
            return

        if hasattr(self, "driver") and self.driver is not None:
            try:
                self.driver.close()
                logger.info("Neo4j connection pool closed")
            except Exception as e:
                logger.warning(f"Error while closing Neo4j driver: {e}")
            finally:
                self._is_closed = True