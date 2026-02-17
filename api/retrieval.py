"""
Dual Retrieval Module - Semantic + Numeric + Paragraph Expansion

Implements the core retrieval logic:
1. Numeric intent detection
2. Semantic search with pgvector
3. Structured numeric filtering
4. Paragraph expansion
5. Context building
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import logging

import psycopg2
from sentence_transformers import SentenceTransformer

from api.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DualRetriever:
    """Handles dual retrieval: semantic + numeric filtering + paragraph expansion."""
    
    def __init__(self):
        """Initialize retriever with database connection and embedding model."""
        self.db_conn = None
        self.model = None
        self._connect_db()
        self._load_model()
    
    def _connect_db(self):
        """Establish database connection."""
        try:
            self.db_conn = psycopg2.connect(
                host=settings.db_host,
                port=settings.db_port,
                dbname=settings.db_name,
                user=settings.db_user,
                password=settings.db_password
            )
            logger.info("✓ Connected to database")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def _load_model(self):
        """Load embedding model."""
        try:
            self.model = SentenceTransformer(
                settings.embedding_model,
                device=settings.device
            )
            logger.info(f"✓ Loaded {settings.embedding_model}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def detect_numeric_intent(self, question: str) -> Tuple[bool, List[str]]:
        """
        Detect if question contains numeric intent.
        
        Args:
            question: User question in Arabic
            
        Returns:
            Tuple of (has_numeric_intent, extracted_values)
        """
        extracted = []
        
        # Patterns for years (Arabic and Western numerals)
        year_patterns = [
            r'٢٠٢[٠-٩]',  # Arabic numerals 2020-2029
            r'202[0-9]',   # Western numerals 2020-2029
        ]
        
        # Keywords indicating numeric intent
        numeric_keywords = [
            'سنة', 'عام', 'ديسمبر', 'يونيو',  # Time
            'مليون', 'ألف', 'جنيه',  # Currency
            'نسبة', '٪', 'بالمئة',  # Percentage
            'كم', 'قيمة', 'مبلغ',  # Amount questions
        ]
        
        # Check for year patterns
        for pattern in year_patterns:
            matches = re.findall(pattern, question)
            extracted.extend(matches)
        
        # Check for numeric keywords
        has_keywords = any(keyword in question for keyword in numeric_keywords)
        
        # Extract Arabic numbers
        arabic_nums = re.findall(r'[٠-٩]+', question)
        extracted.extend(arabic_nums)
        
        # Extract Western numbers
        western_nums = re.findall(r'\d+', question)
        extracted.extend(western_nums)
        
        has_intent = bool(extracted) or has_keywords
        
        if has_intent:
            logger.info(f"Numeric intent detected: {extracted}")
        
        return has_intent, list(set(extracted))  # Deduplicate
    
    def semantic_search(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search using pgvector.
        
        Args:
            question: User question in Arabic
            top_k: Number of results to return
            
        Returns:
            List of units with similarity scores
        """
        if top_k is None:
            top_k = settings.top_k_semantic
        
        # Generate query embedding
        embedding = self.model.encode([question], convert_to_numpy=True)[0].tolist()
        
        query = """
            SELECT 
                id, unit_id, page_number, paragraph_number, sentence_index,
                unit_type, raw_text, normalized_text, numeric_data, source_pdf,
                1 - (embedding <=> %s::vector) AS similarity
            FROM information_units
            WHERE embedding IS NOT NULL
                AND 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        
        try:
            with self.db_conn.cursor() as cur:
                cur.execute(query, (
                    embedding, 
                    embedding, 
                    settings.similarity_threshold,
                    embedding,
                    top_k
                ))
                rows = cur.fetchall()
                
                units = []
                for row in rows:
                    units.append({
                        'id': row[0],
                        'unit_id': row[1],
                        'page': row[2],
                        'paragraph': row[3],
                        'sentence_index': row[4],
                        'unit_type': row[5],
                        'raw_text': row[6],
                        'normalized_text': row[7],
                        'numeric_data': row[8],
                        'source_pdf': row[9],
                        'similarity': float(row[10])
                    })
                
                logger.info(f"Semantic search returned {len(units)} units")
                return units
                
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    def numeric_filter(self, question: str, extracted_values: List[str]) -> List[Dict[str, Any]]:
        """
        Filter units by numeric data (JSONB).
        
        Args:
            question: User question
            extracted_values: Extracted numeric values from question
            
        Returns:
            List of units containing the numeric values
        """
        if not extracted_values:
            return []
        
        # Build JSONB filter queries
        units = []
        
        for value in extracted_values:
            # Common date patterns in Arabic financial docs
            date_patterns = [
                f"٣١ ديسمبر {value}",  # December 31, YEAR
                f"٣٠ يونيو {value}",   # June 30, YEAR
                value  # Raw value
            ]
            
            for pattern in date_patterns:
                query = """
                    SELECT 
                        id, unit_id, page_number, paragraph_number, sentence_index,
                        unit_type, raw_text, normalized_text, numeric_data, source_pdf
                    FROM information_units
                    WHERE numeric_data::text LIKE %s
                    LIMIT 50;
                """
                
                try:
                    with self.db_conn.cursor() as cur:
                        cur.execute(query, (f'%{pattern}%',))
                        rows = cur.fetchall()
                        
                        for row in rows:
                            units.append({
                                'id': row[0],
                                'unit_id': row[1],
                                'page': row[2],
                                'paragraph': row[3],
                                'sentence_index': row[4],
                                'unit_type': row[5],
                                'raw_text': row[6],
                                'normalized_text': row[7],
                                'numeric_data': row[8],
                                'source_pdf': row[9],
                                'similarity': 1.0  # Exact match
                            })
                except Exception as e:
                    logger.warning(f"Numeric filter warning for {pattern}: {e}")
        
        # Deduplicate by unit_id
        seen = set()
        deduped = []
        for unit in units:
            if unit['unit_id'] not in seen:
                seen.add(unit['unit_id'])
                deduped.append(unit)
        
        logger.info(f"Numeric filter returned {len(deduped)} units")
        return deduped
    
    def expand_paragraphs(self, units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Expand retrieved units to include full paragraphs.
        
        Args:
            units: List of retrieved units
            
        Returns:
            Expanded list with complete paragraphs
        """
        if not settings.enable_paragraph_expansion:
            return units
        
        # Get unique (page, paragraph) pairs
        paragraphs = set()
        for unit in units:
            paragraphs.add((unit['page'], unit['paragraph']))
        
        # Retrieve all units from these paragraphs
        expanded = []
        
        for page, para in paragraphs:
            query = """
                SELECT 
                    id, unit_id, page_number, paragraph_number, sentence_index,
                    unit_type, raw_text, normalized_text, numeric_data, source_pdf
                FROM information_units
                WHERE page_number = %s AND paragraph_number = %s
                ORDER BY sentence_index;
            """
            
            try:
                with self.db_conn.cursor() as cur:
                    cur.execute(query, (page, para))
                    rows = cur.fetchall()
                    
                    for row in rows:
                        expanded.append({
                            'id': row[0],
                            'unit_id': row[1],
                            'page': row[2],
                            'paragraph': row[3],
                            'sentence_index': row[4],
                            'unit_type': row[5],
                            'raw_text': row[6],
                            'normalized_text': row[7],
                            'numeric_data': row[8],
                            'source_pdf': row[9],
                            'similarity': 0.0  # Context units
                        })
            except Exception as e:
                logger.warning(f"Paragraph expansion warning for p{page}_para{para}: {e}")
        
        # Deduplicate and sort
        seen = set()
        deduped = []
        for unit in expanded:
            if unit['unit_id'] not in seen:
                seen.add(unit['unit_id'])
                deduped.append(unit)
        
        # Sort by page, paragraph, sentence
        deduped.sort(key=lambda u: (u['page'], u['paragraph'], u['sentence_index']))
        
        logger.info(f"Paragraph expansion: {len(units)} → {len(deduped)} units")
        return deduped
    
    def build_context(self, units: List[Dict[str, Any]]) -> str:
        """
        Build context string for LLM from retrieved units.
        
        Args:
            units: List of retrieved units
            
        Returns:
            Formatted context string
        """
        if not units:
            return "لا توجد معلومات ذات صلة."
        
        context_parts = []
        current_page = None
        
        for unit in units:
            # Add page header if changed
            if unit['page'] != current_page:
                context_parts.append(f"\n[صفحة {unit['page']}]")
                current_page = unit['page']
            
            # Add unit text
            context_parts.append(f"{unit['raw_text']}")
            
            # Add numeric data if available
            if unit['numeric_data']:
                context_parts.append(f"  البيانات الرقمية: {unit['numeric_data']}")
        
        return "\n".join(context_parts)
    
    def retrieve(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Main retrieval method implementing dual retrieval logic.
        
        Args:
            question: User question in Arabic
            
        Returns:
            Tuple of (context_string, list_of_units)
        """
        logger.info(f"Retrieving for question: {question}")
        
        # Step 1: Detect numeric intent
        has_numeric, extracted_values = self.detect_numeric_intent(question)
        
        # Step 2: Semantic search
        semantic_units = self.semantic_search(question)
        
        # Step 3: Numeric filter (if applicable)
        if has_numeric:
            numeric_units = self.numeric_filter(question, extracted_values)
            
            # Combine results (prioritize numeric matches)
            combined = numeric_units + semantic_units
        else:
            combined = semantic_units
        
        # Step 4: Paragraph expansion
        expanded_units = self.expand_paragraphs(combined)
        
        # Step 5: Build context
        context = self.build_context(expanded_units)
        
        logger.info(f"Retrieved {len(expanded_units)} units for context")
        
        return context, expanded_units
    
    def close(self):
        """Close database connection."""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Database connection closed")
