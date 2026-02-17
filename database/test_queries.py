#!/usr/bin/env python3
"""
Arabic Financial RAG System - Test Queries

Demonstrates hybrid search capabilities:
1. Semantic similarity search
2. Numeric filtering
3. Citation retrieval
4. Combined hybrid queries
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


class RAGQueryTester:
    """Test queries for Arabic Financial RAG system."""
    
    def __init__(self, env_path: str = None):
        """Initialize query tester."""
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
        
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'dbname': os.getenv('DB_NAME', 'arab_rag_db'),
            'user': os.getenv('DB_USER', 'arab_rag'),
            'password': os.getenv('DB_PASSWORD', 'arab_rag_pass_2024')
        }
        
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
        self.device = os.getenv('DEVICE', 'cpu')
        
        self.conn = None
        self.model = None
        
        # Buffer to store results for the text file
        self.report_buffer = []
    
    def connect(self):
        """Connect to database and load model."""
        print("Connecting to database...")
        self.conn = psycopg2.connect(**self.db_config)
        
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.model = SentenceTransformer(self.embedding_model_name, device=self.device)
        print("✓ Ready for testing\n")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a query text."""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()
    
    def print_results(self, results: List[tuple], query_desc: str):
        """Pretty print query results and buffer them for TXT export."""
        header = f"📊 {query_desc}"
        divider = "=" * 80
        
        # Add to terminal
        print(divider)
        print(header)
        print(divider)
        
        # Add to buffer
        self.report_buffer.append(divider)
        self.report_buffer.append(header)
        self.report_buffer.append(divider)
        
        if not results:
            msg = "No results found.\n"
            print(msg)
            self.report_buffer.append(msg)
            return
        
        for i, row in enumerate(results, 1):
            if len(row) >= 7:
                unit_id, page, para, sent_idx, raw_text, normalized_text, similarity = row[:7]
                
                output = [
                    f"\n[{i}] Unit: {unit_id}",
                    f"    📄 Citation: Page {page}, Paragraph {para}, Sentence {sent_idx}"
                ]
                if similarity is not None:
                    output.append(f"    🎯 Similarity: {similarity:.4f}")
                output.append(f"    📝 Raw Text: {raw_text}")
                output.append(f"    🔤 Normalized: {normalized_text}")
                
                if len(row) > 7 and row[7]:
                    output.append(f"    💰 Numeric Data: {row[7]}")
                
                # Print to terminal and save to buffer
                for line in output:
                    print(line)
                    self.report_buffer.append(line)
            else:
                msg = f"[{i}] {row}"
                print(msg)
                self.report_buffer.append(msg)
        
        print()
        self.report_buffer.append("\n")

    def generate_txt_report(self):
        """Writes all buffered results to a .txt file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"RAG_Test_Results_{timestamp}.txt"
        
        print(f"Generating TXT report: {filename}...")
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("🧪 ARABIC FINANCIAL RAG SYSTEM - TEST REPORT\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 80 + "\n\n")
            f.write("\n".join(self.report_buffer))
            
        print(f"✅ Report saved successfully to {filename}")
    
    def test_semantic_search(self):
        """Test 1: Semantic similarity search."""
        query_text = "القروض والتسهيلات المصرفية"
        embedding = self.get_embedding(query_text)
        
        query = """
            SELECT 
                unit_id,
                page_number,
                paragraph_number,
                sentence_index,
                raw_text,
                normalized_text,
                1 - (embedding <=> %s::vector) AS cosine_similarity
            FROM information_units
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 5;
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, (embedding, embedding))
            results = cur.fetchall()
        
        self.print_results(results, f"Semantic Search: '{query_text}'")
    
    def test_numeric_filter(self):
        """Test 2: Numeric data filtering."""
        query = """
            SELECT 
                unit_id,
                page_number,
                paragraph_number,
                sentence_index,
                raw_text,
                normalized_text,
                NULL AS similarity,
                numeric_data
            FROM information_units
            WHERE numeric_data IS NOT NULL
                AND numeric_data ? '٣١ ديسمبر ٢٠٢٤'
            ORDER BY page_number, paragraph_number, sentence_index
            LIMIT 10;
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
        
        self.print_results(results, "Numeric Filter: Units with '٣١ ديسمبر ٢٠٢٤' data")
    
    def test_citation_retrieval(self):
        """Test 3: Citation-based retrieval."""
        page_num = 3
        para_num = 2
        
        query = """
            SELECT 
                unit_id,
                page_number,
                paragraph_number,
                sentence_index,
                raw_text,
                normalized_text,
                NULL AS similarity
            FROM information_units
            WHERE page_number = %s
                AND paragraph_number = %s
            ORDER BY sentence_index;
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, (page_num, para_num))
            results = cur.fetchall()
        
        self.print_results(results, f"Citation Retrieval: Page {page_num}, Paragraph {para_num}")
    
    def test_hybrid_search(self):
        """Test 4: Hybrid search (semantic + numeric filter)."""
        query_text = "الاستثمارات المالية"
        embedding = self.get_embedding(query_text)
        
        query = """
            SELECT 
                unit_id,
                page_number,
                paragraph_number,
                sentence_index,
                raw_text,
                normalized_text,
                1 - (embedding <=> %s::vector) AS cosine_similarity,
                numeric_data
            FROM information_units
            WHERE embedding IS NOT NULL
                AND numeric_data IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 5;
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, (embedding, embedding))
            results = cur.fetchall()
        
        self.print_results(results, f"Hybrid Search: '{query_text}' + Numeric Data Filter")
    
    def test_full_text_search(self):
        """Test 5: Full-text search on normalized text."""
        search_term = "احتياطي"
        
        query = """
            SELECT 
                unit_id,
                page_number,
                paragraph_number,
                sentence_index,
                raw_text,
                normalized_text,
                ts_rank(to_tsvector('arabic', normalized_text), 
                        plainto_tsquery('arabic', %s)) AS rank
            FROM information_units
            WHERE to_tsvector('arabic', normalized_text) @@ plainto_tsquery('arabic', %s)
            ORDER BY rank DESC
            LIMIT 5;
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, (search_term, search_term))
            results = cur.fetchall()
        
        self.print_results(results, f"Full-Text Search: '{search_term}'")
    
    def run_all_tests(self):
        """Execute all test queries."""
        print("\n" + "=" * 80)
        print("🧪 Arabic Financial RAG System - Query Tests")
        print("=" * 80 + "\n")
        
        try:
            self.connect()
            
            # Run all tests
            self.test_semantic_search()
            self.test_numeric_filter()
            self.test_citation_retrieval()
            self.test_hybrid_search()
            self.test_full_text_search()
            
            # Export everything to text file
            self.generate_txt_report()
            
            print("=" * 80)
            print("✓ All tests completed successfully")
            print("=" * 80)
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            sys.exit(1)
        finally:
            if self.conn:
                self.conn.close()


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    env_path = script_dir / ".env"
    
    if not env_path.exists():
        env_example = script_dir / ".env.example"
        if env_example.exists():
            env_path = env_example
        else:
            env_path = None
    
    tester = RAGQueryTester(str(env_path) if env_path else None)
    tester.run_all_tests()


if __name__ == "__main__":
    main()