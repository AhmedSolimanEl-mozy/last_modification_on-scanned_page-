#!/usr/bin/env python3
"""
Arabic Financial RAG System - Phase 1 Ingestion Pipeline

Loads structured JSON information units and indexes them into PostgreSQL with pgvector.
Uses BAAI/bge-m3 for semantic embeddings.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArabicRAGIngestion:
    """Handles ingestion of Arabic financial documents into PostgreSQL + pgvector."""
    
    def __init__(self, json_path: str, env_path: str = None):
        """
        Initialize ingestion pipeline.
        
        Args:
            json_path: Path to final_json_18_pages.json
            env_path: Path to .env file (optional)
        """
        # Load environment variables
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
        
        self.json_path = json_path
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'dbname': os.getenv('DB_NAME', 'arab_rag_db'),
            'user': os.getenv('DB_USER', 'arab_rag'),
            'password': os.getenv('DB_PASSWORD', 'arab_rag_pass_2024')
        }
        
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
        self.batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))
        self.device = os.getenv('DEVICE', 'cpu')
        
        self.conn = None
        self.model = None
        
    def connect_db(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info(f"✓ Connected to PostgreSQL at {self.db_config['host']}:{self.db_config['port']}")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            raise
    
    def load_embedding_model(self):
        """Load BAAI/bge-m3 embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name, device=self.device)
            logger.info(f"✓ Model loaded successfully on device: {self.device}")
            logger.info(f"  Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise
    
    def load_json_data(self) -> List[Dict[str, Any]]:
        """Load information units from JSON file."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✓ Loaded {len(data)} units from {self.json_path}")
            return data
        except Exception as e:
            logger.error(f"✗ Failed to load JSON: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of normalized Arabic texts
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def prepare_insert_data(self, units: List[Dict[str, Any]]) -> List[tuple]:
        """
        Prepare data for batch insertion.
        
        Args:
            units: List of information units from JSON
            
        Returns:
            List of tuples ready for database insertion
        """
        # Extract normalized texts for embedding
        texts = [unit['sentence']['normalized_text'] for unit in units]
        
        # Generate embeddings in batches
        logger.info("Generating embeddings...")
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.generate_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # Prepare insertion tuples
        insert_data = []
        for unit, embedding in zip(units, all_embeddings):
            # Convert numeric_data to JSON string (or None)
            numeric_json = json.dumps(unit['numeric_data']) if unit['numeric_data'] else None
            
            insert_data.append((
                unit['unit_id'],
                unit['page'],
                unit['paragraph'],
                unit['sentence_index'],
                unit['unit_type'],
                unit['sentence']['raw_text'],
                unit['sentence']['normalized_text'],
                numeric_json,
                unit['source_pdf'],
                embedding
            ))
        
        return insert_data
    
    def insert_batch(self, data: List[tuple]):
        """
        Insert data into PostgreSQL using batch insertion.
        
        Args:
            data: List of tuples to insert
        """
        insert_query = """
            INSERT INTO information_units (
                unit_id, page_number, paragraph_number, sentence_index,
                unit_type, raw_text, normalized_text, numeric_data,
                source_pdf, embedding
            ) VALUES %s
        """
        
        try:
            with self.conn.cursor() as cur:
                execute_values(cur, insert_query, data, page_size=100)
            self.conn.commit()
            logger.info(f"✓ Successfully inserted {len(data)} units")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"✗ Insertion failed: {e}")
            raise
    
    def verify_insertion(self, expected_count: int):
        """
        Verify that all units were inserted correctly.
        
        Args:
            expected_count: Expected number of rows
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM information_units")
                actual_count = cur.fetchone()[0]
                
                if actual_count == expected_count:
                    logger.info(f"✓ Verification passed: {actual_count} units in database")
                else:
                    logger.error(f"✗ Verification failed: Expected {expected_count}, found {actual_count}")
                    raise ValueError(f"Row count mismatch: {actual_count} != {expected_count}")
        except Exception as e:
            logger.error(f"✗ Verification error: {e}")
            raise
    
    def run(self):
        """Execute the complete ingestion pipeline."""
        try:
            logger.info("=" * 60)
            logger.info("Arabic Financial RAG System - Ingestion Pipeline")
            logger.info("=" * 60)
            
            # Step 1: Connect to database
            self.connect_db()
            
            # Step 2: Load embedding model
            self.load_embedding_model()
            
            # Step 3: Load JSON data
            units = self.load_json_data()
            
            # Step 4: Generate embeddings and prepare data
            insert_data = self.prepare_insert_data(units)
            
            # Step 5: Insert into database
            logger.info("Inserting data into PostgreSQL...")
            self.insert_batch(insert_data)
            
            # Step 6: Verify insertion
            self.verify_insertion(len(units))
            
            logger.info("=" * 60)
            logger.info("✓ INGESTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            sys.exit(1)
        finally:
            if self.conn:
                self.conn.close()
                logger.info("Database connection closed")


def main():
    """Main entry point."""
    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    json_path = project_root / "final_json_18_pages.json"
    env_path = script_dir / ".env"
    
    # Check if JSON file exists
    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        sys.exit(1)
    
    # Check if .env exists, use .env.example as fallback
    if not env_path.exists():
        env_example = script_dir / ".env.example"
        if env_example.exists():
            logger.warning(".env file not found, using .env.example")
            env_path = env_example
        else:
            logger.warning(".env file not found, using default values")
            env_path = None
    
    # Run ingestion
    ingestion = ArabicRAGIngestion(str(json_path), str(env_path) if env_path else None)
    ingestion.run()


if __name__ == "__main__":
    main()
