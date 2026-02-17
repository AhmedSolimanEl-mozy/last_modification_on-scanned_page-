-- PostgreSQL + pgvector schema for Arabic Financial RAG System
-- Phase 1: Hybrid indexing for semantic search, numeric filtering, and citations

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop table if exists (for clean setup)
DROP TABLE IF EXISTS information_units CASCADE;

-- Create main table for information units
CREATE TABLE information_units (
    -- Primary identifier
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Original identifier from JSON
    unit_id TEXT NOT NULL UNIQUE,
    
    -- Location metadata for citations
    page_number INTEGER NOT NULL,
    paragraph_number INTEGER NOT NULL,
    sentence_index INTEGER NOT NULL,
    
    -- Unit type classification
    unit_type TEXT NOT NULL CHECK (unit_type IN ('text_only_unit', 'sentence_table_unit')),
    
    -- Arabic text content (exact preservation)
    raw_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    
    -- Numeric data stored as JSONB for flexible filtering
    numeric_data JSONB,
    
    -- Source document reference
    source_pdf TEXT NOT NULL,
    
    -- 1024-dimensional embedding from BAAI/bge-m3
    embedding VECTOR(1024),
    
    -- Timestamp for tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for hybrid search

-- 1. pgvector index for semantic similarity (HNSW for fast approximate search)
-- Using cosine distance (most common for embeddings)
CREATE INDEX IF NOT EXISTS idx_embedding_cosine 
ON information_units 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 2. GIN index on numeric_data JSONB for exact numeric filtering
CREATE INDEX IF NOT EXISTS idx_numeric_data_gin 
ON information_units 
USING gin (numeric_data jsonb_path_ops);

-- 3. Full-text search index on normalized_text
-- Using Arabic text search configuration
CREATE INDEX IF NOT EXISTS idx_normalized_text_fts 
ON information_units 
USING gin (to_tsvector('arabic', normalized_text));

-- 4. B-tree indexes for citation retrieval (page, paragraph lookups)
CREATE INDEX IF NOT EXISTS idx_page_number 
ON information_units (page_number);

CREATE INDEX IF NOT EXISTS idx_paragraph_number 
ON information_units (paragraph_number);

CREATE INDEX IF NOT EXISTS idx_page_paragraph 
ON information_units (page_number, paragraph_number);

-- 5. Index on unit_type for filtering
CREATE INDEX IF NOT EXISTS idx_unit_type 
ON information_units (unit_type);

-- Create helpful comments
COMMENT ON TABLE information_units IS 
'Stores Arabic financial document information units with hybrid indexing for semantic search, numeric filtering, and citations';

COMMENT ON COLUMN information_units.embedding IS 
'1024-dimensional vector from BAAI/bge-m3 model for semantic similarity search';

COMMENT ON COLUMN information_units.numeric_data IS 
'JSONB field storing numeric values from financial tables (e.g., {"٣١ ديسمبر ٢٠٢٤": {"value": "٣٤٨ ٥٣٠", "ocr_confidence": 1.0}})';

-- Print success message
DO $$
BEGIN
    RAISE NOTICE 'Schema created successfully for Arabic Financial RAG System';
    RAISE NOTICE 'Table: information_units';
    RAISE NOTICE 'Indexes: 7 total (1 vector, 1 GIN JSONB, 1 GIN FTS, 4 B-tree)';
END $$;
