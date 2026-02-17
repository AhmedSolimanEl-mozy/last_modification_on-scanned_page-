# Arabic Financial RAG Database

PostgreSQL + pgvector hybrid indexing system for high-accuracy Arabic financial document retrieval.

## Overview

This database indexes **417 information units** extracted from Arabic financial statements with:

- **Semantic Search**: 1024-dimensional embeddings from BAAI/bge-m3
- **Numeric Filtering**: JSONB indexes for exact financial data queries
- **Full-Text Search**: Arabic-optimized text search
- **Citations**: Page/paragraph references for source tracking

## Quick Start

### 1. Setup Database

```bash
cd database
docker compose up -d
```

See [DOCKER_SETUP.md](DOCKER_SETUP.md) for detailed instructions.

### 2. Install Python Dependencies

```bash
cd /home/ahmedsoliman/AI_projects/venv_arabic_rag
source bin/activate
pip install -r database/requirements_db.txt
```

### 3. Configure Environment

```bash
cd database
cp .env.example .env
# Edit .env if needed (default values work out of the box)
```

### 4. Run Ingestion

```bash
python database/ingest.py
```

Expected output:
```
============================================================
Arabic Financial RAG System - Ingestion Pipeline
============================================================
✓ Connected to PostgreSQL at localhost:5432
Loading embedding model: BAAI/bge-m3
✓ Model loaded successfully on device: cpu
  Embedding dimension: 1024
✓ Loaded 417 units from final_json_18_pages.json
Generating embeddings...
Embedding batches: 100%|██████████| 13/13 [00:45<00:00]
Inserting data into PostgreSQL...
✓ Successfully inserted 417 units
✓ Verification passed: 417 units in database
============================================================
✓ INGESTION COMPLETED SUCCESSFULLY
============================================================
```

### 5. Test Queries

```bash
python database/test_queries.py
```

## Database Schema

### Table: `information_units`

| Column              | Type         | Description                                    |
|---------------------|--------------|------------------------------------------------|
| `id`                | UUID         | Primary key                                    |
| `unit_id`           | TEXT         | Original identifier (e.g., `p3_para2_s4`)      |
| `page_number`       | INTEGER      | Page number in source PDF                      |
| `paragraph_number`  | INTEGER      | Paragraph number on page                       |
| `sentence_index`    | INTEGER      | Sentence index in paragraph                    |
| `unit_type`         | TEXT         | `text_only_unit` or `sentence_table_unit`      |
| `raw_text`          | TEXT         | Original Arabic text (exact)                   |
| `normalized_text`   | TEXT         | Normalized Arabic text (for search)            |
| `numeric_data`      | JSONB        | Financial data (e.g., `{"٣١ ديسمبر ٢٠٢٤": {"value": "٣٤٨ ٥٣٠"}}`) |
| `source_pdf`        | TEXT         | Source filename                                |
| `embedding`         | VECTOR(1024) | Semantic embedding from BAAI/bge-m3            |
| `created_at`        | TIMESTAMPTZ  | Insertion timestamp                            |

### Indexes

1. **HNSW Cosine Index** on `embedding` - Fast approximate similarity search
2. **GIN JSONB Index** on `numeric_data` - Exact numeric filtering
3. **GIN FTS Index** on `normalized_text` - Arabic full-text search
4. **B-tree Indexes** on `page_number`, `paragraph_number` - Citation lookups

## Query Examples

### Semantic Search

Find similar content using natural language:

```python
from sentence_transformers import SentenceTransformer
import psycopg2

# Load model
model = SentenceTransformer('BAAI/bge-m3')

# Generate query embedding
query = "الاستثمارات المالية"  # Financial investments
embedding = model.encode([query])[0].tolist()

# Search
conn = psycopg2.connect(
    host='localhost', port=5432, dbname='arab_rag_db',
    user='arab_rag', password='arab_rag_pass_2024'
)
cur = conn.cursor()

cur.execute("""
    SELECT unit_id, page_number, raw_text,
           1 - (embedding <=> %s::vector) AS similarity
    FROM information_units
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %s::vector
    LIMIT 5;
""", (embedding, embedding))

for row in cur.fetchall():
    print(f"[Page {row[1]}] {row[2]} (similarity: {row[3]:.4f})")
```

### Numeric Filtering

Find units with specific numeric data:

```sql
-- Units with 2024 financial data
SELECT unit_id, page_number, raw_text, numeric_data
FROM information_units
WHERE numeric_data ? '٣١ ديسمبر ٢٠٢٤'
ORDER BY page_number;

-- Units with both 2024 and 2023 data
SELECT unit_id, raw_text, numeric_data
FROM information_units
WHERE numeric_data ?& ARRAY['٣١ ديسمبر ٢٠٢٤', '٣١ ديسمبر ٢٠٢٣'];
```

### Citation Retrieval

Get all content from a specific location:

```sql
-- All sentences from page 3, paragraph 2
SELECT unit_id, sentence_index, raw_text
FROM information_units
WHERE page_number = 3 AND paragraph_number = 2
ORDER BY sentence_index;
```

### Hybrid Search

Combine semantic search with filters:

```sql
-- Semantic search + numeric data filter
SELECT unit_id, page_number, raw_text,
       1 - (embedding <=> %s::vector) AS similarity,
       numeric_data
FROM information_units
WHERE embedding IS NOT NULL
  AND numeric_data IS NOT NULL
ORDER BY embedding <=> %s::vector
LIMIT 10;
```

### Full-Text Search

Arabic text search:

```sql
SELECT unit_id, page_number, raw_text,
       ts_rank(to_tsvector('arabic', normalized_text), 
               plainto_tsquery('arabic', 'احتياطي')) AS rank
FROM information_units
WHERE to_tsvector('arabic', normalized_text) 
      @@ plainto_tsquery('arabic', 'احتياطي')
ORDER BY rank DESC;
```

## API Usage (Example)

For future API development:

```python
def search_financial_data(query: str, filters: dict = None, limit: int = 5):
    """
    Hybrid search with optional filters.
    
    Args:
        query: Natural language query in Arabic
        filters: Optional dict with:
            - page_number: int
            - has_numeric_data: bool
            - year: str (e.g., '٢٠٢٤')
        limit: Number of results
    """
    # Generate embedding
    embedding = model.encode([query])[0].tolist()
    
    # Build query
    sql = """
        SELECT unit_id, page_number, paragraph_number, 
               raw_text, numeric_data,
               1 - (embedding <=> %s::vector) AS similarity
        FROM information_units
        WHERE embedding IS NOT NULL
    """
    params = [embedding, embedding]
    
    # Apply filters
    if filters:
        if 'page_number' in filters:
            sql += " AND page_number = %s"
            params.append(filters['page_number'])
        if filters.get('has_numeric_data'):
            sql += " AND numeric_data IS NOT NULL"
        if 'year' in filters:
            sql += " AND numeric_data ? %s"
            params.append(f"٣١ ديسمبر {filters['year']}")
    
    sql += f" ORDER BY embedding <=> %s::vector LIMIT {limit}"
    
    # Execute
    cur.execute(sql, params)
    return cur.fetchall()
```

## Performance Considerations

- **Batch Size**: Default 32 for embedding generation
- **HNSW Parameters**: `m=16`, `ef_construction=64` (tune for accuracy vs speed)
- **GPU**: Set `DEVICE=cuda` in `.env` for faster embedding
- **Connection Pooling**: Use `psycopg2.pool` for production

## Troubleshooting

### Slow Queries

Create additional indexes if needed:
```sql
CREATE INDEX idx_custom ON information_units (page_number, unit_type);
```

### Embedding Dimension Mismatch

Ensure model produces 1024-dimensional vectors:
```python
print(model.get_sentence_embedding_dimension())  # Should be 1024
```

### Arabic Text Issues

Verify Arabic text search configuration:
```sql
SELECT * FROM pg_ts_config WHERE cfgname = 'arabic';
```

## Next Steps

- [ ] Build REST API for queries
- [ ] Add authentication and rate limiting
- [ ] Implement caching layer (Redis)
- [ ] Create web frontend
- [ ] Add query analytics

## References

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [BAAI/bge-m3 Model](https://huggingface.co/BAAI/bge-m3)
- [PostgreSQL Full-Text Search](https://www.postgresql.org/docs/current/textsearch.html)
