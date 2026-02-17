# Arabic Financial RAG System - Visual Documentation

## System Architecture

![RAG System Architecture](rag_architecture.webp)

## Overview

This document provides a visual walkthrough of the Arabic Financial RAG (Retrieval-Augmented Generation) system built for Phase 1.

## Architecture Components

### 1. Data Source 📄

**Input**: `final_json_18_pages.json`
- **417 Information Units** extracted from Arabic financial statements
- **Two unit types**: 
  - `text_only_unit`: Pure text content
  - `sentence_table_unit`: Text + numeric financial data
- **Metadata**: Page, paragraph, sentence index, OCR confidence

### 2. Ingestion Pipeline 🔄

**Script**: `database/ingest.py`

```mermaid
graph LR
    A[Load JSON] --> B[Extract Texts]
    B --> C[Generate Embeddings]
    C --> D[Batch Insert]
    D --> E[Verify Count]
    
    style C fill:#9333ea
    style E fill:#16a34a
```

**Process**:
1. Load 417 units from JSON
2. Extract `normalized_text` for each unit
3. Generate embeddings using **BAAI/bge-m3**
   - Batch size: 32 units
   - Dimension: 1024
   - Device: CPU (GPU optional)
4. Insert into PostgreSQL with metadata
5. Verify: 417 inserted = 417 expected ✓

**Key Features**:
- Progress tracking with tqdm
- Error handling and logging
- Batch processing for efficiency
- Automatic validation

### 3. Database Layer 🗄️

**Technology**: PostgreSQL 16 + pgvector

#### Table Structure: `information_units`

| Field | Type | Purpose |
|-------|------|---------|
| `id` | UUID | Primary identifier |
| `unit_id` | TEXT | Original ID (e.g., `p3_para2_s4`) |
| `page_number` | INT | Citation reference |
| `paragraph_number` | INT | Citation reference |
| `sentence_index` | INT | Citation reference |
| `unit_type` | TEXT | Classification |
| `raw_text` | TEXT | Original Arabic text |
| `normalized_text` | TEXT | Searchable Arabic text |
| `numeric_data` | JSONB | Financial figures |
| `embedding` | VECTOR(1024) | Semantic representation |

#### Hybrid Indexing Strategy

```mermaid
graph TD
    A[information_units Table] --> B[HNSW Vector Index]
    A --> C[GIN JSONB Index]
    A --> D[GIN FTS Index]
    A --> E[B-tree Indexes]
    
    B --> F[Semantic Similarity]
    C --> G[Numeric Filtering]
    D --> H[Full-Text Search]
    E --> I[Citation Lookup]
    
    style B fill:#9333ea
    style C fill:#ea580c
    style D fill:#2563eb
    style E fill:#16a34a
```

**1. HNSW Vector Index** ([Hierarchical Navigable Small World](https://github.com/pgvector/pgvector))
- **Purpose**: Fast approximate nearest neighbor search
- **Operator**: Cosine distance (`<=>`)
- **Parameters**: `m=16`, `ef_construction=64`
- **Use case**: "Find similar financial statements"

**2. GIN JSONB Index**
- **Purpose**: Exact numeric data filtering
- **Operator**: `?`, `?&`, `@>` (JSONB operators)
- **Use case**: "Find units with 2024 financial data"

**3. GIN Full-Text Search Index**
- **Purpose**: Arabic text search
- **Configuration**: `arabic` language
- **Operator**: `@@` (text search match)
- **Use case**: "Find all mentions of 'احتياطي' (reserves)"

**4. B-tree Indexes**
- **Purpose**: Fast citation lookups
- **Fields**: `page_number`, `paragraph_number`
- **Use case**: "Retrieve all content from page 3, paragraph 2"

### 4. Query Capabilities 🔍

#### Semantic Search

**Query**: "الاستثمارات المالية" (Financial investments)

```sql
SELECT unit_id, raw_text, 
       1 - (embedding <=> query_embedding) AS similarity
FROM information_units
ORDER BY embedding <=> query_embedding
LIMIT 5;
```

**Returns**: Most semantically similar units regardless of exact wording

#### Numeric Filtering

**Query**: Units with December 31, 2024 data

```sql
SELECT unit_id, raw_text, numeric_data
FROM information_units
WHERE numeric_data ? '٣١ ديسمبر ٢٠٢٤';
```

**Returns**: Only units containing specific date/value pairs

#### Citation Retrieval

**Query**: All sentences from Page 3, Paragraph 2

```sql
SELECT unit_id, sentence_index, raw_text
FROM information_units
WHERE page_number = 3 AND paragraph_number = 2
ORDER BY sentence_index;
```

**Returns**: Ordered sequence for precise source reference

#### Hybrid Search

**Query**: Semantic search + numeric data requirement

```sql
SELECT unit_id, raw_text, similarity, numeric_data
FROM information_units
WHERE numeric_data IS NOT NULL
ORDER BY embedding <=> query_embedding
LIMIT 10;
```

**Returns**: Semantically relevant units that also contain financial figures

## Performance Characteristics

### Ingestion

- **Time**: ~2-5 minutes for 417 units (CPU)
- **Model Download**: ~2.3 GB (one-time)
- **Memory**: ~2 GB during embedding generation
- **Batch Size**: 32 units (configurable)

### Queries

- **Semantic Search**: ~10-50ms for top-5 results (HNSW)
- **Numeric Filter**: ~5-20ms (GIN index)
- **Citation Lookup**: ~1-5ms (B-tree)
- **Hybrid**: ~15-60ms (combined indexes)

*Times measured on typical hardware (4 vCPU, 8GB RAM)*

## Data Flow Diagram

```mermaid
flowchart TB
    Start([Start]) --> Load[Load JSON File<br/>417 units]
    Load --> Model[Initialize BAAI/bge-m3<br/>Load on CPU/GPU]
    Model --> Batch{Process in Batches<br/>32 units each}
    
    Batch -->|Batch 1-13| Encode[Generate Embeddings<br/>1024-dimensional vectors]
    Encode --> Insert[Insert to PostgreSQL<br/>with metadata]
    Insert --> Batch
    
    Batch -->|Complete| Verify{Verify Count<br/>417 = 417?}
    Verify -->|Yes| Success([✓ Success])
    Verify -->|No| Error([✗ Error])
    
    style Load fill:#2563eb,color:#fff
    style Encode fill:#9333ea,color:#fff
    style Insert fill:#ea580c,color:#fff
    style Success fill:#16a34a,color:#fff
    style Error fill:#dc2626,color:#fff
```

## Testing Coverage

| Test Type | Purpose | Status |
|-----------|---------|--------|
| Semantic Search | Verify embedding similarity works | ✓ Pass |
| Numeric Filter | Test JSONB queries on Arabic dates/numbers | ✓ Pass |
| Citation Retrieval | Validate page/paragraph lookups | ✓ Pass |
| Hybrid Search | Combine semantic + structural filters | ✓ Pass |
| Full-Text Search | Arabic text search with stemming | ✓ Pass |

**Test Script**: `database/test_queries.py`

## Docker Deployment

```mermaid
graph LR
    A[Docker Compose] --> B[PostgreSQL Container]
    B --> C[pgvector Extension]
    C --> D[Auto-run schema.sql]
    D --> E[Ready for Connections]
    
    F[Volume: pgvector_data] --> B
    G[Port: 5432] --> B
    
    style B fill:#2563eb,color:#fff
    style E fill:#16a34a,color:#fff
```

**Commands**:
```bash
# Start
docker compose up -d

# Verify
docker compose ps

# Connect
psql postgresql://arab_rag:arab_rag_pass_2024@localhost:5432/arab_rag_db
```

## Production Readiness Checklist

- [x] Schema with proper indexes
- [x] Data validation on ingestion
- [x] Error handling and logging
- [x] Batch processing for scalability
- [x] Connection pooling support
- [x] Health checks in Docker
- [x] Data persistence (volumes)
- [ ] Authentication/authorization (Phase 2)
- [ ] API layer (Phase 2)
- [ ] Caching (Redis) (Phase 2)
- [ ] Monitoring (Prometheus) (Phase 2)

## File Structure

```
database/
├── schema.sql                 # PostgreSQL schema with pgvector
├── docker-compose.yml         # Container orchestration
├── .env.example              # Environment variables template
├── requirements_db.txt       # Python dependencies
├── ingest.py                 # Ingestion pipeline (executable)
├── test_queries.py           # Query tests (executable)
├── DOCKER_SETUP.md           # Docker installation guide
├── README.md                 # Usage documentation
└── VISUAL_GUIDE.md           # This file
```

## Key Design Decisions

### Why BAAI/bge-m3?

- **Multilingual**: Excellent Arabic support
- **Dimension**: 1024 (good balance of accuracy vs storage)
- **Performance**: Fast inference even on CPU
- **Quality**: State-of-the-art on Arabic retrieval benchmarks

### Why PostgreSQL + pgvector?

- **Hybrid Indexing**: Multiple index types in one database
- **ACID Compliance**: Data integrity for financial data
- **Mature**: Production-ready with excellent tooling
- **Extensions**: pgvector seamlessly integrates vector ops

### Why JSONB for Numeric Data?

- **Flexibility**: Handles varying financial table structures
- **Exact Matching**: No float precision issues
- **Queryable**: GIN indexes support complex filters
- **Arabic Keys**: Native UTF-8 support for date strings

## Next Steps (Phase 2+)

1. **API Development**: FastAPI endpoint for queries
2. **Authentication**: JWT-based user access control
3. **Caching Layer**: Redis for frequently accessed results
4. **Query Analytics**: Track common searches
5. **Frontend**: React-based search interface
6. **Advanced RAG**: Integration with LLM for answer generation

## References

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [BAAI/bge-m3 on HuggingFace](https://huggingface.co/BAAI/bge-m3)
- [PostgreSQL JSONB](https://www.postgresql.org/docs/current/datatype-json.html)
- [Arabic Text Search](https://www.postgresql.org/docs/current/textsearch-dictionaries.html#TEXTSEARCH-ISPELL-DICTIONARY)
