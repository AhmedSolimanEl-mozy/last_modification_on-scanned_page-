# Arabic Financial RAG API - Phase 2

FastAPI backend with dual retrieval and LLM integration.

## Quick Start

### 1. Install Dependencies

```bash
cd /home/ahmedsoliman/AI_projects/venv_arabic_rag
source bin/activate
pip install -r api/requirements.txt
```

### 2. Configure Environment

```bash
cp api/.env.example api/.env
# Edit api/.env if needed (API key already set)
```

### 3. Start Database (if not running)

```bash
cd database
docker compose up -d
cd ..
```

### 4. Run API Server

```bash
python -m uvicorn api.main:app --reload
```

Server starts at: `http://localhost:8000`

### 5. Test API

**Interactive Docs:** http://localhost:8000/docs

**Command Line:**
```bash
python api/test_api.py
```

---

## Files

### Core Implementation
- [`main.py`](file:///home/ahmedsoliman/AI_projects/venv_arabic_rag/api/main.py) - FastAPI application
- [`retrieval.py`](file:///home/ahmedsoliman/AI_projects/venv_arabic_rag/api/retrieval.py) - Dual retrieval logic
- [`llm_client.py`](file:///home/ahmedsoliman/AI_projects/venv_arabic_rag/api/llm_client.py) - Groq LLM integration
- [`models.py`](file:///home/ahmedsoliman/AI_projects/venv_arabic_rag/api/models.py) - Pydantic models
- [`config.py`](file:///home/ahmedsoliman/AI_projects/venv_arabic_rag/api/config.py) - Configuration

### Testing
- [`test_api.py`](file:///home/ahmedsoliman/AI_projects/venv_arabic_rag/api/test_api.py) - Test script (7 test cases)

### Documentation
- [`API_DOCUMENTATION.md`](file:///home/ahmedsoliman/AI_projects/venv_arabic_rag/api/API_DOCUMENTATION.md) - API reference
- [`RETRIEVAL_FLOW.md`](file:///home/ahmedsoliman/AI_projects/venv_arabic_rag/api/RETRIEVAL_FLOW.md) - Technical deep dive

---

## Features

### Dual Retrieval
1. **Numeric Intent Detection** - Detects years, amounts, dates
2. **Semantic Search** - pgvector cosine similarity
3. **Numeric Filtering** - JSONB exact matching
4. **Paragraph Expansion** - Full context retrieval
5. **Context Building** - Structured LLM input

### LLM Integration
- **Model**: deepseek-r1-distill-llama-70b (via Groq)
- **Persona**: Professional financial analyst
- **Language**: Formal Arabic
- **Rules**: No fabrication, always cite sources

---

## API Endpoint

**POST** `/ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي الأصول في ٢٠٢٤؟"}'
```

**Response:**
```json
{
  "answer": "إجمالي الأصول في ٣١ ديسمبر ٢٠٢٤ بلغ ٨٬١٣٧٬٣٩٤ مليون جنيه (صفحة ٣)",
  "citations": [
    {"page": 3, "text": "إجمالي الأصول..."}
  ]
}
```

---

## Configuration

Edit `api/.env`:

```bash
# Database (from Phase 1)
DB_HOST=localhost
DB_PORT=5432

# Groq API
GROQ_API_KEY=your_key_here
GROQ_MODEL=deepseek-r1-distill-llama-70b

# Retrieval
TOP_K_SEMANTIC=10
SIMILARITY_THRESHOLD=0.3
```

---

## Testing

Run automated tests:
```bash
python api/test_api.py
```

**Test Cases:**
1. ✓ General semantic query
2. ✓ Numeric query (2024)
3. ✓ Comparison (2024 vs 2023)
4. ✓ Semantic (paid-up capital)
5. ✓ Numeric (customer deposits 2024)
6. ✓ Missing data (2025 - should say not available)
7. ✓ Semantic (auditors)

**Generates:**
- `api_test_results_{timestamp}.json`
- `api_test_report_{timestamp}.md`

---

## Troubleshooting

**API won't start:**
```bash
# Check database is running
docker compose -f database/docker-compose.yml ps

# Check .env file exists
ls api/.env
```

**Groq API error:**
- Check API key in `api/.env`
- Verify rate limits not exceeded
- Check internet connection

**No results returned:**
- Ensure Phase 1 ingestion completed (417 units)
- Check database connection settings

---

## Next Phase

Phase 3 will add:
- JWT authentication
- Rate limiting
- Response caching
- Frontend interface
