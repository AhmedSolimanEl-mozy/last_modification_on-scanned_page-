# Arabic RAG Financial Analyst 📊

Complete Arabic Financial Document RAG (Retrieval-Augmented Generation) system for analyzing bank financial statements with citations.

## 🎯 System Overview

Transform complex Arabic financial PDFs into an intelligent Q&A system that provides accurate, cited answers in formal Arabic.

**Example**:
```
Question: ما هي الأصول في ديسمبر ٢٠٢٤؟
Answer: إجمالي الأصول في ٣١ ديسمبر ٢٠٢٤ بلغ ٨٬١٣٧٬٣٩٤ مليون جنيه (صفحة ٣)
```

---

## 🏗️ Architecture

### 3-Phase System

```
Phase 1: PostgreSQL + pgvector Database (417 indexed units)
    ↓
Phase 2: FastAPI Backend (Dual Retrieval + Groq LLM)
    ↓
Phase 3: Streamlit Frontend (RTL Arabic Chat Interface)
```

### Technology Stack

| Component | Technology |
|-----------|------------|
| **Database** | PostgreSQL 16 + pgvector |
| **Embeddings** | BAAI/bge-m3 (1024-dim) |
| **LLM** | llama-3.3-70b-versatile (Groq API) |
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Language** | Modern Standard Arabic |
| **Domain** | Bank financial statements |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- 8GB RAM minimum
- Groq API key (free tier works)

### 1. Clone Repository

```bash
git clone https://github.com/AhmedSolimanEl-mozy/Arabic_RAG_financail_analyst.git
cd Arabic_RAG_financail_analyst
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Phase 1: Database Setup

```bash
# Start PostgreSQL with pgvector
cd database
docker compose up -d

# Install dependencies
pip install -r requirements_db.txt

# Configure environment
cp .env.example .env
# Edit .env if needed (defaults work for local development)

# Run ingestion (loads 417 units)
python ingest.py

# Verify
python test_queries.py
```

### 4. Phase 2: API Backend

```bash
# Install dependencies
pip install -r api/requirements.txt

# Configure environment
cp api/.env.example api/.env
# Add your Groq API key to api/.env

# Start API server
python -m uvicorn api.main:app --reload
```

**Access**: http://localhost:8000/docs

### 5. Phase 3: Frontend

```bash
# Install dependencies
pip install -r frontend/requirements.txt

# Run Streamlit app
streamlit run frontend/app.py
```

**Access**: http://localhost:8501

---

## 📁 Project Structure

```
Arabic_RAG_financail_analyst/
│
├── database/                    # Phase 1: Data Storage
│   ├── schema.sql               # PostgreSQL schema with pgvector
│   ├── docker-compose.yml       # Database container
│   ├── ingest.py                # Data ingestion pipeline
│   ├── test_queries.py          # Query testing
│   ├── requirements_db.txt      # Python dependencies
│   └── *.md                     # Documentation
│
├── api/                         # Phase 2: Backend
│   ├── main.py                  # FastAPI application
│   ├── retrieval.py             # Dual retrieval logic
│   ├── llm_client.py            # Groq LLM integration
│   ├── models.py                # Pydantic models
│   ├── config.py                # Configuration
│   ├── test_api.py              # API tests
│   ├── requirements.txt         # Python dependencies
│   └── *.md                     # Documentation
│
├── frontend/                    # Phase 3: UI
│   ├── app.py                   # Streamlit application
│   ├── requirements.txt         # Python dependencies
│   ├── ui_screenshot.png        # UI mockup
│   └── README.md                # Usage guide
│
├── architecture/                # System Architecture
│   ├── architecture_description.md
│   ├── drawio_specification.md
│   └── README.md
│
├── final_json_18_pages.json    # Source data (417 units)
├── deploy_phase1.sh             # Phase 1 automation
├── deploy_phase2.sh             # Phase 2 automation
├── PHASE1_COMPLETE.md           # Phase 1 summary
├── PHASE2_COMPLETE.md           # Phase 2 summary
├── PHASE3_COMPLETE.md           # Phase 3 summary
└── README.md                    # This file
```

---

## 💡 Features

### Dual Retrieval System

Combines **semantic search** (pgvector) with **exact numeric filtering** (JSONB):

1. **Numeric Intent Detection**: Identifies years, amounts, dates in Arabic (٢٠٢٤) and Western (2024) numerals
2. **Semantic Search**: Top-5 similar units via BAAI/bge-m3 embeddings
3. **Numeric Filtering**: Exact JSONB matching for financial figures
4. **Paragraph Expansion**: Retrieves complete context
5. **Citation Tracking**: Links answers to source pages

### LLM Integration

- **Model**: llama-3.3-70b-versatile (Groq)
- **Persona**: Professional Arabic financial analyst
- **Rules**:
  - ✅ No number fabrication
  - ✅ Always cite sources
  - ✅ Formal Arabic only
  - ✅ State when data is missing

### RTL Arabic Frontend

- **Layout**: Complete RTL support
- **Font**: Cairo (Google Fonts)
- **Chat**: User bubbles (purple) + Assistant bubbles (gray)
- **Citations**: Yellow highlight boxes with page numbers

---

## 🧪 Testing

### Phase 1: Database

```bash
python database/test_queries.py
```

**Tests**: Semantic search, numeric filtering, citations, hybrid queries

### Phase 2: API

```bash
python api/test_api.py
```

**7 Test Cases**:
- General semantic queries
- Numeric queries (year-specific)
- Comparisons (2024 vs 2023)
- Missing data handling
- Citation accuracy

### Phase 3: Frontend

Manual testing in browser at http://localhost:8501

**Example Questions**:
```
ما هي الأصول في ديسمبر ٢٠٢٤؟
كم بلغت ودائع العملاء في ٢٠٢٤؟
قارن بين القروض في ٢٠٢٤ و ٢٠٢٣
ما هو رأس المال المدفوع؟
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **End-to-end latency** | 2-5 seconds |
| **Retrieval** | ~200ms |
| **LLM generation** | 1-4s |
| **Database queries** | ~50ms |
| **Indexed units** | 417 units |
| **Pages** | 18 pages |

---

## 🔧 Configuration

### Database (.env in database/)

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=arab_rag_db
DB_USER=arab_rag
DB_PASSWORD=arab_rag_pass_2024

EMBEDDING_MODEL=BAAI/bge-m3
DEVICE=cpu  # or 'cuda' for GPU
```

### API (.env in api/)

```bash
# Database (same as Phase 1)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=arab_rag_db
DB_USER=arab_rag
DB_PASSWORD=arab_rag_pass_2024

# Groq API
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Retrieval
TOP_K_SEMANTIC=5
SIMILARITY_THRESHOLD=0.3
ENABLE_PARAGRAPH_EXPANSION=True
```

---

## 📖 Documentation

### Complete Guides

- **Phase 1**: `database/README.md` - Database setup and ingestion
- **Phase 2**: `api/README.md` - API backend usage
- **Phase 3**: `frontend/README.md` - Frontend interface
- **Architecture**: `architecture/architecture_description.md` - System design

### Quick References

- **API Docs**: `api/API_DOCUMENTATION.md`
- **Retrieval Flow**: `api/RETRIEVAL_FLOW.md`
- **Docker Setup**: `database/DOCKER_SETUP.md`
- **Visual Guide**: `database/VISUAL_GUIDE.md`

---

## 🐳 Docker Deployment

### Database Only

```bash
cd database
docker compose up -d
```

### Full Stack (Future)

Will add docker-compose for complete stack in future updates.

---

## 🔐 Security Notes

⚠️ **Current State**: Development mode

**Before Production**:
- [ ] Change default database password
- [ ] Add JWT authentication
- [ ] Restrict CORS origins
- [ ] Enable rate limiting
- [ ] Add SSL/TLS
- [ ] Implement API key rotation

---

## 🛠️ Troubleshooting

### Database Won't Start

```bash
docker compose -f database/docker-compose.yml ps
docker compose -f database/docker-compose.yml logs
```

### API Connection Failed

```bash
# Check database
psql postgresql://arab_rag:arab_rag_pass_2024@localhost:5432/arab_rag_db -c "SELECT COUNT(*) FROM information_units;"

# Check API is running
curl http://localhost:8000/health
```

### Frontend Can't Connect

Ensure:
1. Database is running (port 5432)
2. API is running (port 8000)
3. Check `api/.env` has correct Groq API key

---

## 🎓 Learning Resources

This project demonstrates:
- **RAG Systems**: Retrieval-Augmented Generation
- **Vector Databases**: pgvector for similarity search
- **Hybrid Search**: Semantic + exact matching
- **Arabic NLP**: RTL support, Arabic embeddings
- **LLM Integration**: Groq API, prompt engineering
- **Full-Stack Development**: FastAPI + Streamlit

Suitable for:
- AI/ML students
- RAG system learners
- Arabic NLP practitioners
- Financial tech developers

---

## 📝 Citation

If you use this project, please cite:

```
Arabic Financial RAG System
Author: Ahmed Soliman El-mozy
Year: 2026
Repository: https://github.com/AhmedSolimanEl-mozy/Arabic_RAG_financail_analyst
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Groq**: Fast LLM inference
- **BAAI**: bge-m3 embeddings
- **pgvector**: PostgreSQL vector extension
- **Gemini 2.5 Flash**: OCR for Arabic documents

---

## 📞 Contact

- **GitHub**: [@AhmedSolimanEl-mozy](https://github.com/AhmedSolimanEl-mozy)
- **Repository**: [Arabic_RAG_financail_analyst](https://github.com/AhmedSolimanEl-mozy/Arabic_RAG_financail_analyst)

---

**Status**: ✅ Production-ready (with security upgrades)  
**Version**: 1.0  
**Last Updated**: February 17, 2026
