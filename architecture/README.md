# Architecture Documentation - Arabic Financial RAG System

Complete visual and technical documentation of the system architecture.

## Files

1. **`architecture_description.md`** - Comprehensive technical documentation
   - All 5 layers explained in detail
   - Dual retrieval mechanics
   - Data flow diagrams
   - Performance metrics
   - Deployment architecture
   - Teaching-oriented explanations
   - 50+ sections covering complete system

2. **`drawio_specification.md`** - Draw.io diagram specifications
   - exact component positioning
   - Color codes and styling
   - Labels in Arabic and English
   - Arrow flows and connections
   - Export instructions
   - Manual creation guide

## Architecture Layers

### Layer 1: Document Ingestion
PDF → Gemini OCR → Noise Filtering → JSON Units (417)

### Layer 2: Storage
- pgvector (embeddings, semantic search)
- PostgreSQL (structured data, JSONB, citations)

### Layer 3: Dual Retrieval ★
User Question → Query Embedding → Semantic Search + Numeric Filter → Paragraph Expansion → Evidence Pack

### Layer 4: Reasoning
Evidence Pack → Groq LLM (llama-3.3-70b-versatile) → Answer + Citations

### Layer 5: Frontend
Streamlit RTL Chat → User Bubbles → Assistant + Citation Boxes

## Key System Characteristics

**Tech Stack**:
- LLM: llama-3.3-70b-versatile (Groq)
- Embeddings: BAAI/bge-m3 (1024-dim)
- Database: PostgreSQL + pgvector
- Backend: FastAPI
- Frontend: Streamlit (RTL Arabic)

**Performance**:
- End-to-end latency: 2-5 seconds
- Retrieval: ~200ms
- LLM generation: 1-4s
- Database queries: ~50ms

**Data**:
- 417 information units
- 18 pages indexed
- Arabic financial statements
- Exact numeric preservation

## Creating the Visual Diagram

### Option 1: Generate Image (Automatic)
Use image generation tool to create visual representation based on `drawio_specification.md`

### Option 2: Manual Creation (Recommended for editing)
1. Open Draw.io (https://app.diagrams.net)
2. Follow `drawio_specification.md` instructions
3. Create 5 layers with specified components
4. Add arrows showing data flow
5. Export as PNG (300 DPI) and save XML

### Option 3: Third-Party Tool
- Use PlantUML, Mermaid, or similar
- Convert specification to appropriate format

## Usage

**For Interviews**:
- Show complete system understanding
- Explain dual retrieval innovation
- Discuss accuracy mechanisms
- Demonstrate Arabic OCR knowledge

**For Teaching**:
- Use layer-by-layer breakdown
- Explain RAG concepts with real example
- Show how accuracy is guaranteed
- Demonstrate production architecture

**For Documentation**:
- Include in technical specs
- Reference in API documentation
- Use for onboarding new developers
- Embed in presentations

## Example Questions the Diagram Answers

1. **How does Arabic OCR work?**
   → Layer 1: Gemini 2.5 Flash with multimodal vision

2. **Why is the system accurate with numbers?**
   → Layer 3: Exact JSONB filtering + Layer 4: LLM constraints

3. **How are citations generated?**
   → Layer 3: Page tracking + Layer 4: Regex extraction

4. **What makes retrieval "dual"?**
   → Layer 3: Semantic search (pgvector) + Numeric filter (JSONB)

5. **How does the frontend handle Arabic?**
   → Layer 5: RTL CSS + Cairo font + RTL bubbles

## Files to Create

- [ ] `architecture_diagram.png` - Visual illustration
- [x] `architecture_description.md` - Technical documentation
- [x] `drawio_specification.md` - Diagram creation guide
- [ ] `architecture_diagram.drawio` - Editable Draw.io file

## See Also

- **Phase 1**: `database/VISUAL_GUIDE.md` (database architecture)
- **Phase 2**: `api/RETRIEVAL_FLOW.md` (retrieval details)
- **Phase 3**: `frontend/README.md` (UI components)
- **Complete System**: `walkthrough.md` (end-to-end guide)

---

**Status**: Documentation complete, visual diagram pending manual creation
**Created**: February 17, 2026
**Target Audience**: Developers, students, interviewers, stakeholders
