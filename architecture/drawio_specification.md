# Draw.io Architecture Diagram Specification
# Arabic Financial RAG System - Complete Pipeline

## Overview
This file provides specifications for creating the architecture diagram in Draw.io/diagrams.net

## Canvas Settings
- **Size**: A3 Landscape (420mm x 297mm)
- **Grid**: 10px
- **Background**: White
- **Export**: PNG at 300 DPI

---

## Layer 1: Document Ingestion (Top)
**Position**: Y=0-150px
**Background Color**: #E3F2FD (Light Blue)

### Components:

#### Box 1.1: PDF Input
- **Position**: X=50, Y=30
- **Size**: 120x80px
- **Icon**: Document/PDF icon
- **Label**: "البنك الأهلي المصري\nFinancial Report PDF"
- **Style**: Rounded rectangle, white fill, blue border

#### Arrow→

#### Box 1.2: OCR Processing
- **Position**: X=220, Y=30
- **Size**: 180x80px
- **Label**: "Gemini 2.5 Flash OCR"
- **Sub-labels**:
  - "✓ Arabic Text (RTL)"
  - "✓ Arabic Numerals (٠-٩)"
  - "✓ Table Extraction"
- **Style**: Rounded rectangle, white fill, blue border

#### Arrow →

#### Box 1.3: Noise Filtering
- **Position**: X=450, Y=30
- **Size**: 150x80px
-**Label**: "Noise Filtering"
- **Sub-labels**:
  - "Remove logos"
  - "Remove stamps"
  - "Remove headers"
- **Style**: Rounded rectangle, white fill, blue border

#### Arrow →

#### Box 1.4: JSON Units Output
- **Position**: X=650, Y=30
- **Size**: 200x80px
- **Label**: "JSON Information Units"
- **Code Block**:
  ```
  {
    unit_id,
    page, paragraph,
    sentence,
    numeric_data,
    unit_type
  }
  ```
-**Badge**: "417 units" (top-right corner)
- **Style**: Rounded rectangle, light yellow fill, orange border

---

## Layer 2: Storage (Middle-Top)
**Position**: Y=200-400px
**Background Color**: #E8F5E9 (Light Green)
**Split into 2 sub-sections**

### Left Section: Vector Database

#### Box 2.1: pgvector
- **Position**: X=50, Y=230
- **Size**: 350x140px
- **Icon**: Database cylinder with lightning bolt
- **Label**: "pgvector Extension"
- **Fields Table**:
  ```
  embedding     vector(1024)
  normalized_text  text
  unit_id          text
  ```
- **Badge**: "BAAI/bge-m3"
- **Index Note**: "HNSW Index (cosine)"
- **Style**: Cylinder shape, light blue fill, blue border

### Right Section: Structured Database

#### Box 2.2: PostgreSQL
- **Position**: X=500, Y=230
- **Size**: 350x140px
- **Icon**: Traditional database cylinder
- **Label**: "PostgreSQL Tables"
- **Tables**:
  - `information_units`
  - `numeric_data (JSONB)`
  - `citations`
- **Indexes**:
  - GIN on JSONB
  - B-tree on page/paragraph
- **Style**: Cylinder shape, light yellow fill, orange border

### Connection

#### Bidirectional Arrow
- **Between**: Box 2.1 and Box 2.2
- **Label**: "unit_id mapping"
- **Style**: Dashed line, both arrows

### Incoming Arrow from Layer 1
- **From**: Box 1.4
- **To**: Center between Box 2.1 and 2.2
- **Split into**: Two arrows (one to each database)
- **Style**: Thick blue arrow

---

## Layer 3: Dual Retrieval (Middle)
**Position**: Y=450-750px
**Background Color**: #FFF3E0 (Light Orange) ← MOST PROMINENT
**Border**: Thick orange border (5px)

### Input Box

#### User Question
- **Position**: X=400, Y=470 (centered)
- **Size**: 300x50px
- **Text**: "ما هي الأصول في ديسمبر ٢٠٢٤؟"
- **Font**: Bold, Arabic, 18pt
- **Style**: Speech bubble pointing down, purple gradient

### Process Flow (Horizontal)

#### Step 3.1: Query Embedding
- **Position**: X=50, Y=550
- **Size**: 140x80px
- **Label**: "Query Embedding"
- **Sub**: "BGE-M3"
- **Icon**: Vector/array icon
- **Style**: Rounded rectangle, white fill

#### Arrow →

#### Step 3.2: Semantic Search
- **Position**: X=220, Y=550
- **Size**: 140x80px
- **Label**: "Semantic Search"
- **Sub**: "pgvector cosine"
- **Badge**: "Top-5 units"
- **Style**: Rounded rectangle, light blue fill

#### Arrow →

#### Step 3.3: Numeric Filter
- **Position**: X=390, Y=550
- **Size**: 140x80px
- **Label**: "Numeric Filter"
- **Sub**: "JSONB ٢٠٢٤"
- **Badge**: "Exact matches"
- **Style**: Rounded rectangle, light yellow fill

#### Arrow →

#### Step 3.4: Merge Results
- **Position**: X=560, Y=550
- **Size**: 140x80px
- **Label**: "Merge & Deduplicate"
- **Icon**: Merge/combine icon
- **Style**: Rounded rectangle, light green fill

#### Arrow →

#### Step 3.5: Paragraph Expansion
- **Position**: X=730, Y=550
- **Size**: 140x80px
- **Label**: "Paragraph Expansion"
- **Sub**: "Full context"
- **Style**: Rounded rectangle, light purple fill

### Output Box

#### Evidence Pack
- **Position**: X=350, Y=660
- **Size**: 400x60px
- **Label**: "Evidence Pack"
- **Content**: "15 units | Pages 3,4 | Citations ready"
- **Style**: Rounded rectangle, gold fill, thick border

---

## Layer 4: Reasoning (Middle-Bottom)
**Position**: Y=800-950px
**Background Color**: #F3E5F5 (Light Purple)

### Main Component

#### Box 4.1: Groq LLM
- **Position**: X=200, Y=830
- **Size**: 500x100px
- **Icon**: Cloud with brain/AI symbol
- **Label**: "Groq API: llama-3.3-70b-versatile"
- **Persona Box** (inside):
  - "Financial Analyst Persona"
  - Rules:
    - "✓ No hallucination"
    - "✓ Citations required"
    - "✓ Arabic RTL"
    - "✓ Exact numbers only"
- **Config**: "temp=0.1, max_tokens=1000"
- **Style**: Cloud shape, purple gradient fill

### Input Arrow
- **From**: Evidence Pack (Layer 3)
- **To**: Box 4.1 (top)
- **Label**: "Context + Question"
- **Style**: Thick purple arrow

### Output Box

#### Answer with Citations
- **Position**: X=750, Y=850
- **Size**: 200x60px
- **Label**: "Generated Answer"
- **Example**: "إجمالي الأصول... (صفحة ٣)"
- **Style**: Rounded rectangle, light green fill, green border

---

## Layer 5: Frontend (Bottom)
**Position**: Y=1000-1200px
**Background Color**: #ECEFF1 (Light Gray)

### Main Component

#### Streamlit Interface Mockup
- **Position**: X=150, Y=1030
- **Size**: 600x150px
- **Style**: Browser window frame

**Contents** (mockup screenshot):
1. **Header**: "📊 المحلل المالي"
2. **User Bubble** (right side):
   - Purple gradient
   - Text: "ما هي الأصول؟"
3. **Assistant Bubble** (left side):
   - Light gray
   - Blue left border
   - Text: "إجمالي الأصول..."
4. **Citation Box** (below assistant):
   - Yellow background
   - Text: "صفحة ٣: ..."
5. **Input Field** (bottom):
   - Text: "مثال: ما هي الأصول في ٢٠٢٤؟"
   - Button: "إرسال" (purple)

### Connection Arrows

#### Input Arrow
- **From**: Box 4.1 Answer
- **To**: Streamlit mockup (top)
- **Label**: "HTTP Response"
- **Style**: Gray arrow

#### Feedback Loop
- **From**: Input field
- **To**: Layer 3 User Question (curved back)
- **Label**: "New Query"
- **Style**: Dotted gray arrow

---

## Additional Elements

### Side Panel: Key Metrics

**Position**: X=900, Y=400
**Size**: 150x300px
**Background**: White with light border

**Contents**:
```
⚙️ System Stats

📊 Data
417 units
18 pages
1 PDF

🔍 Retrieval
Top-5 semantic
~200ms latency

🤖 LLM
llama-3.3-70b
2-4s generation

💾 Storage
pgvector
PostgreSQL
JSONB indexes

🎨 Frontend
Streamlit
RTL Arabic
Cairo font
```

###Legend Box

**Position**: X=50, Y=1200
**Size**: 800x50px

**Color Key**:
- 🔵 Blue = Ingestion
- 🟢 Green = Storage
- 🟠 Orange = Retrieval
- 🟣 Purple = Reasoning
- ⚪ Gray = Frontend

**Icon Key**:
- 📄 = Document
- 🗄️ = Database
- ⚡ = Processing
- 🤖 = AI/LLM
- 💬 = Chat

---

## Arrows & Flow

### Main Data Flow (Sequential)
```
PDF → OCR → JSON → Databases → Retrieval → LLM → Frontend
```
- **Style**: Thick solid arrows
- **Color**: Gradient from blue → purple → gray

### Query Flow (User-initiated)
```
User Input → Dual Retrieval → LLM → Display
```
- **Style**: Dashed arrows
- **Color**: Orange

### Feedback Loop
```
Frontend → New Query → back to Retrieval
```
- **Style**: Curved dotted arrow
- **Color**: Light gray

---

## Special Annotations

### Note Boxes (Small yellow boxes with icons)

1. **Near OCR**: "Handles ٠-٩ and 0-9"
2. **Near JSON**: "Text-only & sentence-table units"
3. **Near Retrieval**: "5 semantic + exact numeric"
4. **Near LLM**: "No fabrication guarantee"
5. **Near Frontend**: "Session state (no DB)"

---

## Typography

- **Headers**: Cairo Bold, 16pt
- **Labels**: Cairo Regular, 12pt
- **Arabic Text**: Cairo Regular, 14pt-18pt
- **Code**: Monospace, 10pt
- **Annotations**: Cairo Regular, 10pt, Italic

---

## Color Palette

| Element | Color Code | Usage |
|---------|-----------|-------|
| Layer 1 Background | #E3F2FD | Light Blue |
| Layer 2 Background | #E8F5E9 | Light Green |
| Layer 3 Background | #FFF3E0 | Light Orange |
| Layer 4 Background | #F3E5F5 | Light Purple |
| Layer 5 Background | #ECEFF1 | Light Gray |
| pgvector Box | #BBDEFB | Blue |
| PostgreSQL Box | #FFF9C4 | Yellow |
| LLM Cloud | #CE93D8 | Purple |
| User Bubble | #7E57C2 | Purple Gradient |
| Assistant Bubble | #F5F5F5 | Gray |
| Citation Box | #FFF9E6 | Light Yellow |
| Arrows | #424242 | Dark Gray |

---

## Export Instructions

1. **File → Export As → PNG**
2. **Settings**:
   - DPI: 300
   - Transparent Background: No
   - Selection Only: No
   - Include Grid: No
3. **Save as**: `architecture_diagram.png`

4. **File → Save As**:
   - Format: Draw.io XML
   - Save as: `architecture_diagram.drawio`

---

## Notes for Manual Creation

1. Start with background rectangles for each layer
2. Add layer labels on left side vertically
3. Create components left-to-right within each layer
4. Add arrows last (to avoid clutter while positioning)
5. Group related components for easier movement
6. Use alignment tools for professional look
7. Add annotations last
8. Export at high resolution for presentations

---

**Created**: February 17, 2026
**For**: Arabic Financial RAG System Documentation
**Tool**: Draw.io / diagrams.net
**Target Audience**: Technical presentations, teaching, interviews
