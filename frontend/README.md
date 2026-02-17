# Arabic Financial RAG - Frontend

Simple Streamlit chat interface for Arabic financial questions.

## Features

✅ **RTL Arabic Layout** - Full right-to-left support
✅ **Chat Interface** - Clean message bubbles
✅ **Citations Panel** - Page references with excerpts
✅ **FastAPI Integration** - Calls backend /ask endpoint
✅ **Minimal Design** - Financial analyst style

## Quick Start

### 1. Install Dependencies

```bash
cd /home/ahmedsoliman/AI_projects/venv_arabic_rag
source bin/activate
pip install -r frontend/requirements.txt
```

### 2. Start Backend API (if not running)

```bash
# In terminal 1
python -m uvicorn api.main:app --reload
```

### 3. Run Streamlit App

```bash
# In terminal 2
streamlit run frontend/app.py
```

**Access:** http://localhost:8501

---

## Usage

1. **Enter Question**: Type your Arabic financial question
2. **Submit**: Click "إرسال" or press Enter
3. **View Answer**: See response with citations below
4. **Check Citations**: Review page numbers and source text

### Example Questions

```
ما هي الأصول في ديسمبر ٢٠٢٤؟
كم بلغت ودائع العملاء في ٢٠٢٤؟
قارن بين القروض في ٢٠٢٤ و ٢٠٢٣
ما هو رأس المال المدفوع؟
```

---

## Features

### RTL Support
- Full Arabic right-to-left layout
- Cairo font for better Arabic rendering
- Proper text alignment

### Chat Interface
- **User messages**: Purple gradient bubbles (right)
- **Assistant messages**: Light gray bubbles with blue accent (left)
- **Citations**: Yellow highlight boxes with page numbers

### Backend Integration
- Calls `http://localhost:8000/ask`
- Handles loading states
- Error messages in Arabic
- 30-second timeout

---

## UI Components

```
┌─────────────────────────────────────┐
│      📊 المحلل المالي              │
│   نظام ذكي للإجابة على الأسئلة    │
├─────────────────────────────────────┤
│                                     │
│  [User Question Bubble]        ←   │
│                                     │
│  →  [Assistant Answer Bubble]      │
│     📚 المراجع:                    │
│     [Citation Box - صفحة 3]        │
│                                     │
├─────────────────────────────────────┤
│  [Input Box]  [إرسال Button]      │
└─────────────────────────────────────┘
```

---

## Configuration

### API Endpoint

Edit in `frontend/app.py`:
```python
API_BASE_URL = "http://localhost:8000"
```

### Styling

Custom CSS in app for:
- Fonts: Cairo (Google Fonts)
- Colors: Financial theme (blues, purples)
- Layout: RTL with proper spacing

---

## Session State

- **Chat history**: Stored in `st.session_state.messages`
- **No database**: History cleared on refresh
- **Clear button**: Sidebar option to reset

---

## Troubleshooting

**"تعذر الاتصال بالخادم"**
- Ensure API is running: `python -m uvicorn api.main:app --reload`
- Check API is on port 8000: `curl http://localhost:8000/health`

**Styling issues**
- Clear browser cache
- Reload page (Ctrl+Shift+R)

**RTL not working**
- Ensure Cairo font loads (check browser network tab)
- Try updating Streamlit: `pip install -U streamlit`

---

## File Structure

```
frontend/
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## Next Steps

Simple enhancements (optional):
- [ ] Export chat history to PDF
- [ ] Dark mode toggle
- [ ] Voice input support
- [ ] Copy answer button
