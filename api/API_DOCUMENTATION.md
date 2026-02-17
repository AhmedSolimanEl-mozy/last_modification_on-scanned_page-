# API Documentation - Arabic Financial RAG System

## Overview

FastAPI backend providing question answering for Arabic financial documents with automatic citation.

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Health Check

**GET** `/`

Returns API status and available endpoints.

**Response:**
```json
{
  "status": "healthy",
  "service": "Arabic Financial RAG API",
  "version": "2.0.0",
  "endpoints": {
    "ask": "/ask (POST)",
    "docs": "/docs",
    "openapi": "/openapi.json"
  }
}
```

---

### 2. Ask Question

**POST** `/ask`

Submit a question in Arabic and receive an answer with citations.

**Request Body:**
```json
{
  "question": "ما هي الأصول في ٢٠٢٤؟"
}
```

**Request Schema:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| question | string | Yes | Arabic question (3-500 chars) |

**Response:**
```json
{
  "answer": "إجمالي الأصول في ٣١ ديسمبر ٢٠٢٤ بلغ ٨٬١٣٧٬٣٩٤ مليون جنيه (صفحة ٣)",
  "citations": [
    {
      "page": 3,
      "text": "إجمالي الأصول ٨٬١٣٧٬٣٩٤ مليون جنيه..."
    }
  ],
  "retrieval_info": {
    "units_retrieved": 15,
    "context_length": 1234,
    "pages_referenced": [3, 4]
  }
}
```

**Response Schema:**
| Field | Type | Description |
|-------|------|-------------|
| answer | string | Answer in formal Arabic |
| citations | array | List of citation objects |
| retrieval_info | object | Metadata about retrieval (optional) |

**Citation Object:**
| Field | Type | Description |
|-------|------|-------------|
| page | integer | Page number from source PDF |
| text | string | Excerpt from that page |

---

### 3. Interactive Documentation

**GET** `/docs`

Access Swagger UI for interactive API testing.

---

## Example Usage

### cURL

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ما هي الأصول في ٢٠٢٤؟"
  }'
```

### Python (httpx)

```python
import httpx

response = httpx.post(
    "http://localhost:8000/ask",
    json={"question": "ما هي الأصول في ٢٠٢٤؟"}
)

data = response.json()
print(data["answer"])
for citation in data["citations"]:
    print(f"  - صفحة {citation['page']}")
```

### JavaScript (fetch)

```javascript
fetch('http://localhost:8000/ask', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    question: 'ما هي الأصول في ٢٠٢٤؟'
  })
})
.then(res => res.json())
.then(data => {
  console.log(data.answer);
  data.citations.forEach(cit => {
    console.log(`صفحة ${cit.page}: ${cit.text}`);
  });
});
```

---

## Error Handling

### 400 Bad Request
Invalid request parameters (e.g., empty question).

```json
{
  "error": "http_error",
  "message": "خطأ في البيانات المدخلة: question too short",
  "status_code": 400
}
```

### 500 Internal Server Error
Server-side error (database, LLM, etc.).

```json
{
  "error": "internal_error",
  "message": "حدث خطأ داخلي. الرجاء المحاولة مرة أخرى.",
  "detail": "Connection timeout"
}
```

---

## Question Types Supported

### 1. General Semantic Queries
Questions about concepts without specific numbers.

**Example:**
```
ما هي الاستثمارات المالية للبنك؟
```

**Retrieval:** Pure semantic search

---

### 2. Numeric Queries
Questions containing years, amounts, percentages.

**Example:**
```
كم بلغت الأصول في ٣١ ديسمبر ٢٠٢٤؟
```

**Retrieval:** Semantic + numeric filtering on JSONB

---

### 3. Comparison Queries
Questions comparing multiple time periods.

**Example:**
```
قارن بين القروض في ٢٠٢٤ و ٢٠٢٣
```

**Retrieval:** Multiple numeric filters + semantic

---

### 4. Missing Data
Questions about unavailable information.

**Example:**
```
ما هي الأرباح في ٢٠٢٥؟
```

**Expected Response:**
```
المعلومة غير موجودة في المستندات المتاحة
```

---

## Rate Limits

### Groq API Limits
- Free tier: Limited requests per minute
- If exceeded, API returns error message in Arabic
- Implement client-side retry with exponential backoff

### Recommended Client Behavior
- Wait 1-2 seconds between requests
- Implement retry logic for 500 errors
- Cache responses when possible

---

## Performance

| Metric | Value |
|--------|-------|
| Average response time | 2-5 seconds |
| Semantic search | ~50ms |
| LLM generation | ~1-4s |
| Paragraph expansion | ~100ms |

---

## Security Notes

> [!WARNING]
> **Current Version**: No authentication implemented. Phase 3 will add JWT tokens.

> [!IMPORTANT]
> **CORS**: Currently allows all origins (`*`). Restrict in production.

---

## Development Mode

### Enable Debug Logging

Set in `api/.env`:
```
API_DEBUG=True
```

This enables:
- Detailed error messages
- Request/response logging
- Auto-reload on code changes

---

## Next Steps (Phase 3)

- [ ] JWT authentication
- [ ] Rate limiting middleware
- [ ] Response caching (Redis)
- [ ] Multi-document support
- [ ] Streaming responses
