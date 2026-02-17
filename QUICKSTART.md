# Document Intelligence Pipeline - Quick Start

## Overview
Streamlined pipeline using Gemini 1.5 Flash to extract structured JSON Information Units from Arabic PDFs.

## Requirements
```bash
pip install google-generativeai PyMuPDF Pillow
```

## Setup
1. Get your Gemini API key from Google AI Studio
2. Set environment variable:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

## Usage
```bash
python3 document_intelligence.py
```

## What It Does
1. **Renders PDF pages** at 300 DPI
2. **Sends to Gemini 1.5 Flash** with structured extraction prompt
3. **Gemini extracts**:
   - Arabic text (exact, no translation)
   - Tables with numeric data
   - Sentence-row alignments
   - Filters noise (stamps, logos, signatures)
4. **Outputs**:
   - `final_json_units.json` - Structured units for retrieval
   - `units_preview.txt` - Human-readable preview
   - `extraction_report.md` - Quality report

## JSON Unit Format
```json
{
  "unit_id": "p3_para1_s2",
  "page": 3,
  "paragraph": 1,
  "sentence_index": 2,
  "sentence": {
    "raw_text": "بلغت الأصول الثابتة...",
    "normalized_text": "بلغت الاصول الثابته...",
    "ocr_confidence": 0.95
  },
  "numeric_data": {
    "القيمة": {
      "value": "٣٤٨ ٥٣٠",
      "ocr_confidence": 0.98
    }
  },
  "unit_type": "sentence_table_unit",
  "source_pdf": "el-bankalahly .pdf"
}
```

## Features
- ✅ No data invention - extracts only what's visible
- ✅ Preserves Arabic text exactly
- ✅ Numbers as strings (no conversion)
- ✅ Automatic noise filtering
- ✅ Rate-limited API calls (2s between pages)
- ✅ Graceful error handling

## Next Steps
After extraction, use units for:
- Vector embeddings (sentence text)
- Keyword search (numeric values)
- Hybrid retrieval with accurate citations
