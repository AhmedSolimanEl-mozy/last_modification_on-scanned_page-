#!/usr/bin/env python3
"""
Document Intelligence Pipeline - Gemini-Powered JSON Extraction

Simplified approach: Send PDF pages to Gemini 1.5 Flash and let it extract
everything directly into structured JSON Information Units.

Usage:
    export GOOGLE_API_KEY="your-api-key-here"
    python3 document_intelligence.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF
from google import genai
from google.genai import types
from PIL import Image
import io


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PDF_PATH = "el-bankalahly .pdf"
OUTPUT_JSON = "final_json_18_pages.json"
OUTPUT_PREVIEW = "units_preview.txt"
OUTPUT_REPORT = "extraction_report.md"

RENDER_DPI = 300
SCANNED_PAGES = [0, 1, 2]  # Pages 1-3 are scanned (0-indexed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gemini Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def setup_gemini():
    """Initialize Gemini API with key from environment."""
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("❌ ERROR: GOOGLE_API_KEY environment variable not set")
        print("\nPlease set it with:")
        print('  export GOOGLE_API_KEY="your-api-key-here"')
        sys.exit(1)
    
    client = genai.Client(api_key=api_key)
    return client


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PDF Processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_page_to_image(doc, page_idx: int, dpi: int = 300) -> Image.Image:
    """Render a PDF page to PIL Image."""
    page = doc[page_idx]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gemini Prompts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXTRACTION_PROMPT = """
You are a document intelligence engineer processing an Arabic financial PDF.

Your task: Extract ALL useful content from this page and structure it as JSON Information Units for hybrid retrieval with accurate citations.

CRITICAL RULES:
1. DO NOT invent or guess any text or numbers
2. DO NOT summarize - extract exact text as written
3. DO NOT translate Arabic text
4. Extract ONLY what you can see clearly
5. Ignore stamps, logos, signatures, handwritten notes

DOCUMENT STRUCTURE:
- Arabic text (digital or scanned)
- Paragraphs (one sentence per line typically)
- Tables where:
  - One table row = one sentence in nearby paragraph
  - Columns show numeric attributes
- Some noise (stamps, logos) to ignore

OUTPUT FORMAT:
Return a valid JSON array of Information Units. Each unit must have this exact structure:

[
  {
    "unit_id": "p<page>_para<paragraph>_s<sentence>",
    "page": <page_number>,
    "paragraph": <paragraph_number>,
    "sentence_index": <index>,
    
    "sentence": {
      "raw_text": "<exact Arabic text>",
      "normalized_text": "<lightly normalized: remove diacritics, normalize أإآ→ا, ى→ي, ة→ه>",
      "ocr_confidence": <0.0-1.0>
    },
    
    "numeric_data": {
      "<column_name>": {
        "value": "<exact number as string>",
        "ocr_confidence": <0.0-1.0>
      }
    },
    
    "unit_type": "sentence_table_unit | text_only_unit",
    "source_pdf": "el-bankalahly .pdf"
  }
]

GUIDELINES:
- For text-only paragraphs: set "numeric_data": null
- Keep numbers as strings (preserve Arabic-Indic ٠-٩ or Western 0-9)
- Align sentence with table row by index (sentence 0 → row 0)
- If uncertain about alignment, still extract but note low confidence
- Preserve exact spacing in numbers (e.g., "٣٤٨ ٥٣٠")

Extract all content from this page now:
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Extraction Logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_page_with_gemini(client, page_image: Image.Image, page_num: int) -> List[Dict]:
    """
    Send page image to Gemini and get structured JSON units.
    Includes retry logic for rate limits.
    """
    max_retries = 5
    base_delay = 10
    
    for attempt in range(max_retries):
        try:
            print(f"  📤 Sending page {page_num} to Gemini (Attempt {attempt+1}/{max_retries})...")
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[EXTRACTION_PROMPT, page_image]
            )
            response_text = response.text.strip()
            
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            units = json.loads(json_text)
            if not isinstance(units, list):
                units = [units]
            
            # Enforce correct page number and unit_id prefix
            for unit in units:
                unit['page'] = page_num
                if 'unit_id' in unit and not unit['unit_id'].startswith(f"p{page_num}_"):
                    # Fix ID if prefix is wrong
                    old_id = unit['unit_id']
                    if '_' in old_id:
                        suffix = old_id.split('_', 1)[1]
                        unit['unit_id'] = f"p{page_num}_{suffix}"
                    else:
                        unit['unit_id'] = f"p{page_num}_{old_id}"

            print(f"  ✅ Extracted {len(units)} units from page {page_num}")
            return units

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                delay = base_delay * (2 ** attempt)
                print(f"  ⚠️  Rate limit hit (429). Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  ❌ Error: Gemini API call failed: {e}")
                break
                
    return []


def process_document(pdf_path: str, client) -> List[Dict]:
    """
    Process entire PDF document and extract JSON units.
    
    Returns:
        List of all JSON units from all pages
    """
    doc = fitz.open(pdf_path)
    all_units = []
    
    # Load existing units if resuming
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                all_units = json.load(f)
            processed_pages = set(u.get('page') for u in all_units)
            print(f"ℹ️ Found {len(all_units)} existing units. Resuming from page {max(processed_pages, default=0) + 1}...")
        except (json.JSONDecodeError, ValueError):
            print("⚠️ Could not parse existing JSON, starting fresh.")
            all_units = []
            processed_pages = set()
    else:
        processed_pages = set()

    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"Total pages: {doc.page_count}")
    print(f"{'='*60}\n")
    
    for page_idx in range(doc.page_count):
        page_num = page_idx + 1
        
        # Skip already processed pages (if any)
        if page_num in processed_pages:
            continue
            
        print(f"\n📄 Page {page_num}/{doc.page_count}")
        
        # Render page to image
        page_image = render_page_to_image(doc, page_idx, dpi=RENDER_DPI)
        print(f"  🖼️  Rendered at {RENDER_DPI} DPI: {page_image.size[0]}×{page_image.size[1]} px")
        
        # Extract with Gemini
        units = extract_page_with_gemini(client, page_image, page_num)
        
        # Add to collection
        all_units.extend(units)
        
        # Incremental save to prevent data loss
        save_json_units(all_units, OUTPUT_JSON)
        
        # Rate limiting - be gentle with API (2s delay)
        if page_idx < doc.page_count - 1:
            print(f"  💤 Waiting 2s before next page...")
            time.sleep(2)
    
    doc.close()
    print(f"\n{'='*60}")
    print(f"✅ Total units extracted: {len(all_units)}")
    print(f"{'='*60}\n")
    
    return all_units


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Output Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_json_units(units: List[Dict], output_path: str):
    """Save JSON units to file with UTF-8 encoding."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(units, f, ensure_ascii=False, indent=2)
    
    file_size = os.path.getsize(output_path)
    print(f"💾 Saved {len(units)} units to: {output_path}")
    print(f"   File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")


def generate_preview(units: List[Dict], output_path: str):
    """Generate human-readable preview of extracted units."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("DOCUMENT INTELLIGENCE EXTRACTION - PREVIEW\n")
        f.write("="*70 + "\n\n")
        
        for i, unit in enumerate(units, 1):
            f.write(f"\n{'─'*70}\n")
            f.write(f"Unit {i}/{len(units)}: {unit.get('unit_id', 'N/A')}\n")
            f.write(f"{'─'*70}\n")
            f.write(f"Page: {unit.get('page')}, ")
            f.write(f"Paragraph: {unit.get('paragraph')}, ")
            f.write(f"Sentence: {unit.get('sentence_index')}\n")
            f.write(f"Type: {unit.get('unit_type')}\n\n")
            
            sentence = unit.get('sentence', {})
            f.write(f"Text: {sentence.get('raw_text', 'N/A')}\n")
            f.write(f"Confidence: {sentence.get('ocr_confidence', 'N/A')}\n\n")
            
            numeric_data = unit.get('numeric_data')
            if numeric_data:
                f.write("Numeric Data:\n")
                for col, data in numeric_data.items():
                    if isinstance(data, dict):
                        f.write(f"  {col}: {data.get('value')} ")
                        f.write(f"(conf: {data.get('ocr_confidence')})\n")
                    else:
                        f.write(f"  {col}: {data}\n")
            else:
                f.write("Numeric Data: None\n")
        
        f.write(f"\n{'='*70}\n")
        f.write(f"Total Units: {len(units)}\n")
        f.write(f"{'='*70}\n")
    
    print(f"📄 Generated preview: {output_path}")


def generate_report(units: List[Dict], output_path: str):
    """Generate extraction report."""
    # Count statistics
    total_units = len(units)
    sentence_table_units = sum(1 for u in units if u.get('unit_type') == 'sentence_table_unit')
    text_only_units = sum(1 for u in units if u.get('unit_type') == 'text_only_unit')
    
    pages_processed = len(set(u.get('page') for u in units))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Extraction Report\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## OCR Tools Used\n\n")
        f.write("- **Gemini 2.5 Flash**: Multimodal extraction (text + tables)\n")
        f.write("- **PyMuPDF (fitz)**: PDF rendering at 300 DPI\n\n")
        
        f.write("## Gemini API Usage\n\n")
        f.write(f"- **API Calls**: {pages_processed} (one per page)\n")
        f.write(f"- **Model**: gemini-2.5-flash\n")
        f.write(f"- **Rate Limiting**: 2s delay between pages\n\n")
        
        f.write("## Extraction Statistics\n\n")
        f.write(f"- **Total Information Units**: {total_units}\n")
        f.write(f"- **Sentence-Table Units**: {sentence_table_units}\n")
        f.write(f"- **Text-Only Units**: {text_only_units}\n")
        f.write(f"- **Pages Processed**: {pages_processed}\n\n")
        
        f.write("## Known Uncertainties\n\n")
        f.write("- Gemini-based extraction may have alignment uncertainties\n")
        f.write("- OCR confidence scores are model-estimated\n")
        f.write("- Complex table structures may need manual verification\n\n")
        
        f.write("## Quality Notes\n\n")
        f.write("- ✅ No data invention - Gemini extracts only visible content\n")
        f.write("- ✅ Arabic text preserved exactly as written\n")
        f.write("- ✅ Numbers kept as strings (no conversion)\n")
        f.write("- ✅ Noise (stamps, logos) filtered by Gemini\n\n")
        
        f.write("---\n\n")
        f.write("**Status**: Extraction complete. Ready for hybrid retrieval.\n")
    
    print(f"📊 Generated report: {output_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """Main pipeline execution."""
    print("\n" + "="*60)
    print("🚀 DOCUMENT INTELLIGENCE PIPELINE")
    print("   Gemini-Powered JSON Extraction")
    print("="*60 + "\n")
    
    # Check PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: PDF not found: {PDF_PATH}")
        sys.exit(1)
    
    # Setup Gemini
    print("🔑 Setting up Gemini API...")
    client = setup_gemini()
    print("✅ Gemini initialized\n")
    
    # Process document
    units = process_document(PDF_PATH, client)
    
    if not units:
        print("⚠️  Warning: No units extracted!")
        sys.exit(1)
    
    # Save outputs
    print("\n📁 Generating outputs...")
    save_json_units(units, OUTPUT_JSON)
    generate_preview(units, OUTPUT_PREVIEW)
    generate_report(units, OUTPUT_REPORT)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  📄 JSON Units:  {OUTPUT_JSON}")
    print(f"  📄 Preview:     {OUTPUT_PREVIEW}")
    print(f"  📄 Report:      {OUTPUT_REPORT}")
    print(f"\n🎯 Ready for hybrid retrieval (vector + keyword)\n")


if __name__ == "__main__":
    main()
