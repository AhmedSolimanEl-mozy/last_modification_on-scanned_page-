#!/usr/bin/env python3
"""
table_pipeline.py — Table-Aware Extraction Pipeline
=====================================================

Master orchestrator that:

  1. SEGMENTS each page into TABLE_BLOCK + TEXT_BLOCK regions.
  2. OCR strategy per region:
       TEXT_BLOCK  → Surya OCR as-is (line-level, with language model)
       TABLE_BLOCK → high-DPI char-level OCR (no LM for numbers)
       TABLE_BLOCK (native) → PyMuPDF rawdict char extraction
  3. RECONSTRUCTS table structure from character-level data:
       glyphs → row clusters → column clusters → cell grid
  4. HANDLES RTL + LTR:
       Arabic text in RTL, numbers as LTR islands, bbox-anchored.
  5. OUTPUTS:
       • table_output.jsonl  — structured per-page layout + tables
       • tables_review.html  — standalone QA review page
       • table_pipeline_log.json — per-page decision log
  6. QA STEP:
       • Column sum verification
       • Low-confidence cell flagging
       • Numeric anomaly detection

Usage:
    python table_pipeline.py
    python table_pipeline.py --input "el-bankalahly .pdf"
    python table_pipeline.py --pages 4,5,6 --no-surya-layout
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import cv2
import fitz
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from table_segment import segment_page, PageRegion
from table_ocr import (
    ocr_table_region, extract_native_table_rawdict,
    render_page_at_dpi, clean_table_image,
    TableOCRResult, TableLine, TableGlyph,
    TEXT_DPI, TABLE_DPI,
)
from table_reconstruct import (
    reconstruct_table, table_grid_to_dict, TableGrid,
)
from table_render import (
    render_tables_standalone, compute_column_sums,
    detect_anomalies, identify_low_confidence_cells,
)

# ────────────────────────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PDF    = BASE_DIR / "el-bankalahly .pdf"
DEFAULT_OUTPUT = BASE_DIR / "table_output.jsonl"
DEFAULT_HTML   = BASE_DIR / "tables_review.html"
DEFAULT_LOG    = BASE_DIR / "table_pipeline_log.json"
DPI            = 300
NATIVE_THRESHOLD = 200


# ────────────────────────────────────────────────────────────────────
#  Surya OCR for TEXT_BLOCK (lazy loaded)
# ────────────────────────────────────────────────────────────────────
_surya_foundation = None
_surya_det = None
_surya_rec = None

def _load_surya_text():
    global _surya_foundation, _surya_det, _surya_rec
    if _surya_foundation is not None:
        return
    print("  [Surya] Loading OCR models (one-time)...", flush=True)
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    _surya_foundation = FoundationPredictor(device="cpu")
    _surya_det        = DetectionPredictor(device="cpu")
    _surya_rec        = RecognitionPredictor(_surya_foundation)
    print("  [Surya] OCR models loaded.\n", flush=True)


def _detect_direction(text: str) -> str:
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin  = sum(1 for c in text if 'A' <= c <= 'z')
    return "rtl" if arabic >= latin else "ltr"


# ────────────────────────────────────────────────────────────────────
#  Page classification
# ────────────────────────────────────────────────────────────────────
def is_native_page(fitz_page: fitz.Page) -> bool:
    """Check if page has meaningful native text."""
    text = fitz_page.get_text("text").strip()
    meaningful = sum(
        1 for c in text
        if ('\u0600' <= c <= '\u06FF') or ('A' <= c <= 'z') or ('0' <= c <= '9')
    )
    return meaningful >= NATIVE_THRESHOLD


# ────────────────────────────────────────────────────────────────────
#  Text block extraction (standard Surya OCR)
# ────────────────────────────────────────────────────────────────────
def extract_text_region_native(
    fitz_page: fitz.Page,
    region_bbox_pt: List[float],
    page_num: int,
) -> List[dict]:
    """Extract text lines from a native page TEXT_BLOCK using PyMuPDF."""
    x0, y0, x1, y1 = region_bbox_pt
    clip = fitz.Rect(x0, y0, x1, y1)
    d = fitz_page.get_text("dict", clip=clip)

    lines = []
    for blk in d.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for line in blk.get("lines", []):
            spans_data = []
            text_parts = []
            for span in line.get("spans", []):
                spans_data.append({
                    "text": span["text"],
                    "bbox": [round(v, 2) for v in span["bbox"]],
                    "font_size": round(span.get("size", 0), 2),
                    "font_name": span.get("font"),
                })
                text_parts.append(span["text"])
            full_text = "".join(text_parts)
            if not full_text.strip():
                continue
            lines.append({
                "line_id": len(lines),
                "bbox": [round(v, 2) for v in line["bbox"]],
                "text": full_text,
                "direction": _detect_direction(full_text),
                "spans": spans_data,
            })
    return lines


def extract_text_region_ocr(
    doc: fitz.Document,
    page_idx: int,
    region_bbox_pt: List[float],
) -> List[dict]:
    """Extract text lines from a scanned page TEXT_BLOCK using Surya OCR."""
    _load_surya_text()
    from PIL import Image

    page_img = render_page_at_dpi(doc, page_idx, DPI)

    # Crop region
    scale = DPI / 72.0
    x0_px = max(0, int(region_bbox_pt[0] * scale) - 8)
    y0_px = max(0, int(region_bbox_pt[1] * scale) - 8)
    x1_px = min(page_img.shape[1], int(region_bbox_pt[2] * scale) + 8)
    y1_px = min(page_img.shape[0], int(region_bbox_pt[3] * scale) + 8)
    crop = page_img[y0_px:y1_px, x0_px:x1_px]

    # Clean
    from table_ocr import clean_table_image
    cleaned = clean_table_image(crop)

    img_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    results = _surya_rec(
        [pil_img],
        task_names=["ocr_with_boxes"],
        det_predictor=_surya_det,
        sort_lines=True,
    )

    if not results or not results[0].text_lines:
        return []

    px_to_pt = 72.0 / DPI
    lines = []
    for tl in results[0].text_lines:
        poly = tl.polygon
        lxs = [p[0] for p in poly]
        lys = [p[1] for p in poly]
        line_bbox = [
            round((min(lxs) + x0_px) * px_to_pt, 2),
            round((min(lys) + y0_px) * px_to_pt, 2),
            round((max(lxs) + x0_px) * px_to_pt, 2),
            round((max(lys) + y0_px) * px_to_pt, 2),
        ]
        direction = _detect_direction(tl.text)
        lines.append({
            "line_id": len(lines),
            "bbox": line_bbox,
            "text": tl.text,
            "direction": direction,
            "spans": [{
                "text": tl.text,
                "bbox": line_bbox,
                "font_size": None,
                "font_name": None,
            }],
        })
    return lines


# ────────────────────────────────────────────────────────────────────
#  Process one page
# ────────────────────────────────────────────────────────────────────
def process_page(
    doc: fitz.Document,
    page_idx: int,
    use_surya_layout: bool = False,
) -> dict:
    """
    Process a single page through the table-aware pipeline.

    Returns a page dict with:
      - text blocks (standard lines)
      - table blocks (structured table JSON)
    """
    page_num = page_idx + 1
    fitz_page = doc[page_idx]
    pw = round(fitz_page.rect.width, 2)
    ph = round(fitz_page.rect.height, 2)
    native = is_native_page(fitz_page)

    t0 = time.time()

    # ── Render page for segmentation ────────────────────────────────
    page_img = render_page_at_dpi(doc, page_idx, DPI)

    # ── Segment page ────────────────────────────────────────────────
    regions = segment_page(doc, page_idx, page_img,
                           use_surya_layout=use_surya_layout)

    table_regions = [r for r in regions if r.region_type == "TABLE_BLOCK"]
    text_regions  = [r for r in regions if r.region_type == "TEXT_BLOCK"]

    # ── Process TEXT_BLOCK regions ──────────────────────────────────
    all_text_lines = []
    for tr in text_regions:
        if native:
            lines = extract_text_region_native(fitz_page, tr.bbox, page_num)
        else:
            lines = extract_text_region_ocr(doc, page_idx, tr.bbox)
        all_text_lines.extend(lines)

    # Re-number text lines
    for i, ln in enumerate(all_text_lines):
        ln["line_id"] = i

    # ── Process TABLE_BLOCK regions ─────────────────────────────────
    table_results = []
    for ti, tr in enumerate(table_regions):
        print(f"    Table region {ti+1}: "
              f"[{tr.bbox[0]:.0f},{tr.bbox[1]:.0f},"
              f"{tr.bbox[2]:.0f},{tr.bbox[3]:.0f}] "
              f"({tr.detection_method}) ", end="", flush=True)

        if native:
            # Native page: extract chars directly from PDF
            ocr_result = extract_native_table_rawdict(
                fitz_page, tr.bbox, page_num
            )
        else:
            # Scanned page: high-DPI char-level OCR
            ocr_result = ocr_table_region(
                doc, page_idx, tr.bbox, dpi=TABLE_DPI
            )

        # Reconstruct table structure
        grid = reconstruct_table(ocr_result, page_number=page_num)
        table_dict = table_grid_to_dict(grid)
        table_dict["region_detection"] = tr.detection_method
        table_dict["region_confidence"] = tr.confidence
        table_dict["has_grid_lines"] = tr.has_grid_lines
        table_dict["numeric_density"] = tr.numeric_density
        table_results.append({
            "grid": grid,
            "dict": table_dict,
        })

        print(f"→ {grid.num_rows}×{grid.num_cols} "
              f"({len([c for c in grid.cells if c.text.strip()])} cells, "
              f"conf={grid.confidence:.2f})", flush=True)

    elapsed = time.time() - t0

    # ── Build page output ───────────────────────────────────────────
    # Group text lines into blocks by proximity
    text_blocks = _group_lines_into_blocks(all_text_lines)

    page_output = {
        "page_number": page_num,
        "page_width": pw,
        "page_height": ph,
        "dpi": 72 if native else DPI,
        "extraction_method": "native" if native else "surya_ocr",
        "processing_time": round(elapsed, 1),
        "regions": {
            "table_count": len(table_regions),
            "text_count": len(text_regions),
        },
        "blocks": text_blocks,
        "tables": [t["dict"] for t in table_results],
    }

    return page_output, [t["grid"] for t in table_results]


def _group_lines_into_blocks(lines: List[dict]) -> List[dict]:
    """Group text lines into blocks by vertical proximity."""
    if not lines:
        return []

    lines.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    blocks = []
    cur_lines = [lines[0]]

    for prev_ln, cur_ln in zip(lines, lines[1:]):
        gap = cur_ln["bbox"][1] - prev_ln["bbox"][3]
        h = prev_ln["bbox"][3] - prev_ln["bbox"][1]
        if gap <= max(h * 1.5, 5.0):
            cur_lines.append(cur_ln)
        else:
            blocks.append(_make_block(len(blocks), cur_lines))
            cur_lines = [cur_ln]
    blocks.append(_make_block(len(blocks), cur_lines))
    return blocks


def _make_block(block_id: int, lines: List[dict]) -> dict:
    x0 = min(ln["bbox"][0] for ln in lines)
    y0 = min(ln["bbox"][1] for ln in lines)
    x1 = max(ln["bbox"][2] for ln in lines)
    y1 = max(ln["bbox"][3] for ln in lines)
    return {
        "block_id": block_id,
        "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        "lines": lines,
    }


# ────────────────────────────────────────────────────────────────────
#  QA Summary
# ────────────────────────────────────────────────────────────────────
def qa_summary(all_grids: List[TableGrid]) -> dict:
    """Generate QA summary across all tables."""
    total_cells = 0
    numeric_cells = 0
    low_conf_cells = 0
    anomaly_count = 0
    col_sums_all = []

    for grid in all_grids:
        for cell in grid.cells:
            if cell.text.strip():
                total_cells += 1
                if cell.is_numeric:
                    numeric_cells += 1
        low_conf_cells += len(identify_low_confidence_cells(grid))
        anomaly_count += len(detect_anomalies(grid))
        sums = compute_column_sums(grid)
        for col_id, data in sums.items():
            col_sums_all.append({
                "page": grid.page_number,
                "col": col_id,
                "sum": round(data["sum"], 2),
                "count": data["count"],
            })

    return {
        "total_tables": len(all_grids),
        "total_cells": total_cells,
        "numeric_cells": numeric_cells,
        "low_confidence_cells": low_conf_cells,
        "numeric_anomalies": anomaly_count,
        "column_sums": col_sums_all,
    }


# ────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Table-aware extraction pipeline.")
    parser.add_argument("--input", "-i", default=str(DEFAULT_PDF))
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--html", default=str(DEFAULT_HTML))
    parser.add_argument("--log", "-l", default=str(DEFAULT_LOG))
    parser.add_argument("--pages", "-p", default=None,
                        help="Comma-separated page numbers (default: all)")
    args = parser.parse_args()

    pdf_path = Path(args.input)
    out_jsonl = Path(args.output)
    out_html  = Path(args.html)
    log_path  = Path(args.log)

    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.", file=sys.stderr)
        sys.exit(1)

    page_filter = None
    if args.pages:
        page_filter = {int(x.strip()) for x in args.pages.split(",")}

    # Surya layout is no longer used; detection is native_band vs cv_scan
    use_surya_layout = False

    t_start = time.time()

    print("=" * 64)
    print("  TABLE-AWARE EXTRACTION PIPELINE")
    print(f"  Input:  {pdf_path.name}")
    print(f"  Output: {out_jsonl.name}")
    print(f"  HTML:   {out_html.name}")
    print(f"  Layout: native_band (PyMuPDF) + cv_scan (OpenCV)")
    print("=" * 64)

    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count
    print(f"\n  PDF: {total_pages} pages\n")

    # Determine which pages to process
    page_indices = list(range(total_pages))
    if page_filter:
        page_indices = [i for i in page_indices if (i + 1) in page_filter]
    print(f"  Processing {len(page_indices)} pages: "
          f"{[i+1 for i in page_indices]}\n")

    # Process pages
    all_pages = []
    all_grids: List[TableGrid] = []
    page_decisions = []

    for page_idx in page_indices:
        page_num = page_idx + 1
        native = is_native_page(doc[page_idx])
        label = "native " if native else "scanned"
        print(f"  Page {page_num:2d}/{total_pages}  [{label}]", flush=True)

        page_output, grids = process_page(doc, page_idx,
                                           use_surya_layout=use_surya_layout)
        all_pages.append(page_output)
        all_grids.extend(grids)

        n_tables = len(grids)
        n_lines = sum(len(b.get("lines", [])) for b in page_output["blocks"])
        n_chars = sum(
            len(ln.get("text", ""))
            for b in page_output["blocks"]
            for ln in b.get("lines", [])
        )
        print(f"    → {n_lines} text lines, {n_chars} chars, "
              f"{n_tables} tables  ({page_output['processing_time']:.1f}s)\n")

        page_decisions.append({
            "page": page_num,
            "method": page_output["extraction_method"],
            "text_lines": n_lines,
            "text_chars": n_chars,
            "table_count": n_tables,
            "processing_time": page_output["processing_time"],
        })

    doc.close()

    # ── Write JSONL output ──────────────────────────────────────────
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for page in all_pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")
    jsonl_kb = out_jsonl.stat().st_size / 1024
    print(f"  JSONL: {out_jsonl} ({jsonl_kb:.1f} KB)")

    # ── Render standalone QA HTML ───────────────────────────────────
    if all_grids:
        render_tables_standalone(all_grids, str(out_html), show_qa=True)
        html_kb = out_html.stat().st_size / 1024
        print(f"  HTML:  {out_html} ({html_kb:.1f} KB)")
    else:
        print("  HTML:  (no tables found — skipping)")

    # ── QA Summary ──────────────────────────────────────────────────
    qa = qa_summary(all_grids)
    print(f"\n─── QA Summary ───")
    print(f"  Total tables:          {qa['total_tables']}")
    print(f"  Total cells:           {qa['total_cells']}")
    print(f"  Numeric cells:         {qa['numeric_cells']}")
    print(f"  Low-confidence cells:  {qa['low_confidence_cells']}")
    print(f"  Numeric anomalies:     {qa['numeric_anomalies']}")

    # ── Write log ───────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    log_data = {
        "pipeline": "table_pipeline.py",
        "input": pdf_path.name,
        "total_pages_processed": len(page_indices),
        "total_time_s": round(elapsed_total, 1),
        "use_surya_layout": use_surya_layout,
        "qa": qa,
        "per_page": page_decisions,
    }
    log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2),
                        encoding='utf-8')
    print(f"  Log:   {log_path}")

    print(f"\n{'=' * 64}")
    print(f"  ✓ Done in {elapsed_total:.1f}s")
    print(f"  ✓ {len(all_grids)} tables reconstructed across "
          f"{len(page_indices)} pages")
    print(f"  ✓ Output: {out_jsonl.name}")
    if all_grids:
        print(f"  ✓ Review: {out_html.name}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
