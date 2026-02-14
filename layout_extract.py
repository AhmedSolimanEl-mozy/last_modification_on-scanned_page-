#!/usr/bin/env python3
"""
layout_extract.py — Layout-Aware Structured Extraction
=======================================================
Produces a JSONL file where each line is one page with:
  - Exact bounding boxes for every text element
  - Block → Line → Span hierarchy
  - Font metadata (native pages) or null (OCR pages)
  - Consistent coordinate system (PDF points, origin = top-left)
  - RTL / LTR direction detection
  - Reading-order preserved

Page classification is automatic (text density heuristic).
Native pages  → PyMuPDF  get_text("dict")
Scanned pages → render@300DPI → clean → Surya OCR (ocr_with_boxes)

Output
------
  layout_output.jsonl   — one JSON object per page (primary)
  layout_output.txt     — plain-text rendering (secondary)
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

import cv2
import fitz  # PyMuPDF

# ====================================================================
#  Configuration
# ====================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PDF_PATH    = os.path.join(BASE_DIR, "el-bankalahly .pdf")
OUT_JSONL   = os.path.join(BASE_DIR, "layout_output.jsonl")
OUT_TXT     = os.path.join(BASE_DIR, "layout_output.txt")
DPI         = 300
# Minimum extractable-text characters for a page to count as "native"
NATIVE_THRESHOLD = 200


# ====================================================================
#  Data model (mirrors the required JSON schema)
# ====================================================================
@dataclass
class Span:
    text: str
    bbox: List[float]                 # [x0, y0, x1, y1] in PDF points
    font_size: Optional[float] = None
    font_name: Optional[str] = None

@dataclass
class Line:
    line_id: int
    bbox: List[float]
    text: str
    direction: str = "rtl"            # "rtl" or "ltr"
    spans: List[Span] = field(default_factory=list)

@dataclass
class Block:
    block_id: int
    bbox: List[float]
    lines: List[Line] = field(default_factory=list)

@dataclass
class PageLayout:
    page_number: int
    page_width: float                  # PDF points
    page_height: float                 # PDF points
    dpi: int
    extraction_method: str             # "native" | "surya_ocr"
    blocks: List[Block] = field(default_factory=list)


# ====================================================================
#  1. Page Classification
# ====================================================================
def classify_page(page: fitz.Page) -> str:
    """
    Determine if a page is native vector text or a scanned image.

    Heuristic:
      - Extract raw text via PyMuPDF
      - Count characters that are actual Arabic (U+0600-U+06FF) or Latin
      - If meaningful text < NATIVE_THRESHOLD → scanned
      - Also verify at least one text block of type=0 exists in dict mode

    Returns "native" or "scanned".
    """
    text = page.get_text("text").strip()
    # Count meaningful characters (Arabic + Latin alphanumeric)
    meaningful = sum(
        1 for c in text
        if ('\u0600' <= c <= '\u06FF')    # Arabic
        or ('\u0750' <= c <= '\u077F')    # Arabic Supplement
        or ('A' <= c <= 'Z')
        or ('a' <= c <= 'z')
        or ('0' <= c <= '9')
    )
    if meaningful < NATIVE_THRESHOLD:
        return "scanned"

    # Double-check: verify text blocks exist in structured mode
    d = page.get_text("dict")
    text_blocks = [b for b in d["blocks"] if b.get("type") == 0]
    if len(text_blocks) < 2:
        return "scanned"

    return "native"


# ====================================================================
#  2. Direction Detection
# ====================================================================
def detect_direction_from_pymupdf(dir_tuple: tuple) -> str:
    """PyMuPDF line 'dir' is (cos, sin) of the writing angle.
    For horizontal LTR: (1, 0).  For RTL Arabic, the dir is often
    still (1,0) because the PDF stores glyphs in visual order.
    We instead check the text content."""
    return "ltr"   # handled by text content below


def detect_direction_from_text(text: str) -> str:
    """Detect if text is predominantly RTL (Arabic) or LTR."""
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin  = sum(1 for c in text if 'A' <= c <= 'z')
    return "rtl" if arabic >= latin else "ltr"


# ====================================================================
#  3. Native Page Extraction (PyMuPDF)
# ====================================================================
def extract_native_page(page: fitz.Page, page_num: int) -> PageLayout:
    """
    Extract structured layout from a native PDF page using
    page.get_text("dict").  Preserves blocks, lines, spans, fonts,
    bboxes.  All coordinates are already in PDF points.
    """
    d = page.get_text("dict")
    pw = d["width"]
    ph = d["height"]

    layout = PageLayout(
        page_number=page_num,
        page_width=round(pw, 2),
        page_height=round(ph, 2),
        dpi=72,  # native PDF is 72 ppi by definition
        extraction_method="native",
    )

    block_id = 0
    for raw_block in d["blocks"]:
        # Skip image blocks (type=1)
        if raw_block.get("type") != 0:
            continue

        blk_bbox = _round_bbox(raw_block["bbox"])
        block = Block(block_id=block_id, bbox=blk_bbox)

        line_id = 0
        for raw_line in raw_block.get("lines", []):
            ln_bbox = _round_bbox(raw_line["bbox"])

            spans: List[Span] = []
            full_text_parts = []
            for raw_span in raw_line.get("spans", []):
                sp_text = raw_span["text"]
                full_text_parts.append(sp_text)
                spans.append(Span(
                    text=sp_text,
                    bbox=_round_bbox(raw_span["bbox"]),
                    font_size=round(raw_span.get("size", 0), 2),
                    font_name=raw_span.get("font"),
                ))

            full_text = "".join(full_text_parts)
            direction = detect_direction_from_text(full_text)

            line = Line(
                line_id=line_id,
                bbox=ln_bbox,
                text=full_text,
                direction=direction,
                spans=spans,
            )
            block.lines.append(line)
            line_id += 1

        if block.lines:
            layout.blocks.append(block)
            block_id += 1

    return layout


# ====================================================================
#  4. Scanned Page Preprocessing
# ====================================================================
def render_page(doc: fitz.Document, page_idx: int) -> np.ndarray:
    """Render a PDF page to a BGR numpy array at the target DPI."""
    page = doc[page_idx]
    zoom = DPI / 72.0
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
               pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def clean_image(img: np.ndarray) -> np.ndarray:
    """Background removal + CLAHE + bilateral denoise + sharpen."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    bg_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, bg_kernel)
    diff  = cv2.subtract(background, gray)
    diff  = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    clean = cv2.bitwise_not(diff)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    clean = clahe.apply(clean)
    clean = cv2.bilateralFilter(clean, d=9, sigmaColor=75, sigmaSpace=75)
    blurred = cv2.GaussianBlur(clean, (0, 0), sigmaX=3)
    sharp   = cv2.addWeighted(clean, 1.5, blurred, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


# ====================================================================
#  5. Surya OCR with Layout (Models loaded once)
# ====================================================================
_surya_foundation = None
_surya_det = None
_surya_rec = None


def _load_surya():
    global _surya_foundation, _surya_det, _surya_rec
    if _surya_foundation is not None:
        return
    print("  [Surya] Loading models (one-time)...", flush=True)
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    _surya_foundation = FoundationPredictor(device="cpu")
    _surya_det        = DetectionPredictor(device="cpu")
    _surya_rec        = RecognitionPredictor(_surya_foundation)
    print("  [Surya] Models loaded.\n", flush=True)


def surya_ocr_with_layout(img_bgr: np.ndarray):
    """
    Run Surya OCR with bounding boxes.
    Returns list of TextLine objects with .text, .bbox, .polygon, .confidence.
    Coordinates are in pixel space at the rendered DPI.
    """
    _load_surya()
    from PIL import Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    results = _surya_rec(
        [pil_img],
        task_names=["ocr_with_boxes"],
        det_predictor=_surya_det,
    )
    return results[0].text_lines if results else []


# ====================================================================
#  6. Coordinate Normalization (pixel → PDF points)
# ====================================================================
def pixel_to_pdf_points(bbox: List[float], dpi: int) -> List[float]:
    """Convert a pixel-space bbox (at given DPI) to PDF points (72 ppi)."""
    scale = 72.0 / dpi
    return [round(v * scale, 2) for v in bbox]


# ====================================================================
#  7. Scanned Page Extraction (Surya OCR → unified schema)
# ====================================================================
def extract_scanned_page(
    doc: fitz.Document, page_idx: int, page_num: int
) -> PageLayout:
    """
    Render → clean → Surya OCR (with boxes) → map to PDF coordinates.

    Groups text lines into blocks using a spatial proximity heuristic:
    lines whose vertical gap < 1.5× line height belong to the same block.
    """
    page = doc[page_idx]
    pw = page.rect.width    # PDF points
    ph = page.rect.height

    # Render and clean
    img = render_page(doc, page_idx)
    cleaned = clean_image(img)

    # OCR with bounding boxes (pixel coords at DPI)
    text_lines = surya_ocr_with_layout(cleaned)

    layout = PageLayout(
        page_number=page_num,
        page_width=round(pw, 2),
        page_height=round(ph, 2),
        dpi=DPI,
        extraction_method="surya_ocr",
    )

    if not text_lines:
        return layout

    # Convert each Surya text_line to our Line dataclass
    parsed_lines: List[Line] = []
    for i, tl in enumerate(text_lines):
        # tl.bbox is [x0, y0, x1, y1] in pixel space
        pdf_bbox = pixel_to_pdf_points(tl.bbox, DPI)
        direction = detect_direction_from_text(tl.text)

        line = Line(
            line_id=i,
            bbox=pdf_bbox,
            text=tl.text,
            direction=direction,
            spans=[Span(
                text=tl.text,
                bbox=pdf_bbox,
                font_size=None,
                font_name=None,
            )],
        )
        parsed_lines.append(line)

    # ── Group lines into blocks (spatial proximity) ──
    # Sort by vertical position (top of bbox), then horizontal
    parsed_lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))

    blocks: List[Block] = []
    current_block_lines: List[Line] = [parsed_lines[0]]

    for prev_ln, cur_ln in zip(parsed_lines, parsed_lines[1:]):
        prev_bottom = prev_ln.bbox[3]
        cur_top     = cur_ln.bbox[1]
        prev_height = prev_ln.bbox[3] - prev_ln.bbox[1]
        gap = cur_top - prev_bottom

        # Same block if vertical gap < 1.5× the previous line's height
        threshold = max(prev_height * 1.5, 5.0)
        if gap <= threshold:
            current_block_lines.append(cur_ln)
        else:
            blocks.append(_make_block(len(blocks), current_block_lines))
            current_block_lines = [cur_ln]

    # Flush last block
    blocks.append(_make_block(len(blocks), current_block_lines))

    # Re-number lines within each block
    for blk in blocks:
        for i, ln in enumerate(blk.lines):
            ln.line_id = i

    layout.blocks = blocks
    return layout


def _make_block(block_id: int, lines: List[Line]) -> Block:
    """Create a Block from a list of Lines, computing the union bbox."""
    x0 = min(ln.bbox[0] for ln in lines)
    y0 = min(ln.bbox[1] for ln in lines)
    x1 = max(ln.bbox[2] for ln in lines)
    y1 = max(ln.bbox[3] for ln in lines)
    return Block(
        block_id=block_id,
        bbox=[round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        lines=lines,
    )


# ====================================================================
#  8. Reading Order Sorting
# ====================================================================
def sort_blocks_reading_order(layout: PageLayout) -> None:
    """
    Sort blocks in top-to-bottom, right-to-left reading order
    (appropriate for predominantly Arabic RTL documents).

    Strategy:
      1. Bin blocks into horizontal bands (rows) — blocks whose
         vertical midpoints are within ±BAND_TOLERANCE of each other.
      2. Within each band, sort right-to-left (descending x0).
      3. Bands are ordered top-to-bottom.
    """
    if not layout.blocks:
        return

    BAND_TOLERANCE = 15.0   # PDF points (~5mm)

    # Assign band index by clustering vertical midpoints
    blocks = layout.blocks[:]
    blocks.sort(key=lambda b: (b.bbox[1] + b.bbox[3]) / 2)

    bands: List[List[Block]] = []
    current_band = [blocks[0]]
    current_mid  = (blocks[0].bbox[1] + blocks[0].bbox[3]) / 2

    for blk in blocks[1:]:
        mid = (blk.bbox[1] + blk.bbox[3]) / 2
        if abs(mid - current_mid) <= BAND_TOLERANCE:
            current_band.append(blk)
        else:
            bands.append(current_band)
            current_band = [blk]
            current_mid = mid
    bands.append(current_band)

    # Sort within each band: right-to-left (desc x) for RTL documents
    sorted_blocks = []
    for band in bands:
        band.sort(key=lambda b: -b.bbox[2])   # rightmost first
        sorted_blocks.extend(band)

    # Re-assign block IDs
    for i, blk in enumerate(sorted_blocks):
        blk.block_id = i

    layout.blocks = sorted_blocks


# ====================================================================
#  9. Serialization
# ====================================================================
def layout_to_dict(layout: PageLayout) -> dict:
    """Convert a PageLayout dataclass tree to a plain dict for JSON."""
    return {
        "page_number":      layout.page_number,
        "page_width":       layout.page_width,
        "page_height":      layout.page_height,
        "dpi":              layout.dpi,
        "extraction_method": layout.extraction_method,
        "blocks": [
            {
                "block_id": blk.block_id,
                "bbox":     blk.bbox,
                "lines": [
                    {
                        "line_id":   ln.line_id,
                        "bbox":      ln.bbox,
                        "text":      ln.text,
                        "direction": ln.direction,
                        "spans": [
                            {
                                "text":      sp.text,
                                "bbox":      sp.bbox,
                                "font_size": sp.font_size,
                                "font_name": sp.font_name,
                            }
                            for sp in ln.spans
                        ],
                    }
                    for ln in blk.lines
                ],
            }
            for blk in layout.blocks
        ],
    }


def layout_to_plaintext(layout: PageLayout) -> str:
    """Render a layout to plain text (secondary artifact)."""
    parts = [f"═══ PAGE {layout.page_number} "
             f"({layout.extraction_method}, "
             f"{layout.page_width}×{layout.page_height} pt) ═══"]
    for blk in layout.blocks:
        for ln in blk.lines:
            parts.append(ln.text)
        parts.append("")    # blank line between blocks
    return "\n".join(parts)


# ====================================================================
#  Utilities
# ====================================================================
def _round_bbox(bbox) -> List[float]:
    """Round bbox tuple/list to 2 decimal places."""
    return [round(float(v), 2) for v in bbox]


# ====================================================================
#  Main Pipeline
# ====================================================================
def main():
    t_start = time.time()

    print("=" * 64)
    print("  LAYOUT-AWARE STRUCTURED EXTRACTION")
    print(f"  PDF:  {os.path.basename(PDF_PATH)}")
    print(f"  JSONL: {OUT_JSONL}")
    print(f"  TXT:   {OUT_TXT}")
    print("=" * 64)

    if not os.path.exists(PDF_PATH):
        print(f"\n  [ERROR] PDF not found: {PDF_PATH}")
        sys.exit(1)

    doc = fitz.open(PDF_PATH)
    total_pages = doc.page_count
    print(f"\n  PDF loaded: {total_pages} pages\n")

    f_jsonl = open(OUT_JSONL, "w", encoding="utf-8")
    f_txt   = open(OUT_TXT,   "w", encoding="utf-8")

    stats = {"native": 0, "scanned": 0,
             "total_blocks": 0, "total_lines": 0, "total_chars": 0}

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]
        t0 = time.time()

        # ── Classify ──
        page_type = classify_page(page)
        label = "native " if page_type == "native" else "scanned"
        print(f"  Page {page_num:2d}/{total_pages}  [{label}] ", end="", flush=True)

        # ── Extract ──
        if page_type == "native":
            layout = extract_native_page(page, page_num)
            stats["native"] += 1
        else:
            layout = extract_scanned_page(doc, page_idx, page_num)
            stats["scanned"] += 1

        # ── Sort reading order ──
        sort_blocks_reading_order(layout)

        # ── Serialise ──
        page_dict = layout_to_dict(layout)
        f_jsonl.write(json.dumps(page_dict, ensure_ascii=False) + "\n")
        f_jsonl.flush()

        plain = layout_to_plaintext(layout)
        f_txt.write(plain + "\n\n")
        f_txt.flush()

        elapsed = time.time() - t0
        n_blocks = len(layout.blocks)
        n_lines  = sum(len(b.lines) for b in layout.blocks)
        n_chars  = sum(len(ln.text) for b in layout.blocks for ln in b.lines)
        stats["total_blocks"] += n_blocks
        stats["total_lines"]  += n_lines
        stats["total_chars"]  += n_chars

        print(f"{elapsed:6.1f}s  "
              f"{n_blocks:3d} blocks  {n_lines:4d} lines  "
              f"{n_chars:5d} chars")

    doc.close()
    f_jsonl.close()
    f_txt.close()

    elapsed_total = time.time() - t_start

    print(f"\n{'=' * 64}")
    print(f"  ✓ Done in {elapsed_total:.1f}s")
    print(f"  ✓ Pages: {stats['native']} native + "
          f"{stats['scanned']} scanned = {total_pages}")
    print(f"  ✓ Total: {stats['total_blocks']} blocks, "
          f"{stats['total_lines']} lines, "
          f"{stats['total_chars']} chars")
    print(f"  ✓ JSONL: {OUT_JSONL}")
    print(f"  ✓ TXT:   {OUT_TXT}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
