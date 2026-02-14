#!/usr/bin/env python3
"""
smart_pipeline.py — Optimized PDF reconstruction pipeline.

Strategy
========
1. CLASSIFY every page as NATIVE_TEXT or IMAGE_ONLY.
2. NATIVE_TEXT pages → copy directly from original PDF
   (preserving fonts, vectors, annotations, exact byte content).
3. IMAGE_ONLY pages → Surya OCR → normalize → render HTML → Playwright
   PDF export → insert reconstructed page.
4. MERGE all pages in original order into final PDF.
5. LOG per-page decisions as JSON for auditability.

This avoids running heavy OCR on pages that already contain
extractable vector text, saving ~95% of processing time for
this PDF (15 native vs 3 scanned).

Quality checks
--------------
  • Page count: output == input
  • Dimensions: each page matches original (±2 pt)
  • No empty pages

Usage
-----
    python smart_pipeline.py
    python smart_pipeline.py --input "el-bankalahly .pdf" --output smart_output.pdf
    python smart_pipeline.py --threshold 200
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import html as html_mod
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

import cv2
import fitz  # PyMuPDF

# ────────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PDF       = BASE_DIR / "el-bankalahly .pdf"
DEFAULT_OUTPUT    = BASE_DIR / "smart_output.pdf"
DEFAULT_LOG       = BASE_DIR / "smart_pipeline_log.json"
DPI               = 300
NATIVE_THRESHOLD  = 200     # min meaningful chars for NATIVE_TEXT
MIN_TEXT_BLOCKS   = 2       # min text blocks for NATIVE_TEXT

# Layout normalization
NORM_MARGIN       = 10      # pt padding beyond content bbox
FONT_SIZE_FACTOR  = 0.85    # bbox_height × factor → estimated font-size
MIN_LINE_GAP      = 1.0     # pt minimum vertical gap
OVERLAP_THRESH    = 2.0     # px overlap threshold for collisions

# HTML rendering
FONT_DRIFT_FACTOR = 1.25    # RTL width expansion for font metric drift
FONT_FAMILY       = '"Arial", "Helvetica", "Noto Sans Arabic", sans-serif'


# ────────────────────────────────────────────────────────────────────
#  Page Classification
# ────────────────────────────────────────────────────────────────────
@dataclass
class PageDecision:
    page_number: int
    page_type: str            # "NATIVE_TEXT" | "IMAGE_ONLY"
    action: str               # "copy_original" | "ocr_reconstruct"
    meaningful_chars: int
    text_blocks: int
    text_area_pct: float
    original_width: float
    original_height: float
    processing_time: float = 0.0
    notes: str = ""


def classify_page(page: fitz.Page, threshold: int = NATIVE_THRESHOLD) -> tuple:
    """Classify a page and return (type, chars, text_blocks, area_pct)."""
    text = page.get_text("text").strip()
    meaningful = sum(
        1 for c in text
        if ('\u0600' <= c <= '\u06FF')
        or ('\u0750' <= c <= '\u077F')
        or ('A' <= c <= 'Z')
        or ('a' <= c <= 'z')
        or ('0' <= c <= '9')
    )

    blks = page.get_text("dict")["blocks"]
    text_blks = [b for b in blks if b.get("type") == 0]

    # Text area coverage
    ta = sum(
        (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
        for b in text_blks
    )
    pa = page.rect.width * page.rect.height
    cov = (ta / pa * 100) if pa > 0 else 0.0

    if meaningful >= threshold and len(text_blks) >= MIN_TEXT_BLOCKS:
        ptype = "NATIVE_TEXT"
    else:
        ptype = "IMAGE_ONLY"

    return ptype, meaningful, len(text_blks), round(cov, 1)


# ────────────────────────────────────────────────────────────────────
#  Surya OCR (lazy-loaded)
# ────────────────────────────────────────────────────────────────────
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
    """Run Surya OCR with bboxes. Returns list of TextLine objects."""
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


# ────────────────────────────────────────────────────────────────────
#  Image preprocessing
# ────────────────────────────────────────────────────────────────────
def render_page_image(doc: fitz.Document, page_idx: int) -> np.ndarray:
    """Render a PDF page to BGR numpy array at target DPI."""
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


# ────────────────────────────────────────────────────────────────────
#  OCR → Layout JSONL (one page)
# ────────────────────────────────────────────────────────────────────
def detect_direction(text: str) -> str:
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin  = sum(1 for c in text if 'A' <= c <= 'z')
    return "rtl" if arabic >= latin else "ltr"


def pixel_to_points(bbox: list, dpi: int) -> list:
    scale = 72.0 / dpi
    return [round(v * scale, 2) for v in bbox]


def extract_scanned_page(doc: fitz.Document, page_idx: int) -> dict:
    """OCR a scanned page → layout JSONL dict."""
    page = doc[page_idx]
    pw, ph = round(page.rect.width, 2), round(page.rect.height, 2)

    img     = render_page_image(doc, page_idx)
    cleaned = clean_image(img)
    text_lines = surya_ocr_with_layout(cleaned)

    lines = []
    for i, tl in enumerate(text_lines):
        pdf_bbox = pixel_to_points(tl.bbox, DPI)
        direction = detect_direction(tl.text)
        lines.append({
            "line_id": i,
            "bbox": pdf_bbox,
            "text": tl.text,
            "direction": direction,
            "spans": [{
                "text": tl.text,
                "bbox": pdf_bbox,
                "font_size": None,
                "font_name": None,
            }],
        })

    # Group lines into blocks by spatial proximity
    lines.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    blocks = []
    if lines:
        cur = [lines[0]]
        for prev, cur_ln in zip(lines, lines[1:]):
            gap = cur_ln["bbox"][1] - prev["bbox"][3]
            h   = prev["bbox"][3] - prev["bbox"][1]
            if gap <= max(h * 1.5, 5.0):
                cur.append(cur_ln)
            else:
                blocks.append(_make_block_dict(len(blocks), cur))
                cur = [cur_ln]
        blocks.append(_make_block_dict(len(blocks), cur))

    # Sort reading order: top→bottom, RTL within bands
    blocks = _sort_reading_order(blocks)

    return {
        "page_number": page_idx + 1,
        "page_width": pw,
        "page_height": ph,
        "dpi": DPI,
        "extraction_method": "surya_ocr",
        "blocks": blocks,
    }


def _make_block_dict(block_id: int, lines: list) -> dict:
    x0 = min(ln["bbox"][0] for ln in lines)
    y0 = min(ln["bbox"][1] for ln in lines)
    x1 = max(ln["bbox"][2] for ln in lines)
    y1 = max(ln["bbox"][3] for ln in lines)
    # Re-number lines within block
    for i, ln in enumerate(lines):
        ln["line_id"] = i
    return {
        "block_id": block_id,
        "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        "lines": lines,
    }


def _sort_reading_order(blocks: list) -> list:
    if not blocks:
        return blocks
    BAND_TOL = 15.0
    blocks.sort(key=lambda b: (b["bbox"][1] + b["bbox"][3]) / 2)
    bands, cur = [[blocks[0]]], (blocks[0]["bbox"][1] + blocks[0]["bbox"][3]) / 2
    for blk in blocks[1:]:
        mid = (blk["bbox"][1] + blk["bbox"][3]) / 2
        if abs(mid - cur) <= BAND_TOL:
            bands[-1].append(blk)
        else:
            bands.append([blk])
            cur = mid
    out = []
    for band in bands:
        band.sort(key=lambda b: -b["bbox"][2])
        out.extend(band)
    for i, blk in enumerate(out):
        blk["block_id"] = i
    return out


# ────────────────────────────────────────────────────────────────────
#  Normalize layout (one page)
# ────────────────────────────────────────────────────────────────────
def normalize_page(page: dict) -> dict:
    """Apply normalization: clamping, font stabilization, collision resolution."""
    page = copy.deepcopy(page)
    corrections = []

    all_lines = [ln for b in page["blocks"] for ln in b["lines"]]
    if not all_lines:
        page["_norm"] = {"corrections": [], "version": 1}
        return page

    pw = page["page_width"]
    ph = page["page_height"]

    # Content bbox → dynamic page sizing
    cx1 = max(ln["bbox"][2] for ln in all_lines)
    cy1 = max(ln["bbox"][3] for ln in all_lines)
    needed_w = cx1 + NORM_MARGIN
    needed_h = cy1 + NORM_MARGIN
    new_w = max(pw, needed_w)
    new_h = max(ph, needed_h)
    if abs(new_w - pw) > 0.5 or abs(new_h - ph) > 0.5:
        corrections.append({
            "type": "page_resize",
            "old": [pw, ph],
            "new": [round(new_w, 2), round(new_h, 2)],
        })
        page["page_width"] = round(new_w, 2)
        page["page_height"] = round(new_h, 2)
        pw, ph = page["page_width"], page["page_height"]

    # Coordinate clamping
    for b in page["blocks"]:
        for ln in b["lines"]:
            x0, y0, x1, y1 = ln["bbox"]
            nb = [x0, y0, x1, y1]
            changed = False
            if x0 < 0:
                nb[0] = 0; changed = True
            if y0 < 0:
                nb[1] = 0; changed = True
            if x1 > pw:
                shift = x1 - pw + 1
                nb[0] = max(0, nb[0] - shift)
                nb[2] = pw - 1
                changed = True
            if y1 > ph:
                shift = y1 - ph + 1
                nb[1] = max(0, nb[1] - shift)
                nb[3] = ph - 1
                changed = True
            if changed:
                corrections.append({
                    "type": "clamp", "line_id": ln.get("line_id"),
                    "old": [round(v, 2) for v in [x0, y0, x1, y1]],
                    "new": [round(v, 2) for v in nb],
                })
                ln["bbox"] = [round(v, 2) for v in nb]

    # Font-size stabilization
    for b in page["blocks"]:
        for ln in b["lines"]:
            has_font = any(sp.get("font_size") for sp in ln.get("spans", []))
            if not has_font:
                _, y0, _, y1 = ln["bbox"]
                est_fs = max((y1 - y0) * FONT_SIZE_FACTOR, 4.0)
                for sp in ln.get("spans", []):
                    sp["font_size"] = round(est_fs, 2)

    # Collision resolution
    flat = [ln for b in page["blocks"] for ln in b["lines"]]
    flat.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    for i in range(1, len(flat)):
        ln = flat[i]
        x0, y0, x1, y1 = ln["bbox"]
        h = y1 - y0
        for j in range(i - 1, max(i - 30, -1), -1):
            prev = flat[j]
            px0, py0, px1, py1 = prev["bbox"]
            if x0 >= px1 or x1 <= px0:
                continue
            if y0 >= py1:
                continue
            ov_x = min(x1, px1) - max(x0, px0)
            ov_y = py1 - y0
            if ov_x > OVERLAP_THRESH and ov_y > OVERLAP_THRESH:
                nudge = ov_y + MIN_LINE_GAP
                corrections.append({
                    "type": "collision_nudge",
                    "line_id": ln.get("line_id"),
                    "nudge_y": round(nudge, 2),
                })
                ln["bbox"][1] = round(py1 + MIN_LINE_GAP, 2)
                ln["bbox"][3] = round(ln["bbox"][1] + h, 2)
                y0, y1 = ln["bbox"][1], ln["bbox"][3]

    # Update block bboxes
    for b in page["blocks"]:
        if b["lines"]:
            b["bbox"] = [
                round(min(ln["bbox"][0] for ln in b["lines"]), 2),
                round(min(ln["bbox"][1] for ln in b["lines"]), 2),
                round(max(ln["bbox"][2] for ln in b["lines"]), 2),
                round(max(ln["bbox"][3] for ln in b["lines"]), 2),
            ]

    page["_norm"] = {"version": 1, "corrections_count": len(corrections),
                     "corrections": corrections}
    return page


# ────────────────────────────────────────────────────────────────────
#  HTML rendering (single page)
# ────────────────────────────────────────────────────────────────────
def _get_font_size(line: dict) -> float:
    for sp in line.get("spans", []):
        fs = sp.get("font_size")
        if fs:
            return fs
    _, y0, _, y1 = line["bbox"]
    return max((y1 - y0) * FONT_SIZE_FACTOR, 4.0)


def _render_line_html(line: dict) -> str:
    x0, y0, x1, y1 = line["bbox"]
    bbox_w = x1 - x0
    height = y1 - y0
    direction = line.get("direction", "rtl")
    dir_class = "rtl" if direction == "rtl" else "ltr"
    font_size = _get_font_size(line)

    if direction == "rtl":
        expanded_w = bbox_w * FONT_DRIFT_FACTOR
        extra = expanded_w - bbox_w
        left = max(0, x0 - extra)
        render_w = x1 - left
    else:
        left = x0
        render_w = bbox_w

    line_height = max(height, font_size)
    raw = line.get("text", "")
    inner = html_mod.escape(raw)

    # Bold / italic detection from font name
    spans = line.get("spans", [])
    if spans and spans[0].get("font_name"):
        fname = spans[0]["font_name"]
        if "Bold" in fname:
            inner = f"<b>{inner}</b>"
        if "Italic" in fname:
            inner = f"<i>{inner}</i>"

    return (
        f'<div class="line {dir_class}" '
        f'style="left:{left:.2f}px; top:{y0:.2f}px; '
        f'width:{render_w:.2f}px; height:{line_height:.2f}px; '
        f'line-height:{line_height:.2f}px; font-size:{font_size:.2f}px;">'
        f'{inner}</div>'
    )


def render_single_page_html(page: dict) -> str:
    """Render one page to a standalone HTML string for Playwright export."""
    pw = page["page_width"]
    ph = page["page_height"]

    # Sort blocks for reading order
    blocks = sorted(page["blocks"],
                    key=lambda b: (b["bbox"][1], -b["bbox"][0]))
    line_divs = []
    for blk in blocks:
        for ln in blk.get("lines", []):
            line_divs.append(_render_line_html(ln))

    lines_html = "\n    ".join(line_divs)
    drift_pct = int((FONT_DRIFT_FACTOR - 1) * 100)

    return f"""<!DOCTYPE html>
<html lang="ar" dir="ltr">
<head>
<meta charset="utf-8">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: {FONT_FAMILY};
  direction: ltr;
  margin: 0;
  padding: 0;
}}
.page {{
  position: relative;
  width: {pw:.2f}px;
  height: {ph:.2f}px;
  background: #ffffff;
  overflow: visible;
}}
.line {{
  position: absolute;
  white-space: pre;
}}
.line.rtl {{
  direction: rtl;
  text-align: right;
  unicode-bidi: plaintext;
}}
.line.ltr {{
  direction: ltr;
  text-align: left;
  unicode-bidi: plaintext;
}}
@media print {{
  body {{ margin: 0; }}
  .page {{ overflow: visible; }}
}}
</style>
</head>
<body>
<div class="page">
  {lines_html}
</div>
</body>
</html>"""


# ────────────────────────────────────────────────────────────────────
#  HTML → PDF via Playwright (single page)
# ────────────────────────────────────────────────────────────────────
async def html_to_pdf_page(html_str: str, width_pt: float, height_pt: float,
                           output_path: Path):
    """Convert a single-page HTML string to a PDF file via Playwright."""
    from playwright.async_api import async_playwright

    w_in = width_pt / 72
    h_in = height_pt / 72

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(html_str, wait_until="networkidle")
        await page.pdf(
            path=str(output_path),
            width=f"{w_in:.4f}in",
            height=f"{h_in:.4f}in",
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            print_background=True,
            scale=1.0,
        )
        await browser.close()


# ────────────────────────────────────────────────────────────────────
#  Main Pipeline
# ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Smart PDF pipeline: skip OCR for native pages.")
    parser.add_argument("--input", "-i", default=str(DEFAULT_PDF))
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--log", "-l", default=str(DEFAULT_LOG))
    parser.add_argument("--threshold", type=int, default=NATIVE_THRESHOLD,
                        help="Minimum meaningful chars for NATIVE_TEXT")
    args = parser.parse_args()

    threshold = args.threshold

    pdf_path = Path(args.input)
    out_path = Path(args.output)
    log_path = Path(args.log)

    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.", file=sys.stderr)
        sys.exit(1)

    t_start = time.time()

    print("=" * 64)
    print("  SMART PDF PIPELINE")
    print(f"  Input:  {pdf_path.name}")
    print(f"  Output: {out_path.name}")
    print(f"  Threshold: {NATIVE_THRESHOLD} chars")
    print("=" * 64)

    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count
    print(f"\n  PDF loaded: {total_pages} pages\n")

    # ── Phase 1: Classify all pages ──────────────────────────────────
    print("─── Phase 1: Page Classification ───")
    decisions: List[PageDecision] = []
    native_pages = []
    scanned_pages = []

    for i in range(total_pages):
        page = doc[i]
        ptype, chars, tblks, cov = classify_page(page, threshold)
        action = "copy_original" if ptype == "NATIVE_TEXT" else "ocr_reconstruct"

        dec = PageDecision(
            page_number=i + 1,
            page_type=ptype,
            action=action,
            meaningful_chars=chars,
            text_blocks=tblks,
            text_area_pct=cov,
            original_width=round(page.rect.width, 2),
            original_height=round(page.rect.height, 2),
        )
        decisions.append(dec)

        label = "NATIVE " if ptype == "NATIVE_TEXT" else "IMAGE  "
        print(f"  P{i+1:2d}  {label}  chars={chars:5d}  "
              f"blocks={tblks:2d}  area={cov:5.1f}%  → {action}")

        if ptype == "NATIVE_TEXT":
            native_pages.append(i)
        else:
            scanned_pages.append(i)

    print(f"\n  Summary: {len(native_pages)} native (copy) + "
          f"{len(scanned_pages)} scanned (OCR)")

    # ── Phase 2: Process scanned pages with OCR ─────────────────────
    # We build per-page PDFs for scanned pages, store in a dict.
    tmp_dir = Path(tempfile.mkdtemp(prefix="smart_pipeline_"))
    scanned_pdfs: dict[int, Path] = {}  # page_idx → temp PDF path

    if scanned_pages:
        print(f"\n─── Phase 2: OCR Reconstruction ({len(scanned_pages)} pages) ───")
        for page_idx in scanned_pages:
            pnum = page_idx + 1
            t0 = time.time()
            print(f"\n  Page {pnum}: render → clean → OCR ... ", end="", flush=True)

            # 2a. Extract with Surya OCR
            page_layout = extract_scanned_page(doc, page_idx)
            n_lines = sum(len(b["lines"]) for b in page_layout["blocks"])
            n_chars = sum(
                len(ln["text"])
                for b in page_layout["blocks"]
                for ln in b["lines"]
            )
            print(f"{n_lines} lines, {n_chars} chars", flush=True)

            # 2b. Normalize
            print(f"  Page {pnum}: normalizing ... ", end="", flush=True)
            page_norm = normalize_page(page_layout)
            nc = page_norm["_norm"]["corrections_count"]
            print(f"{nc} corrections", flush=True)

            # 2c. Render HTML
            print(f"  Page {pnum}: rendering HTML ... ", end="", flush=True)
            page_html = render_single_page_html(page_norm)
            print("done", flush=True)

            # 2d. Export to PDF via Playwright
            print(f"  Page {pnum}: Playwright PDF export ... ", end="", flush=True)
            tmp_pdf = tmp_dir / f"page_{pnum:03d}.pdf"
            asyncio.run(html_to_pdf_page(
                page_html,
                page_norm["page_width"],
                page_norm["page_height"],
                tmp_pdf,
            ))
            scanned_pdfs[page_idx] = tmp_pdf
            elapsed = time.time() - t0
            decisions[page_idx].processing_time = round(elapsed, 1)
            decisions[page_idx].notes = (
                f"{n_lines} lines, {n_chars} chars, {nc} corrections"
            )
            print(f"done ({elapsed:.1f}s)", flush=True)
    else:
        print("\n─── Phase 2: No scanned pages — skipping OCR ───")

    # ── Phase 3: Merge all pages ─────────────────────────────────────
    print(f"\n─── Phase 3: Merging {total_pages} pages ───")
    merged = fitz.open()

    for page_idx in range(total_pages):
        pnum = page_idx + 1
        if page_idx in scanned_pdfs:
            # Insert reconstructed OCR page
            src = fitz.open(str(scanned_pdfs[page_idx]))
            merged.insert_pdf(src)
            src.close()
            print(f"  P{pnum:2d}: inserted OCR-reconstructed page")
        else:
            # Copy original page directly (preserves fonts, vectors, etc.)
            merged.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
            t0 = time.time()
            decisions[page_idx].processing_time = round(time.time() - t0, 4)
            print(f"  P{pnum:2d}: copied from original (native)")

    merged.save(str(out_path))
    merged.close()
    doc.close()

    # ── Phase 4: Quality Checks ──────────────────────────────────────
    print(f"\n─── Phase 4: Quality Checks ───")
    check_doc = fitz.open(str(out_path))
    all_ok = True

    # Check page count
    if check_doc.page_count != total_pages:
        print(f"  ✗ Page count mismatch: expected {total_pages}, "
              f"got {check_doc.page_count}")
        all_ok = False
    else:
        print(f"  ✓ Page count: {check_doc.page_count} (matches original)")

    # Check dimensions
    orig_doc = fitz.open(str(pdf_path))
    dim_issues = 0
    for i in range(total_pages):
        ow, oh = orig_doc[i].rect.width, orig_doc[i].rect.height
        nw, nh = check_doc[i].rect.width, check_doc[i].rect.height
        if abs(ow - nw) > 2 or abs(oh - nh) > 2:
            print(f"  ✗ P{i+1}: dims {nw:.1f}×{nh:.1f} vs "
                  f"original {ow:.1f}×{oh:.1f}")
            dim_issues += 1
    orig_doc.close()

    if dim_issues == 0:
        print(f"  ✓ All page dimensions match (±2pt tolerance)")
    else:
        print(f"  ✗ {dim_issues} pages with dimension mismatches")
        all_ok = False

    # Check for empty pages
    empty = 0
    for i in range(check_doc.page_count):
        pg = check_doc[i]
        # A page is not empty if it has text OR images OR drawings
        has_text = len(pg.get_text().strip()) > 0
        has_images = len(pg.get_images()) > 0
        has_drawings = len(pg.get_drawings()) > 0
        if not (has_text or has_images or has_drawings):
            print(f"  ✗ P{i+1}: appears empty")
            empty += 1
    if empty == 0:
        print(f"  ✓ No empty pages")
    else:
        all_ok = False

    out_kb = out_path.stat().st_size / 1024
    print(f"\n  Output: {out_path}  ({out_kb:.1f} KB)")

    check_doc.close()

    # ── Phase 5: Save decision log ───────────────────────────────────
    elapsed_total = time.time() - t_start
    log_data = {
        "pipeline": "smart_pipeline.py",
        "input": str(pdf_path.name),
        "output": str(out_path.name),
        "total_pages": total_pages,
        "native_pages": len(native_pages),
        "scanned_pages": len(scanned_pages),
        "total_time_s": round(elapsed_total, 1),
        "quality_ok": all_ok,
        "decisions": [
            {
                "page": d.page_number,
                "type": d.page_type,
                "action": d.action,
                "meaningful_chars": d.meaningful_chars,
                "text_blocks": d.text_blocks,
                "text_area_pct": d.text_area_pct,
                "original_dims": [d.original_width, d.original_height],
                "processing_time_s": d.processing_time,
                "notes": d.notes,
            }
            for d in decisions
        ],
    }
    log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"  Log:    {log_path}")

    # ── Cleanup temp files ───────────────────────────────────────────
    for tf in scanned_pdfs.values():
        tf.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"  ✓ Done in {elapsed_total:.1f}s")
    print(f"  ✓ {len(native_pages)} pages copied directly (instant)")
    print(f"  ✓ {len(scanned_pages)} pages OCR-reconstructed")
    print(f"  ✓ Output: {out_path.name}  ({out_kb:.1f} KB)")
    if all_ok:
        print(f"  ✓ All quality checks passed")
    else:
        print(f"  ⚠ Some quality checks had issues — review log")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
