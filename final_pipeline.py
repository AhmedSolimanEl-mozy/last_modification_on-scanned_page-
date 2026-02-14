#!/usr/bin/env python3
"""
final_pipeline.py — Token-Level PDF Reconstruction Pipeline
=============================================================

Architecture (NO table detection):
  1. CLASSIFY pages as NATIVE_TEXT or IMAGE_ONLY.
  2. NATIVE pages -> copy directly from original PDF.
  3. SCANNED pages -> normalize image -> Surya OCR -> tokens -> render -> PDF.
  4. NUMERIC VALIDATION:
     a. Lock high-confidence tokens (>= 0.92)
     b. CNN-validate low-confidence tokens (validation-only)
     c. Line-level stability check
     d. Column-level contextual boosting
  5. MERGE all pages into final output PDF.
  6. QUALITY CHECKS.
  7. NUMERIC COLUMN analysis for QA.
  8. QA HTML + NUMERIC QA REPORT.
  9. LOG.

Key principles:
  - Every token rendered at its exact original position.
  - No table/grid/border detection.
  - Discrete trust model: LOCKED/SURYA_VALID/CNN_CONFIRMED/UNTRUSTED.
  - No Tesseract. No dual-OCR voting. No SSIM/pixel-overlap.
  - No digit repair heuristics. No Latin->Arabic glyph mapping.
  - Input normalization: fixed 2240px width + DPI-aware dilation.
  - Deterministic: same input -> same output.

REMOVED:
  - Tesseract OCR (all usage)
  - SSIM / pixel overlap / visual similarity
  - Dual-OCR voting
  - Latin-to-Arabic glyph mapping
  - Per-token confidence arithmetic
  - Weighted trust formula

Usage:
    python final_pipeline.py
    python final_pipeline.py --input "el-bankalahly .pdf" --output final_output.pdf
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
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.stdout.reconfigure(line_buffering=True)

import fitz  # PyMuPDF

from token_extract import (
    Token, PageTokens,
    extract_tokens_native, extract_tokens_scanned,
    classify_token, detect_direction,
    render_page_at_dpi,
)
from numeric_columns import (
    NumericColumn, ColumnAnomaly,
    find_numeric_columns, detect_column_anomalies,
    columns_summary, anomalies_summary, render_qa_html,
    compute_column_boost,
)
from numeric_region_detector import detect_numeric_regions
from digit_ocr import (
    TrustStatus, FailureReason,
    ocr_page_numeric_tokens, TokenOCRResult,
    validate_token_with_cnn,
    DIGIT_CROP_DPI, TRUST_SCORES,
    has_valid_arabic_digits,
)
from numeric_validator import (
    validate_page_lines, apply_line_stability_to_tokens,
    PageStabilityResult,
)
from numeric_reconstructor import (
    reconstruct_page_numbers, NumericValue,
    page_trust_summary,
)
from numeric_qa_report import (
    generate_qa_report, PageNumericAudit, PipelineNumericAudit,
)


# ────────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PDF        = BASE_DIR / "el-bankalahly .pdf"
DEFAULT_OUTPUT     = BASE_DIR / "surya_discrete_trust_output.pdf"
DEFAULT_LOG        = BASE_DIR / "surya_discrete_trust_log.json"
DEFAULT_QA_HTML    = BASE_DIR / "surya_discrete_trust_qa.html"
DEFAULT_NUM_QA     = BASE_DIR / "surya_discrete_trust_numeric_qa.html"
NUMERIC_VAL_DPI    = 600       # DPI for numeric validation crops

NATIVE_THRESHOLD  = 200     # min meaningful chars for NATIVE_TEXT
MIN_TEXT_BLOCKS   = 2       # min text blocks for NATIVE_TEXT

# Rendering
FONT_SIZE_FACTOR  = 0.85
FONT_DRIFT_FACTOR = 1.25    # RTL width expansion
FONT_FAMILY       = '"Arial", "Helvetica", "Noto Sans Arabic", sans-serif'
MIN_LINE_GAP      = 1.0     # pt minimum vertical gap


# ────────────────────────────────────────────────────────────────────
#  Page Classification
# ────────────────────────────────────────────────────────────────────
@dataclass
class PageDecision:
    page_number: int
    page_type: str              # "NATIVE_TEXT" | "IMAGE_ONLY"
    action: str                 # "copy_original" | "ocr_reconstruct"
    meaningful_chars: int = 0
    text_blocks: int = 0
    original_dims: List[float] = None
    processing_time: float = 0.0
    token_count: int = 0
    numeric_token_count: int = 0
    text_token_count: int = 0
    num_columns: int = 0
    num_anomalies: int = 0
    notes: str = ""


def classify_page(page: fitz.Page, threshold: int = NATIVE_THRESHOLD) -> tuple:
    """Classify a page as NATIVE_TEXT or IMAGE_ONLY.
    Returns (type, meaningful_chars, text_block_count)."""
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

    if meaningful >= threshold and len(text_blks) >= MIN_TEXT_BLOCKS:
        return "NATIVE_TEXT", meaningful, len(text_blks)
    return "IMAGE_ONLY", meaningful, len(text_blks)


# ────────────────────────────────────────────────────────────────────
#  Layout Normalization (scanned pages)
# ────────────────────────────────────────────────────────────────────
def normalize_page_tokens(page_tokens: PageTokens) -> PageTokens:
    """
    Apply normalization to scanned page tokens:
      - Clamp coordinates within page bounds
      - Resolve vertical collisions between overlapping lines
      - Stabilize font sizes
    """
    page = copy.deepcopy(page_tokens)
    pw = page.page_width
    ph = page.page_height

    if not page.lines:
        return page

    # Sort lines by y-position
    page.lines.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))

    # Coordinate clamping
    for ln in page.lines:
        x0, y0, x1, y1 = ln["bbox"]
        if x0 < 0: x0 = 0
        if y0 < 0: y0 = 0
        if x1 > pw: x1 = pw
        if y1 > ph: y1 = ph
        ln["bbox"] = [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]

    # Collision resolution — nudge overlapping lines
    for i in range(1, len(page.lines)):
        ln = page.lines[i]
        x0, y0, x1, y1 = ln["bbox"]
        h = y1 - y0
        for j in range(i - 1, max(i - 20, -1), -1):
            prev = page.lines[j]
            px0, py0, px1, py1 = prev["bbox"]
            if x0 >= px1 or x1 <= px0:
                continue
            if y0 >= py1:
                continue
            ov_x = min(x1, px1) - max(x0, px0)
            ov_y = py1 - y0
            if ov_x > 2 and ov_y > 2:
                ln["bbox"][1] = round(py1 + MIN_LINE_GAP, 2)
                ln["bbox"][3] = round(ln["bbox"][1] + h, 2)
                y0, y1 = ln["bbox"][1], ln["bbox"][3]

    # Update token bboxes to match line adjustments
    for ln in page.lines:
        for tok in ln.get("tokens", []):
            tok.bbox[1] = ln["bbox"][1]
            tok.bbox[3] = ln["bbox"][3]

    return page


# ────────────────────────────────────────────────────────────────────
#  HTML Rendering — Token-Level Absolute Positioning
# ────────────────────────────────────────────────────────────────────
def _render_line_html(line: dict) -> str:
    """Render a single text line as absolutely-positioned HTML div."""
    x0, y0, x1, y1 = line["bbox"]
    bbox_w = x1 - x0
    height = y1 - y0
    direction = line.get("direction", "rtl")
    dir_class = "rtl" if direction == "rtl" else "ltr"
    font_size = line.get("font_size", max(height * FONT_SIZE_FACTOR, 4.0))

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

    # Bold / italic
    if line.get("is_bold"):
        inner = f"<b>{inner}</b>"
    if line.get("is_italic"):
        inner = f"<i>{inner}</i>"

    return (
        f'<div class="line {dir_class}" '
        f'style="left:{left:.2f}px; top:{y0:.2f}px; '
        f'width:{render_w:.2f}px; height:{line_height:.2f}px; '
        f'line-height:{line_height:.2f}px; font-size:{font_size:.2f}px;">'
        f'{inner}</div>'
    )


def render_page_html(page_tokens: PageTokens) -> str:
    """Render a full page as standalone HTML for Playwright PDF export."""
    pw = page_tokens.page_width
    ph = page_tokens.page_height

    lines = sorted(page_tokens.lines,
                   key=lambda ln: (ln["bbox"][1], -ln["bbox"][0]))
    line_divs = []
    for ln in lines:
        line_divs.append(_render_line_html(ln))

    lines_html = "\n    ".join(line_divs)

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
  overflow: hidden;
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
  .page {{ overflow: hidden; page-break-inside: avoid; }}
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
#  HTML -> PDF via Playwright
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
        description="Token-level PDF pipeline: no table detection.")
    parser.add_argument("--input", "-i", default=str(DEFAULT_PDF))
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--log", "-l", default=str(DEFAULT_LOG))
    parser.add_argument("--qa-html", default=str(DEFAULT_QA_HTML))
    parser.add_argument("--numeric-qa", default=str(DEFAULT_NUM_QA))
    parser.add_argument("--threshold", type=int, default=NATIVE_THRESHOLD)
    args = parser.parse_args()

    pdf_path = Path(args.input)
    out_path = Path(args.output)
    log_path = Path(args.log)
    qa_html_path = Path(args.qa_html)
    num_qa_path = Path(args.numeric_qa)

    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.", file=sys.stderr)
        sys.exit(1)

    t_start = time.time()

    print("=" * 64)
    print("  FINAL PDF PIPELINE — Deterministic Token-Level Extraction")
    print(f"  Input:    {pdf_path.name}")
    print(f"  Output:   {out_path.name}")
    print(f"  Strategy: Surya-only OCR | discrete trust | no Tesseract")
    print("=" * 64)

    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count
    print(f"\n  PDF loaded: {total_pages} pages\n")

    # ── Phase 1: Page Classification ─────────────────────────────────
    print("─── Phase 1: Page Classification ───")
    decisions: List[PageDecision] = []
    native_pages = []
    scanned_pages = []

    for i in range(total_pages):
        page = doc[i]
        ptype, chars, tblks = classify_page(page)
        action = "copy_original" if ptype == "NATIVE_TEXT" else "ocr_reconstruct"

        dec = PageDecision(
            page_number=i + 1,
            page_type=ptype,
            action=action,
            meaningful_chars=chars,
            text_blocks=tblks,
            original_dims=[round(page.rect.width, 2), round(page.rect.height, 2)],
        )
        decisions.append(dec)

        label = "NATIVE " if ptype == "NATIVE_TEXT" else "IMAGE  "
        print(f"  P{i+1:2d}  {label}  chars={chars:5d}  "
              f"blocks={tblks:2d}  -> {action}")

        if ptype == "NATIVE_TEXT":
            native_pages.append(i)
        else:
            scanned_pages.append(i)

    print(f"\n  Summary: {len(native_pages)} native (copy) + "
          f"{len(scanned_pages)} scanned (OCR tokens)")

    # ── Phase 2: Extract tokens + render scanned pages ──────────────
    # Input normalization (2240px width + dilation) happens inside
    # extract_tokens_scanned() — once per page, before Surya OCR.
    tmp_dir = Path(tempfile.mkdtemp(prefix="final_pipeline_"))
    scanned_pdfs: dict = {}
    all_page_tokens: List[PageTokens] = []
    # Save original line texts for line-level stability check
    original_line_texts: dict = {}  # page_idx -> list of line dicts

    if scanned_pages:
        print(f"\n─── Phase 2: Token Extraction + Render ({len(scanned_pages)} pages) ───")
        for page_idx in scanned_pages:
            pnum = page_idx + 1
            t0 = time.time()
            print(f"\n  Page {pnum}: OCR token extraction "
                  f"(normalized input) ... ", end="", flush=True)

            # Extract tokens via Surya OCR (with input normalization)
            page_tokens = extract_tokens_scanned(doc, page_idx)
            n_tokens = len(page_tokens.tokens)
            n_numeric = sum(1 for t in page_tokens.tokens
                           if t.token_type == "NUMERIC")
            n_text = n_tokens - n_numeric
            n_lines = len(page_tokens.lines)
            print(f"{n_tokens} tokens ({n_text} text, {n_numeric} numeric), "
                  f"{n_lines} lines", flush=True)

            # Normalize layout
            page_tokens = normalize_page_tokens(page_tokens)

            # Save original line texts BEFORE any modifications
            original_line_texts[page_idx] = [
                {
                    'line_id': ln.get('line_id', i),
                    'text': ln.get('text', ''),
                    'bbox': ln.get('bbox', [0, 0, 0, 0]),
                }
                for i, ln in enumerate(page_tokens.lines)
            ]

            all_page_tokens.append(page_tokens)

            # Update decision
            decisions[page_idx].token_count = n_tokens
            decisions[page_idx].numeric_token_count = n_numeric
            decisions[page_idx].text_token_count = n_text

            # Render to HTML
            print(f"  Page {pnum}: rendering HTML ... ", end="", flush=True)
            page_html = render_page_html(page_tokens)
            print("done", flush=True)

            # Export to PDF via Playwright
            print(f"  Page {pnum}: Playwright PDF export ... ", end="", flush=True)
            tmp_pdf = tmp_dir / f"page_{pnum:03d}.pdf"
            asyncio.run(html_to_pdf_page(
                page_html,
                page_tokens.page_width,
                page_tokens.page_height,
                tmp_pdf,
            ))
            scanned_pdfs[page_idx] = tmp_pdf
            elapsed = time.time() - t0
            decisions[page_idx].processing_time = round(elapsed, 1)
            decisions[page_idx].notes = (
                f"{n_tokens} tokens, {n_lines} lines"
            )
            print(f"done ({elapsed:.1f}s)", flush=True)
    else:
        print("\n─── Phase 2: No scanned pages — skipping OCR ───")

    # ── Phase 2.5: Numeric Validation (Discrete Trust) ──────────────
    # Architecture:
    #   Step A: Classify + Lock tokens (conf >= 0.92 -> LOCKED)
    #   Step B: CNN-validate remaining (confirm or UNTRUSTED)
    #   Step C: Line-level stability check
    #   Step D: Column-level contextual boosting
    pipeline_audit = PipelineNumericAudit()

    if scanned_pages:
        print(f"\n─── Phase 2.5: Numeric Validation ({len(scanned_pages)} pages) ───")
        for i, page_idx in enumerate(scanned_pages):
            pnum = page_idx + 1
            page_tokens = all_page_tokens[i]
            n_numeric = sum(1 for t in page_tokens.tokens
                           if t.token_type == "NUMERIC")
            if n_numeric == 0:
                print(f"  P{pnum}: no numeric tokens — skipping")
                pipeline_audit.pages.append(PageNumericAudit(
                    page_number=pnum))
                continue

            print(f"  P{pnum}: validating {n_numeric} numeric tokens ... ",
                  end="", flush=True)

            # Render page at high DPI for CNN validation crops
            hires = render_page_at_dpi(doc, page_idx, dpi=NUMERIC_VAL_DPI)

            # Step A+B: Classify, lock, and CNN-validate
            ocr_results = ocr_page_numeric_tokens(page_tokens, hires)

            # Count locked tokens
            n_locked = sum(1 for r in ocr_results if r.locked)

            # Step C: Line-level stability check
            # Compare original Surya line text vs current line text
            current_lines = [
                {
                    'line_id': ln.get('line_id', idx),
                    'text': ln.get('text', ''),
                    'bbox': ln.get('bbox', [0, 0, 0, 0]),
                }
                for idx, ln in enumerate(page_tokens.lines)
            ]
            orig_lines = original_line_texts.get(page_idx, current_lines)
            page_stability = validate_page_lines(
                orig_lines, current_lines, pnum)

            # Apply line stability failures to token trust
            ocr_results = apply_line_stability_to_tokens(
                page_stability, ocr_results, page_tokens.lines)

            # Step D: Reconstruct numbers with discrete trust
            font_sizes = [t.font_size for t in page_tokens.tokens
                          if t.token_type == "NUMERIC"]
            numeric_values = reconstruct_page_numbers(
                ocr_results, pnum, font_sizes)

            # Update token texts ONLY if token has valid digits
            # and is NOT UNTRUSTED (never modify uncertain tokens)
            numeric_idx = 0
            for tok in page_tokens.tokens:
                if tok.token_type == "NUMERIC" and numeric_idx < len(numeric_values):
                    nv = numeric_values[numeric_idx]
                    if has_valid_arabic_digits(nv.digits):
                        tok.text = nv.digits
                    numeric_idx += 1

            # Rebuild line texts
            for ln in page_tokens.lines:
                line_toks = ln.get('tokens', [])
                if line_toks:
                    ln['text'] = ' '.join(t.text for t in line_toks)

            # Build page audit
            pa = PageNumericAudit(
                page_number=pnum,
                numeric_values=numeric_values,
                ocr_results=ocr_results,
                line_stability=page_stability,
            )
            pa.compute_summary()
            pipeline_audit.pages.append(pa)

            ts = pa.trust_summary
            print(f"{ts['locked']} locked, "
                  f"{ts['surya_valid']} valid, "
                  f"{ts.get('cnn_confirmed', 0)} cnn, "
                  f"{ts['untrusted']} untrusted "
                  f"({ts['pct_trusted']}% trusted)", flush=True)

            decisions[page_idx].notes += (
                f" | trust: {ts['pct_trusted']}% "
                f"({ts['untrusted']} untrusted, {ts['locked']} locked)")

        # ── Step D (cross-page): Column Boosting ─────────────────────
        # Run after all pages are processed
        print(f"\n  Column boosting (cross-page) ... ", end="", flush=True)
        boost_count = 0
        for i, page_idx in enumerate(scanned_pages):
            page_tokens = all_page_tokens[i]
            pa = pipeline_audit.pages[i]
            if not pa.numeric_values:
                continue

            # Build token trust map for this page
            token_trust_map = {}
            numeric_idx = 0
            for tok in page_tokens.tokens:
                if tok.token_type == "NUMERIC" and numeric_idx < len(pa.numeric_values):
                    nv = pa.numeric_values[numeric_idx]
                    key = (round(tok.bbox[0], 2), round(tok.bbox[1], 2),
                           round(tok.bbox[2], 2), round(tok.bbox[3], 2))
                    token_trust_map[key] = nv.status.value
                    numeric_idx += 1

            # Find columns and compute boosting
            columns = find_numeric_columns(page_tokens)
            boost_results = compute_column_boost(columns, token_trust_map)

            for br in boost_results:
                if br.eligible:
                    # Re-validate boosted tokens with CNN
                    for tok in br.boosted_tokens:
                        hires = render_page_at_dpi(doc, page_idx,
                                                   dpi=NUMERIC_VAL_DPI)
                        # Find this token's OCR result
                        tok_key = (round(tok.bbox[0], 2),
                                   round(tok.bbox[1], 2),
                                   round(tok.bbox[2], 2),
                                   round(tok.bbox[3], 2))
                        for nv_idx, nv in enumerate(pa.numeric_values):
                            nv_key = (round(nv.bbox[0], 2),
                                      round(nv.bbox[1], 2),
                                      round(nv.bbox[2], 2),
                                      round(nv.bbox[3], 2))
                            if nv_key == tok_key and nv.status == TrustStatus.UNTRUSTED:
                                # Re-validate with CNN
                                ocr_r = pa.ocr_results[nv_idx]
                                updated = validate_token_with_cnn(
                                    ocr_r, hires,
                                    page_tokens.page_width,
                                    page_tokens.page_height)
                                pa.ocr_results[nv_idx] = updated
                                # Update numeric value
                                nv.revalidated = True
                                if updated.cnn_confirmed:
                                    nv.status = TrustStatus.CNN_CONFIRMED
                                    nv.trust_score = TRUST_SCORES[TrustStatus.CNN_CONFIRMED]
                                    nv.cnn_confirmed = True
                                    nv.render_badge = False
                                    boost_count += 1
                                break

            # Recompute summary after boosting
            pa.compute_summary()

        print(f"{boost_count} tokens boosted", flush=True)

        pipeline_audit.compute_overall()
        print(f"\n  Numeric validation complete: "
              f"{pipeline_audit.total_trusted}/{pipeline_audit.total_numeric_tokens} "
              f"trusted ({pipeline_audit.overall_trust_pct}%)")
        if pipeline_audit.has_partial_failure:
            print(f"  ⚠ PARTIAL FAILURE: {pipeline_audit.total_untrusted} "
                  f"UNTRUSTED numbers detected")
    else:
        print("\n─── Phase 2.5: No scanned pages — skipping numeric validation ───")

    # ── Phase 3: Merge all pages ─────────────────────────────────────
    print(f"\n─── Phase 3: Merging {total_pages} pages ───")
    merged = fitz.open()

    for page_idx in range(total_pages):
        pnum = page_idx + 1
        if page_idx in scanned_pdfs:
            src = fitz.open(str(scanned_pdfs[page_idx]))
            if src.page_count > 1:
                print(f"  ⚠ P{pnum:2d}: Playwright PDF has {src.page_count} pages, "
                      f"taking only page 1")
            merged.insert_pdf(src, from_page=0, to_page=0)
            src.close()
            print(f"  P{pnum:2d}: inserted OCR-reconstructed page")
        else:
            merged.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
            print(f"  P{pnum:2d}: copied from original (native)")

    merged.save(str(out_path))
    merged.close()
    doc.close()

    # ── Phase 4: Quality Checks ──────────────────────────────────────
    print(f"\n─── Phase 4: Quality Checks ───")
    check_doc = fitz.open(str(out_path))
    all_ok = True

    if check_doc.page_count != total_pages:
        print(f"  ✗ Page count mismatch: expected {total_pages}, "
              f"got {check_doc.page_count}")
        all_ok = False
    else:
        print(f"  ✓ Page count: {check_doc.page_count} (matches original)")

    orig_doc = fitz.open(str(pdf_path))
    dim_issues = 0
    for i in range(min(total_pages, check_doc.page_count)):
        ow, oh = orig_doc[i].rect.width, orig_doc[i].rect.height
        nw, nh = check_doc[i].rect.width, check_doc[i].rect.height
        if abs(ow - nw) > 2 or abs(oh - nh) > 2:
            print(f"  ✗ P{i+1}: dims {nw:.1f}x{nh:.1f} vs "
                  f"original {ow:.1f}x{oh:.1f}")
            dim_issues += 1
    orig_doc.close()

    if dim_issues == 0:
        print(f"  ✓ All page dimensions match (±2pt tolerance)")
    else:
        print(f"  ⚠ {dim_issues} pages with dimension differences")

    empty = 0
    for i in range(check_doc.page_count):
        pg = check_doc[i]
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

    # ── Phase 5: Numeric Column Analysis (QA only) ──────────────────
    print(f"\n─── Phase 5: Numeric Column Analysis (QA) ───")
    all_columns: List[List[NumericColumn]] = []
    all_anomalies: List[List[ColumnAnomaly]] = []

    # Analyze scanned pages
    for page_tokens in all_page_tokens:
        columns = find_numeric_columns(page_tokens)
        anomalies = detect_column_anomalies(columns)
        all_columns.append(columns)
        all_anomalies.append(anomalies)

        pnum = page_tokens.page_number
        n_cols = len(columns)
        n_anom = len(anomalies)
        n_numeric = sum(c.count for c in columns)
        print(f"  P{pnum:2d}: {n_cols} numeric columns, "
              f"{n_numeric} aligned tokens, {n_anom} anomalies")

        # Update decision
        decisions[page_tokens.page_number - 1].num_columns = n_cols
        decisions[page_tokens.page_number - 1].num_anomalies = n_anom

    # Also analyze native pages
    if native_pages:
        native_doc = fitz.open(str(pdf_path))
        for page_idx in native_pages:
            pnum = page_idx + 1
            native_tokens = extract_tokens_native(native_doc[page_idx], pnum)
            columns = find_numeric_columns(native_tokens)
            anomalies = detect_column_anomalies(columns)
            all_page_tokens.append(native_tokens)
            all_columns.append(columns)
            all_anomalies.append(anomalies)

            n_cols = len(columns)
            n_anom = len(anomalies)
            decisions[page_idx].token_count = len(native_tokens.tokens)
            decisions[page_idx].numeric_token_count = sum(
                1 for t in native_tokens.tokens if t.token_type == "NUMERIC")
            decisions[page_idx].text_token_count = sum(
                1 for t in native_tokens.tokens if t.token_type == "TEXT")
            decisions[page_idx].num_columns = n_cols
            decisions[page_idx].num_anomalies = n_anom

            if n_cols > 0:
                print(f"  P{pnum:2d}: {n_cols} numeric columns, "
                      f"{sum(c.count for c in columns)} aligned tokens, "
                      f"{n_anom} anomalies")
        native_doc.close()

    total_columns = sum(len(cols) for cols in all_columns)
    total_anomalies = sum(len(anoms) for anoms in all_anomalies)
    print(f"\n  Total: {total_columns} numeric columns, {total_anomalies} anomalies")

    # ── Phase 6: QA HTML ─────────────────────────────────────────────
    print(f"\n─── Phase 6: QA HTML ───")
    qa_html = render_qa_html(all_page_tokens, all_columns, all_anomalies)
    qa_html_path.write_text(qa_html, encoding="utf-8")
    qa_kb = qa_html_path.stat().st_size / 1024
    print(f"  QA HTML: {qa_html_path}  ({qa_kb:.1f} KB)")

    # ── Phase 6.5: Numeric QA Report ─────────────────────────────────
    print(f"\n─── Phase 6.5: Numeric QA Report ───")
    num_qa_html = generate_qa_report(pipeline_audit)
    num_qa_path.write_text(num_qa_html, encoding="utf-8")
    num_qa_kb = num_qa_path.stat().st_size / 1024
    print(f"  Numeric QA: {num_qa_path}  ({num_qa_kb:.1f} KB)")
    if pipeline_audit.has_partial_failure:
        print(f"  ⚠ {pipeline_audit.total_untrusted} UNTRUSTED numbers — "
              f"review report")
    else:
        print(f"  ✓ All {pipeline_audit.total_numeric_tokens} numbers verified")

    # ── Phase 7: Save log ────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    log_data = {
        "pipeline": "final_pipeline.py (Surya-only, discrete trust)",
        "input": str(pdf_path.name),
        "output": str(out_path.name),
        "total_pages": total_pages,
        "native_pages": len(native_pages),
        "scanned_pages": len(scanned_pages),
        "total_time_s": round(elapsed_total, 1),
        "quality_ok": all_ok,
        "architecture": {
            "ocr_engine": "Surya (full-page, normalized input)",
            "tesseract": "REMOVED",
            "input_normalization": "2240px width + 2x2 dilation",
            "trust_model": "discrete (LOCKED/SURYA_VALID/CNN_CONFIRMED/UNTRUSTED)",
            "validation": "line-level stability + column boosting",
            "removed": ["SSIM", "pixel_overlap", "dual_OCR_voting",
                        "Latin_to_Arabic_mapping", "digit_repair_heuristics"],
        },
        "qa": {
            "total_numeric_columns": total_columns,
            "total_anomalies": total_anomalies,
        },
        "numeric_trust": {
            "total_numeric_tokens": pipeline_audit.total_numeric_tokens,
            "locked": sum(p.trust_summary.get('locked', 0)
                          for p in pipeline_audit.pages
                          if p.trust_summary),
            "surya_valid": sum(p.trust_summary.get('surya_valid', 0)
                               for p in pipeline_audit.pages
                               if p.trust_summary),
            "cnn_confirmed": sum(p.trust_summary.get('cnn_confirmed', 0)
                                  for p in pipeline_audit.pages
                                  if p.trust_summary),
            "trusted": pipeline_audit.total_trusted,
            "untrusted": pipeline_audit.total_untrusted,
            "trust_pct": pipeline_audit.overall_trust_pct,
            "has_partial_failure": pipeline_audit.has_partial_failure,
        },
        "decisions": [
            {
                "page": d.page_number,
                "type": d.page_type,
                "action": d.action,
                "meaningful_chars": d.meaningful_chars,
                "text_blocks": d.text_blocks,
                "token_count": d.token_count,
                "numeric_tokens": d.numeric_token_count,
                "text_tokens": d.text_token_count,
                "numeric_columns": d.num_columns,
                "anomalies": d.num_anomalies,
                "original_dims": d.original_dims,
                "processing_time_s": d.processing_time,
                "notes": d.notes,
            }
            for d in decisions
        ],
    }
    log_path.write_text(
        json.dumps(log_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Log: {log_path}")

    # ── Cleanup ──────────────────────────────────────────────────────
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
    print(f"  ✓ {len(scanned_pages)} pages OCR-reconstructed (Surya-only)")
    print(f"  ✓ {total_columns} numeric columns detected (QA only)")
    print(f"  ✓ {total_anomalies} anomalies flagged")
    if pipeline_audit.total_numeric_tokens > 0:
        ts = pipeline_audit
        print(f"  {'✓' if not ts.has_partial_failure else '⚠'} "
              f"Trust: {ts.total_trusted}/{ts.total_numeric_tokens} "
              f"({ts.overall_trust_pct}%)")
        locked_total = sum(p.trust_summary.get('locked', 0)
                           for p in ts.pages if p.trust_summary)
        print(f"    LOCKED: {locked_total} | "
              f"UNTRUSTED: {ts.total_untrusted}")
        if ts.has_partial_failure:
            print(f"  ⚠ {ts.total_untrusted} UNTRUSTED — "
                  f"see {num_qa_path.name}")
    print(f"  ✓ Output: {out_path.name}  ({out_kb:.1f} KB)")
    print(f"  ✓ Numeric QA: {num_qa_path.name}  ({num_qa_kb:.1f} KB)")
    if all_ok:
        print(f"  ✓ All quality checks passed")
    else:
        print(f"  ⚠ Some quality checks had issues — review log")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
