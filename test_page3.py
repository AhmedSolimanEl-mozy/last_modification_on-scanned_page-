#!/usr/bin/env python3
"""
test_page3.py — Single-page test harness for page 3 only.

Runs the full pipeline (OCR → validate → render → PDF) on page 3 alone.
Output files all prefixed with "p3_test_" for quick iteration.

Usage:
    python3 test_page3.py
"""

from __future__ import annotations

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

import fitz

from token_extract import (
    Token, PageTokens,
    extract_tokens_scanned,
    classify_token, detect_direction,
    render_page_at_dpi,
)
from numeric_columns import (
    NumericColumn, ColumnAnomaly,
    find_numeric_columns, detect_column_anomalies,
    columns_summary, anomalies_summary, render_qa_html,
    compute_column_boost,
)
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
BASE_DIR       = Path(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH       = BASE_DIR / "el-bankalahly .pdf"
PAGE_INDEX     = 2               # 0-based → page 3
PAGE_NUM       = PAGE_INDEX + 1  # 1-based → 3

OUT_PDF        = BASE_DIR / "p3_test_output.pdf"
OUT_QA_HTML    = BASE_DIR / "p3_test_qa.html"
OUT_NUM_QA     = BASE_DIR / "p3_test_numeric_qa.html"
OUT_LOG        = BASE_DIR / "p3_test_log.json"
OUT_TOKENS     = BASE_DIR / "p3_test_tokens.jsonl"

NUMERIC_VAL_DPI = 600

# Rendering constants (same as final_pipeline)
FONT_SIZE_FACTOR  = 0.85
FONT_DRIFT_FACTOR = 1.25
FONT_FAMILY       = '"Arial", "Helvetica", "Noto Sans Arabic", sans-serif'
MIN_LINE_GAP      = 1.0


# ────────────────────────────────────────────────────────────────────
#  Layout normalization (copied from final_pipeline)
# ────────────────────────────────────────────────────────────────────
def normalize_page_tokens(page_tokens: PageTokens) -> PageTokens:
    page = copy.deepcopy(page_tokens)
    pw, ph = page.page_width, page.page_height
    if not page.lines:
        return page
    page.lines.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    for ln in page.lines:
        x0, y0, x1, y1 = ln["bbox"]
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(pw, x1); y1 = min(ph, y1)
        ln["bbox"] = [round(x0,2), round(y0,2), round(x1,2), round(y1,2)]
    for i in range(1, len(page.lines)):
        ln = page.lines[i]
        x0, y0, x1, y1 = ln["bbox"]
        h = y1 - y0
        for j in range(i-1, max(i-20, -1), -1):
            prev = page.lines[j]
            px0, py0, px1, py1 = prev["bbox"]
            if x0 >= px1 or x1 <= px0:
                continue
            if y0 >= py1:
                continue
            ov_x = min(x1,px1) - max(x0,px0)
            ov_y = py1 - y0
            if ov_x > 2 and ov_y > 2:
                ln["bbox"][1] = round(py1 + MIN_LINE_GAP, 2)
                ln["bbox"][3] = round(ln["bbox"][1] + h, 2)
                y0, y1 = ln["bbox"][1], ln["bbox"][3]
    for ln in page.lines:
        for tok in ln.get("tokens", []):
            tok.bbox[1] = ln["bbox"][1]
            tok.bbox[3] = ln["bbox"][3]
    return page


# ────────────────────────────────────────────────────────────────────
#  HTML Rendering (copied from final_pipeline)
# ────────────────────────────────────────────────────────────────────
def _render_line_html(line: dict) -> str:
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
    pw, ph = page_tokens.page_width, page_tokens.page_height
    lines = sorted(page_tokens.lines,
                   key=lambda ln: (ln["bbox"][1], -ln["bbox"][0]))
    lines_html = "\n    ".join(_render_line_html(ln) for ln in lines)
    return f"""<!DOCTYPE html>
<html lang="ar" dir="ltr">
<head>
<meta charset="utf-8">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: {FONT_FAMILY};
  direction: ltr;
  margin: 0; padding: 0;
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


async def html_to_pdf_page(html_str, width_pt, height_pt, output_path):
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
            margin={"top":"0","right":"0","bottom":"0","left":"0"},
            print_background=True,
            scale=1.0,
        )
        await browser.close()


# ────────────────────────────────────────────────────────────────────
#  Main — Page 3 Only
# ────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    print("=" * 64)
    print(f"  PAGE 3 TEST — Single Page Pipeline")
    print(f"  Input:    {PDF_PATH.name}  (page {PAGE_NUM} only)")
    print(f"  Output:   {OUT_PDF.name}")
    print("=" * 64)

    doc = fitz.open(str(PDF_PATH))
    page = doc[PAGE_INDEX]
    pw, ph = round(page.rect.width, 2), round(page.rect.height, 2)
    print(f"\n  Page {PAGE_NUM}: {pw} x {ph} pt\n")

    # ── Step 1: OCR Token Extraction ─────────────────────────────────
    print("─── Step 1: Surya OCR (normalized input) ───")
    t0 = time.time()
    page_tokens = extract_tokens_scanned(doc, PAGE_INDEX)
    t_ocr = time.time() - t0

    n_tokens = len(page_tokens.tokens)
    n_numeric = sum(1 for t in page_tokens.tokens if t.token_type == "NUMERIC")
    n_text = n_tokens - n_numeric
    n_lines = len(page_tokens.lines)
    print(f"  {n_tokens} tokens ({n_text} text, {n_numeric} numeric), "
          f"{n_lines} lines  ({t_ocr:.1f}s)")

    # Dump raw tokens for inspection
    with open(OUT_TOKENS, "w", encoding="utf-8") as f:
        for tok in page_tokens.tokens:
            f.write(json.dumps({
                "text": tok.text,
                "type": tok.token_type,
                "bbox": [round(b,2) for b in tok.bbox],
                "conf": round(tok.confidence, 4) if tok.confidence else None,
                "font_size": round(tok.font_size, 2),
            }, ensure_ascii=False) + "\n")
    print(f"  Raw tokens → {OUT_TOKENS.name}")

    # ── Step 2: Normalize Layout ─────────────────────────────────────
    print("\n─── Step 2: Layout Normalization ───")
    page_tokens = normalize_page_tokens(page_tokens)

    # Save original line texts for stability check
    original_lines = [
        {
            'line_id': ln.get('line_id', i),
            'text': ln.get('text', ''),
            'bbox': ln.get('bbox', [0,0,0,0]),
        }
        for i, ln in enumerate(page_tokens.lines)
    ]
    print(f"  {len(page_tokens.lines)} lines normalized")

    # ── Step 3: Numeric Validation ───────────────────────────────────
    print("\n─── Step 3: Numeric Validation (Discrete Trust) ───")
    hires = render_page_at_dpi(doc, PAGE_INDEX, dpi=NUMERIC_VAL_DPI)

    # A+B: Classify + Lock + CNN
    ocr_results = ocr_page_numeric_tokens(page_tokens, hires)
    n_locked = sum(1 for r in ocr_results if r.locked)
    print(f"  {len(ocr_results)} numeric tokens classified, {n_locked} LOCKED")

    # C: Line-level stability
    current_lines = [
        {
            'line_id': ln.get('line_id', i),
            'text': ln.get('text', ''),
            'bbox': ln.get('bbox', [0,0,0,0]),
        }
        for i, ln in enumerate(page_tokens.lines)
    ]
    page_stability = validate_page_lines(original_lines, current_lines, PAGE_NUM)
    ocr_results = apply_line_stability_to_tokens(
        page_stability, ocr_results, page_tokens.lines)
    print(f"  Line stability: {page_stability.stable_lines} stable, "
          f"{page_stability.unstable_lines} unstable")

    # D: Reconstruct numbers
    font_sizes = [t.font_size for t in page_tokens.tokens
                  if t.token_type == "NUMERIC"]
    numeric_values = reconstruct_page_numbers(ocr_results, PAGE_NUM, font_sizes)

    # Update token texts
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

    # Build audit
    pa = PageNumericAudit(
        page_number=PAGE_NUM,
        numeric_values=numeric_values,
        ocr_results=ocr_results,
        line_stability=page_stability,
    )
    pa.compute_summary()
    ts = pa.trust_summary
    print(f"\n  Trust Summary:")
    print(f"    🔒 LOCKED:       {ts.get('locked',0)}")
    print(f"    ✓ SURYA_VALID:   {ts.get('surya_valid',0)}")
    print(f"    🧠 CNN_CONFIRMED: {ts.get('cnn_confirmed',0)}")
    print(f"    ⚠ UNTRUSTED:     {ts.get('untrusted',0)}")
    print(f"    Trust rate:      {ts.get('pct_trusted',0)}%")

    # Column boosting
    print("\n  Column boosting ... ", end="", flush=True)
    columns = find_numeric_columns(page_tokens)
    token_trust_map = {}
    numeric_idx = 0
    for tok in page_tokens.tokens:
        if tok.token_type == "NUMERIC" and numeric_idx < len(numeric_values):
            nv = numeric_values[numeric_idx]
            key = (round(tok.bbox[0],2), round(tok.bbox[1],2),
                   round(tok.bbox[2],2), round(tok.bbox[3],2))
            token_trust_map[key] = nv.status.value
            numeric_idx += 1
    boost_results = compute_column_boost(columns, token_trust_map)
    boost_count = 0
    for br in boost_results:
        if br.eligible:
            for tok in br.boosted_tokens:
                tok_key = (round(tok.bbox[0],2), round(tok.bbox[1],2),
                           round(tok.bbox[2],2), round(tok.bbox[3],2))
                for nv_idx, nv in enumerate(numeric_values):
                    nv_key = (round(nv.bbox[0],2), round(nv.bbox[1],2),
                              round(nv.bbox[2],2), round(nv.bbox[3],2))
                    if nv_key == tok_key and nv.status == TrustStatus.UNTRUSTED:
                        ocr_r = ocr_results[nv_idx]
                        updated = validate_token_with_cnn(
                            ocr_r, hires, page_tokens.page_width,
                            page_tokens.page_height)
                        ocr_results[nv_idx] = updated
                        nv.revalidated = True
                        if updated.cnn_confirmed:
                            nv.status = TrustStatus.CNN_CONFIRMED
                            nv.trust_score = TRUST_SCORES[TrustStatus.CNN_CONFIRMED]
                            nv.cnn_confirmed = True
                            boost_count += 1
                        break
    pa.compute_summary()
    print(f"{boost_count} boosted")

    pipeline_audit = PipelineNumericAudit(pages=[pa])
    pipeline_audit.compute_overall()

    # ── Step 4: Render HTML → PDF ────────────────────────────────────
    print(f"\n─── Step 4: Render Page → PDF ───")
    t0 = time.time()
    page_html = render_page_html(page_tokens)
    print(f"  HTML rendered ({len(page_html)} bytes)")

    tmp_pdf = Path(tempfile.mktemp(suffix=".pdf"))
    asyncio.run(html_to_pdf_page(page_html, pw, ph, tmp_pdf))
    t_render = time.time() - t0
    print(f"  Playwright PDF: {tmp_pdf}  ({t_render:.1f}s)")

    # Copy to output as single-page PDF
    merged = fitz.open(str(tmp_pdf))
    merged.save(str(OUT_PDF))
    merged.close()
    tmp_pdf.unlink(missing_ok=True)
    out_kb = OUT_PDF.stat().st_size / 1024
    print(f"  Output: {OUT_PDF.name}  ({out_kb:.1f} KB)")

    # ── Step 5: QA Reports ───────────────────────────────────────────
    print(f"\n─── Step 5: QA Reports ───")

    # Numeric QA
    num_qa_html = generate_qa_report(pipeline_audit)
    OUT_NUM_QA.write_text(num_qa_html, encoding="utf-8")
    print(f"  Numeric QA: {OUT_NUM_QA.name}  "
          f"({OUT_NUM_QA.stat().st_size/1024:.1f} KB)")

    # Column QA
    all_columns = [columns]
    all_anomalies = [detect_column_anomalies(columns)]
    qa_html = render_qa_html([page_tokens], all_columns, all_anomalies)
    OUT_QA_HTML.write_text(qa_html, encoding="utf-8")
    print(f"  Column QA:  {OUT_QA_HTML.name}  "
          f"({OUT_QA_HTML.stat().st_size/1024:.1f} KB)")

    # ── Step 6: Print All Numeric Tokens ─────────────────────────────
    print(f"\n─── Numeric Token Detail ───")
    print(f"  {'#':>3}  {'Status':<15} {'Conf':>6} {'Trust':>5}  {'Digits'}")
    print(f"  {'─'*3}  {'─'*15} {'─'*6} {'─'*5}  {'─'*20}")
    for idx, nv in enumerate(numeric_values):
        icon = {
            TrustStatus.LOCKED: '🔒',
            TrustStatus.SURYA_VALID: '✓ ',
            TrustStatus.CNN_CONFIRMED: '🧠',
            TrustStatus.UNTRUSTED: '⚠ ',
        }.get(nv.status, '? ')
        reasons = ', '.join(r.value for r in nv.failure_reasons) if nv.failure_reasons else ''
        extra = f"  ({reasons})" if reasons else ""
        print(f"  {idx+1:3d}  {icon} {nv.status.value:<13} "
              f"{nv.surya_confidence:6.3f} {nv.trust_score:5.2f}  "
              f"{nv.digits}{extra}")

    # ── Step 7: Log ──────────────────────────────────────────────────
    elapsed = time.time() - t_start
    log_data = {
        "test": "page3_only",
        "page": PAGE_NUM,
        "time_s": round(elapsed, 1),
        "ocr_time_s": round(t_ocr, 1),
        "render_time_s": round(t_render, 1),
        "tokens": n_tokens,
        "numeric_tokens": n_numeric,
        "text_tokens": n_text,
        "lines": n_lines,
        "trust": {
            "total": len(numeric_values),
            "locked": ts.get('locked', 0),
            "surya_valid": ts.get('surya_valid', 0),
            "cnn_confirmed": ts.get('cnn_confirmed', 0),
            "untrusted": ts.get('untrusted', 0),
            "trust_pct": ts.get('pct_trusted', 0),
        },
        "column_boost": boost_count,
        "columns": len(columns),
        "anomalies": len(all_anomalies[0]),
        "output_files": {
            "pdf": OUT_PDF.name,
            "numeric_qa": OUT_NUM_QA.name,
            "column_qa": OUT_QA_HTML.name,
            "tokens": OUT_TOKENS.name,
        },
    }
    OUT_LOG.write_text(
        json.dumps(log_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    doc.close()

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  ✓ Page 3 done in {elapsed:.1f}s")
    print(f"  ✓ {n_tokens} tokens ({n_numeric} numeric)")
    ts2 = pipeline_audit
    print(f"  {'✓' if not ts2.has_partial_failure else '⚠'} "
          f"Trust: {ts2.total_trusted}/{ts2.total_numeric_tokens} "
          f"({ts2.overall_trust_pct}%)")
    print(f"    🔒 LOCKED: {ts.get('locked',0)} | "
          f"⚠ UNTRUSTED: {ts.get('untrusted',0)}")
    print(f"  ✓ PDF:    {OUT_PDF.name}  ({out_kb:.1f} KB)")
    print(f"  ✓ QA:     {OUT_NUM_QA.name}")
    print(f"  ✓ Tokens: {OUT_TOKENS.name}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
