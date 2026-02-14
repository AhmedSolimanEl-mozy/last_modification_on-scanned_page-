#!/usr/bin/env python3
"""
p3_stage1_clahe_determinism.py — Stage 1: Deterministic OCR Input Enhancement
==============================================================================

Enhancements over baseline:
  1. CLAHE (Contrast-Limited Adaptive Histogram Equalization) applied before
     binarization — improves digit-background separation for low-contrast regions.
  2. Image hash logging — proves determinism (same hash across runs).
  3. Per-page confidence stats in the log.

Output files: p3_stage1_*

Usage:
    python3 p3_stage1_clahe_determinism.py
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import html as html_mod
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import cv2
import numpy as np
import fitz

from token_extract import (
    Token, PageTokens,
    extract_tokens_scanned,
    classify_token, detect_direction,
    render_page_at_dpi,
    normalize_page_image as _original_normalize,
    NORMALIZED_WIDTH, OCR_DPI, DILATION_KERNEL_SIZE,
)
from numeric_columns import (
    find_numeric_columns, detect_column_anomalies,
    render_qa_html, compute_column_boost,
)
from digit_ocr import (
    TrustStatus, FailureReason,
    ocr_page_numeric_tokens,
    validate_token_with_cnn,
    TRUST_SCORES, has_valid_arabic_digits,
)
from numeric_validator import (
    validate_page_lines, apply_line_stability_to_tokens,
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
BASE_DIR   = Path(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH   = BASE_DIR / "el-bankalahly .pdf"
PAGE_INDEX = 2
PAGE_NUM   = PAGE_INDEX + 1

PREFIX = "p3_stage1"
OUT_PDF     = BASE_DIR / f"{PREFIX}_output.pdf"
OUT_QA_HTML = BASE_DIR / f"{PREFIX}_qa.html"
OUT_NUM_QA  = BASE_DIR / f"{PREFIX}_numeric_qa.html"
OUT_LOG     = BASE_DIR / f"{PREFIX}_log.json"
OUT_TOKENS  = BASE_DIR / f"{PREFIX}_tokens.jsonl"

NUMERIC_VAL_DPI = 600

# Rendering constants
FONT_SIZE_FACTOR  = 0.85
FONT_DRIFT_FACTOR = 1.25
FONT_FAMILY       = '"Arial", "Helvetica", "Noto Sans Arabic", sans-serif'
MIN_LINE_GAP      = 1.0


# ────────────────────────────────────────────────────────────────────
#  Stage 1: Enhanced Normalize with CLAHE
# ────────────────────────────────────────────────────────────────────
_image_hashes = {}  # Store per-stage image hash for determinism check


def normalize_page_image_clahe(
    img: np.ndarray,
    target_width: int = NORMALIZED_WIDTH,
    dpi: int = OCR_DPI,
) -> np.ndarray:
    """Enhanced normalization with CLAHE before binarization.

    Pipeline:
      1. Resize to fixed pixel width (deterministic, INTER_AREA).
      2. Convert to grayscale.
      3. CLAHE contrast enhancement (improves digit separation).
      4. Otsu binarization.
      5. Mild dilation (2×2 kernel, stroke consolidation).
      6. Convert back to BGR.
    """
    h, w = img.shape[:2]
    if w == 0:
        return img

    # Step 1: Resize to fixed width
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(img, (target_width, new_h),
                         interpolation=cv2.INTER_AREA)

    # Step 2: Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Step 3: CLAHE — adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 4: Otsu binarization (invert: strokes white)
    _, binary = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 5: Mild dilation (stroke consolidation)
    kernel_size = DILATION_KERNEL_SIZE  # 2
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Invert back to dark strokes on white
    consolidated = cv2.bitwise_not(dilated)

    # Step 6: BGR output
    result = cv2.cvtColor(consolidated, cv2.COLOR_GRAY2BGR)

    # Log hash for determinism verification
    img_hash = hashlib.sha256(result.tobytes()).hexdigest()[:16]
    _image_hashes['normalized'] = img_hash
    print(f"    Image hash (CLAHE-normalized): {img_hash}")

    return result


# ────────────────────────────────────────────────────────────────────
#  Monkey-patch token_extract to use CLAHE normalization
# ────────────────────────────────────────────────────────────────────
import token_extract
token_extract.normalize_page_image = normalize_page_image_clahe


# ────────────────────────────────────────────────────────────────────
#  Layout / Rendering (same as test_page3.py)
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
        f'width:{render_w:.2f}px; height:{height:.2f}px; '
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
#  Main — Stage 1 Test
# ────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    print("=" * 64)
    print(f"  STAGE 1: Deterministic OCR Input (CLAHE Enhancement)")
    print(f"  Input:    {PDF_PATH.name}  (page {PAGE_NUM} only)")
    print(f"  Output:   {PREFIX}_*")
    print("=" * 64)

    doc = fitz.open(str(PDF_PATH))
    page = doc[PAGE_INDEX]
    pw, ph = round(page.rect.width, 2), round(page.rect.height, 2)
    print(f"\n  Page {PAGE_NUM}: {pw} x {ph} pt\n")

    # ── Step 1: OCR Token Extraction (using CLAHE-enhanced normalization) ──
    print("─── Step 1: Surya OCR (CLAHE-normalized input) ───")
    t0 = time.time()
    page_tokens = extract_tokens_scanned(doc, PAGE_INDEX)
    t_ocr = time.time() - t0

    n_tokens = len(page_tokens.tokens)
    n_numeric = sum(1 for t in page_tokens.tokens if t.token_type == "NUMERIC")
    n_text = n_tokens - n_numeric
    n_lines = len(page_tokens.lines)
    print(f"  {n_tokens} tokens ({n_text} text, {n_numeric} numeric), "
          f"{n_lines} lines  ({t_ocr:.1f}s)")

    # Dump raw tokens
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

    # ── Step 2: Normalize Layout ──
    print("\n─── Step 2: Layout Normalization ───")
    page_tokens = normalize_page_tokens(page_tokens)
    original_lines = [
        {
            'line_id': ln.get('line_id', i),
            'text': ln.get('text', ''),
            'bbox': ln.get('bbox', [0,0,0,0]),
        }
        for i, ln in enumerate(page_tokens.lines)
    ]
    print(f"  {len(page_tokens.lines)} lines normalized")

    # ── Step 3: Numeric Validation ──
    print("\n─── Step 3: Numeric Validation (Discrete Trust) ───")
    hires = render_page_at_dpi(doc, PAGE_INDEX, dpi=NUMERIC_VAL_DPI)

    ocr_results = ocr_page_numeric_tokens(page_tokens, hires)
    n_locked = sum(1 for r in ocr_results if r.locked)
    print(f"  {len(ocr_results)} numeric tokens classified, {n_locked} LOCKED")

    # Line-level stability
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

    # Reconstruct numbers
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

    # ── Step 4: Render HTML → PDF ──
    print(f"\n─── Step 4: Render Page → PDF ───")
    t0 = time.time()
    page_html = render_page_html(page_tokens)
    print(f"  HTML rendered ({len(page_html)} bytes)")

    tmp_pdf = Path(tempfile.mktemp(suffix=".pdf"))
    asyncio.run(html_to_pdf_page(page_html, pw, ph, tmp_pdf))
    t_render = time.time() - t0
    print(f"  Playwright PDF: {tmp_pdf}  ({t_render:.1f}s)")

    merged = fitz.open(str(tmp_pdf))
    merged.save(str(OUT_PDF))
    merged.close()
    tmp_pdf.unlink(missing_ok=True)
    out_kb = OUT_PDF.stat().st_size / 1024
    print(f"  Output: {OUT_PDF.name}  ({out_kb:.1f} KB)")

    # ── Step 5: QA Reports ──
    print(f"\n─── Step 5: QA Reports ───")
    num_qa_html = generate_qa_report(pipeline_audit)
    OUT_NUM_QA.write_text(num_qa_html, encoding="utf-8")
    print(f"  Numeric QA: {OUT_NUM_QA.name}  "
          f"({OUT_NUM_QA.stat().st_size/1024:.1f} KB)")

    all_columns = [columns]
    all_anomalies = [detect_column_anomalies(columns)]
    qa_html = render_qa_html([page_tokens], all_columns, all_anomalies)
    OUT_QA_HTML.write_text(qa_html, encoding="utf-8")
    print(f"  Column QA:  {OUT_QA_HTML.name}  "
          f"({OUT_QA_HTML.stat().st_size/1024:.1f} KB)")

    # ── Step 6: Print All Numeric Tokens ──
    print(f"\n─── Numeric Token Detail ───")
    print(f"  {'#':>3}  {'Status':<15} {'Conf':>6} {'Trust':>5}  {'Digits'}")
    print(f"  {'─'*3}  {'─'*15} {'─'*6} {'─'*5}  {'─'*20}")
    untrusted_details = []
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
        if nv.status == TrustStatus.UNTRUSTED:
            untrusted_details.append({
                "idx": idx+1,
                "digits": nv.digits,
                "conf": nv.surya_confidence,
                "reasons": [r.value for r in nv.failure_reasons],
                "bbox": [round(b,2) for b in nv.bbox],
            })

    # ── Step 7: Log ──
    elapsed = time.time() - t_start
    log_data = {
        "stage": "stage1_clahe_determinism",
        "description": "CLAHE contrast enhancement before binarization",
        "page": PAGE_NUM,
        "time_s": round(elapsed, 1),
        "ocr_time_s": round(t_ocr, 1),
        "render_time_s": round(t_render, 1),
        "tokens": n_tokens,
        "numeric_tokens": n_numeric,
        "text_tokens": n_text,
        "lines": n_lines,
        "image_hashes": dict(_image_hashes),
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
        "untrusted_tokens": untrusted_details,
        "confidence_stats": {
            "min": round(min(nv.surya_confidence for nv in numeric_values), 3) if numeric_values else 0,
            "max": round(max(nv.surya_confidence for nv in numeric_values), 3) if numeric_values else 0,
            "mean": round(sum(nv.surya_confidence for nv in numeric_values) / len(numeric_values), 3) if numeric_values else 0,
        },
        "baseline_comparison": {
            "baseline_trust_pct": 94.2,
            "baseline_untrusted": 8,
        },
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

    # ── Summary ──
    delta_trust = ts.get('pct_trusted', 0) - 94.2
    delta_sign = "+" if delta_trust >= 0 else ""
    print(f"\n{'='*64}")
    print(f"  ✓ Stage 1 done in {elapsed:.1f}s")
    print(f"  ✓ {n_tokens} tokens ({n_numeric} numeric)")
    print(f"  Trust: {ts.get('pct_trusted',0)}%  "
          f"(Δ {delta_sign}{delta_trust:.1f}pp from baseline 94.2%)")
    print(f"    🔒 LOCKED: {ts.get('locked',0)} | "
          f"⚠ UNTRUSTED: {ts.get('untrusted',0)}")
    print(f"  Columns: {len(columns)} | Anomalies: {len(all_anomalies[0])}")
    print(f"  Image hash: {_image_hashes.get('normalized', 'N/A')}")
    print(f"  Output: {PREFIX}_*")
    print(f"{'='*64}")

    meets_target = ts.get('pct_trusted', 0) > 98 and len(all_anomalies[0]) < (len(numeric_values) * 0.01)
    if meets_target:
        print(f"\n  🎯 TARGET MET! trust > 98% AND anomalies < 1%")
    else:
        print(f"\n  ⚠ Target NOT met. Proceeding to Stage 2...")


if __name__ == "__main__":
    main()
