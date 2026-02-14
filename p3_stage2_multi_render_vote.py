#!/usr/bin/env python3
"""
p3_stage2_multi_render_vote.py — Stage 2: Multi-Render OCR Voting
=================================================================

Strategy:
  1. Render page at 3 different preprocessing variants:
     a. CLAHE + standard dilation (2×2, 1 iter) — same as Stage 1
     b. CLAHE + heavier dilation (3×3, 1 iter) — bolder strokes
     c. CLAHE + no dilation (binarization only) — thin strokes
  2. Run Surya OCR on each variant independently.
  3. Match numeric tokens across variants by bbox overlap.
  4. For each token: if ≥2 variants agree on digit text → promote trust.
  5. UNTRUSTED tokens confirmed by multi-vote → SURYA_VALID.

Output files: p3_stage2_*

Usage:
    python3 p3_stage2_multi_render_vote.py
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
from typing import List, Dict, Tuple, Optional
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)

import cv2
import numpy as np
import fitz

from token_extract import (
    Token, PageTokens,
    classify_token, detect_direction,
    render_page_at_dpi,
    NORMALIZED_WIDTH, OCR_DPI, DILATION_KERNEL_SIZE,
    clean_surya_artifacts,
)
from numeric_columns import (
    find_numeric_columns, detect_column_anomalies,
    render_qa_html, compute_column_boost,
)
from digit_ocr import (
    TrustStatus, FailureReason, TokenOCRResult,
    ocr_page_numeric_tokens,
    validate_token_with_cnn,
    TRUST_SCORES, has_valid_arabic_digits,
    extract_digits, CONFIDENCE_THRESHOLD,
    classify_and_lock_token,
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

PREFIX = "p3_stage2"
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

# Vote threshold: at least 2 out of 3 must agree
VOTE_QUORUM = 2


# ────────────────────────────────────────────────────────────────────
#  Multi-Render Preprocessing Variants
# ────────────────────────────────────────────────────────────────────
def _preprocess_variant_a(img: np.ndarray, target_width: int) -> np.ndarray:
    """Variant A: CLAHE + standard 2×2 dilation (same as Stage 1)."""
    h, w = img.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    result = cv2.bitwise_not(dilated)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def _preprocess_variant_b(img: np.ndarray, target_width: int) -> np.ndarray:
    """Variant B: CLAHE + heavier 3×3 dilation (bolder strokes)."""
    h, w = img.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    result = cv2.bitwise_not(dilated)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def _preprocess_variant_c(img: np.ndarray, target_width: int) -> np.ndarray:
    """Variant C: CLAHE + no dilation (thin strokes, binarization only)."""
    h, w = img.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    result = cv2.bitwise_not(binary)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


# ────────────────────────────────────────────────────────────────────
#  Run Surya OCR on a single preprocessed image
# ────────────────────────────────────────────────────────────────────
def _run_surya_on_image(
    preprocessed_bgr: np.ndarray,
    doc: fitz.Document,
    page_idx: int,
    dpi: int = OCR_DPI,
) -> PageTokens:
    """Run Surya OCR on a preprocessed BGR image, return PageTokens."""
    import token_extract as te
    from PIL import Image

    te._load_surya()

    page = doc[page_idx]
    pw = round(page.rect.width, 2)
    ph = round(page.rect.height, 2)
    page_number = page_idx + 1
    px_to_pt = 72.0 / dpi

    img_rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    results = te._surya_rec(
        [pil_img],
        task_names=["ocr_with_boxes"],
        det_predictor=te._surya_det,
        return_words=True,
        sort_lines=True,
    )

    if not results or not results[0].text_lines:
        return PageTokens(
            page_number=page_number, page_width=pw, page_height=ph,
            tokens=[], extraction_method="surya_ocr", lines=[],
        )

    tokens = []
    lines_data = []

    for line_idx, tl in enumerate(results[0].text_lines):
        line_poly = tl.polygon
        lxs = [p[0] for p in line_poly]
        lys = [p[1] for p in line_poly]
        line_bbox_pt = [
            round(min(lxs) * px_to_pt, 2),
            round(min(lys) * px_to_pt, 2),
            round(max(lxs) * px_to_pt, 2),
            round(max(lys) * px_to_pt, 2),
        ]
        line_text = tl.text
        line_conf = tl.confidence or 0.0
        direction = detect_direction(line_text)
        line_tokens = []

        if tl.chars:
            current_word_chars = []
            current_word_bboxes = []

            def flush_word():
                if not current_word_chars:
                    return
                word_text = "".join(current_word_chars)
                x0 = min(b[0] for b in current_word_bboxes)
                y0 = min(b[1] for b in current_word_bboxes)
                x1 = max(b[2] for b in current_word_bboxes)
                y1 = max(b[3] for b in current_word_bboxes)
                fs = max((y1 - y0) * 0.85, 4.0)
                tok = Token(
                    text=word_text,
                    bbox=[round(x0, 2), round(y0, 2),
                          round(x1, 2), round(y1, 2)],
                    font_size=round(fs, 2),
                    confidence=round(line_conf, 3),
                    token_type=classify_token(word_text),
                    direction=detect_direction(word_text),
                    line_id=line_idx,
                )
                tokens.append(tok)
                line_tokens.append(tok)
                current_word_chars.clear()
                current_word_bboxes.clear()

            for ch in tl.chars:
                ch_poly = ch.polygon
                cxs = [p[0] for p in ch_poly]
                cys = [p[1] for p in ch_poly]
                ch_bbox_pt = [
                    round(min(cxs) * px_to_pt, 2),
                    round(min(cys) * px_to_pt, 2),
                    round(max(cxs) * px_to_pt, 2),
                    round(max(cys) * px_to_pt, 2),
                ]
                c = ch.text
                if c.isspace():
                    flush_word()
                else:
                    current_word_chars.append(c)
                    current_word_bboxes.append(ch_bbox_pt)

            flush_word()
        else:
            fs = max((line_bbox_pt[3] - line_bbox_pt[1]) * 0.85, 4.0)
            tok = Token(
                text=line_text, bbox=line_bbox_pt,
                font_size=round(fs, 2), confidence=round(line_conf, 3),
                token_type=classify_token(line_text),
                direction=direction, line_id=line_idx,
            )
            tokens.append(tok)
            line_tokens.append(tok)

        lines_data.append({
            "line_id": line_idx, "bbox": line_bbox_pt,
            "text": line_text, "direction": direction,
            "font_size": line_tokens[0].font_size if line_tokens else 8.0,
            "is_bold": False, "is_italic": False,
            "tokens": line_tokens,
        })

    page_tokens = PageTokens(
        page_number=page_number, page_width=pw, page_height=ph,
        tokens=tokens, extraction_method="surya_ocr", lines=lines_data,
    )
    page_tokens = clean_surya_artifacts(page_tokens)
    return page_tokens


# ────────────────────────────────────────────────────────────────────
#  Token Matching + Majority Voting
# ────────────────────────────────────────────────────────────────────
def _bbox_iou(a: List[float], b: List[float]) -> float:
    """Compute IoU between two bboxes [x0, y0, x1, y1]."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def multi_render_vote(
    primary_tokens: PageTokens,
    variant_results: List[PageTokens],
    vote_quorum: int = VOTE_QUORUM,
) -> Tuple[PageTokens, Dict]:
    """Match numeric tokens across variants and majority-vote.

    For each numeric token in primary_tokens:
      1. Find matching tokens in each variant (by bbox IoU > 0.3).
      2. Extract digits from each match.
      3. If ≥ vote_quorum variants agree on digit text, boost confidence.

    Returns:
        Updated primary_tokens and vote_stats dict.
    """
    vote_stats = {
        "total_numeric": 0,
        "voted": 0,
        "promoted": 0,
        "unchanged": 0,
        "details": [],
    }

    for tok in primary_tokens.tokens:
        if tok.token_type != "NUMERIC":
            continue

        vote_stats["total_numeric"] += 1
        primary_digits = extract_digits(tok.text)
        if not primary_digits:
            continue

        # Collect digit readings from all variants
        digit_readings = [primary_digits]  # primary is vote #1
        conf_readings = [tok.confidence]

        for vr in variant_results:
            best_iou = 0.0
            best_tok = None
            for vtok in vr.tokens:
                if vtok.token_type != "NUMERIC":
                    continue
                iou = _bbox_iou(tok.bbox, vtok.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tok = vtok

            if best_tok and best_iou > 0.3:
                vdigits = extract_digits(best_tok.text)
                if vdigits:
                    digit_readings.append(vdigits)
                    conf_readings.append(best_tok.confidence)

        # Majority vote
        if len(digit_readings) >= 2:
            counter = Counter(digit_readings)
            most_common, count = counter.most_common(1)[0]

            detail = {
                "text": tok.text,
                "digits": primary_digits,
                "readings": digit_readings,
                "winner": most_common,
                "votes": count,
                "bbox": [round(b, 2) for b in tok.bbox],
                "conf": tok.confidence,
            }

            if count >= vote_quorum:
                vote_stats["voted"] += 1

                # If primary was low-conf but vote confirms, boost confidence
                if tok.confidence < CONFIDENCE_THRESHOLD and most_common == primary_digits:
                    # Promote: set confidence to the max across agreeing variants
                    max_conf = max(
                        c for d, c in zip(digit_readings, conf_readings)
                        if d == most_common
                    )
                    new_conf = max(max_conf, CONFIDENCE_THRESHOLD + 0.01)
                    detail["action"] = f"PROMOTED conf {tok.confidence:.3f} → {new_conf:.3f}"
                    tok.confidence = new_conf
                    vote_stats["promoted"] += 1
                elif count >= vote_quorum and most_common != primary_digits:
                    # Majority disagrees with primary — adopt majority text
                    detail["action"] = f"CORRECTED {primary_digits} → {most_common}"
                    # Rebuild token text with majority digits
                    tok.text = most_common
                    tok.token_type = classify_token(tok.text)
                    # Set confidence to max of agreeing voters
                    max_conf = max(
                        c for d, c in zip(digit_readings, conf_readings)
                        if d == most_common
                    )
                    tok.confidence = max(max_conf, CONFIDENCE_THRESHOLD + 0.01)
                    vote_stats["promoted"] += 1
                else:
                    detail["action"] = "CONFIRMED (already trusted)"
                    vote_stats["unchanged"] += 1

                vote_stats["details"].append(detail)
            else:
                detail["action"] = "NO_QUORUM"
                vote_stats["details"].append(detail)

    # Rebuild line texts after vote corrections
    for ln in primary_tokens.lines:
        line_toks = ln.get('tokens', [])
        if line_toks:
            ln['text'] = ' '.join(t.text for t in line_toks)

    return primary_tokens, vote_stats


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
#  Main — Stage 2 Test
# ────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    print("=" * 64)
    print(f"  STAGE 2: Multi-Render OCR Voting")
    print(f"  Input:    {PDF_PATH.name}  (page {PAGE_NUM} only)")
    print(f"  Output:   {PREFIX}_*")
    print(f"  Variants: A (2×2 dil), B (3×3 dil), C (no dil)")
    print("=" * 64)

    doc = fitz.open(str(PDF_PATH))
    page = doc[PAGE_INDEX]
    pw, ph = round(page.rect.width, 2), round(page.rect.height, 2)
    print(f"\n  Page {PAGE_NUM}: {pw} x {ph} pt\n")

    # ── Step 1: Render base image ──
    print("─── Step 1: Render & Preprocess 3 Variants ───")
    t0 = time.time()
    base_img = render_page_at_dpi(doc, PAGE_INDEX, OCR_DPI)
    print(f"  Base image: {base_img.shape[1]}×{base_img.shape[0]} px")

    var_a = _preprocess_variant_a(base_img, NORMALIZED_WIDTH)
    hash_a = hashlib.sha256(var_a.tobytes()).hexdigest()[:12]
    print(f"  Variant A (CLAHE + 2×2 dil): hash={hash_a}")

    var_b = _preprocess_variant_b(base_img, NORMALIZED_WIDTH)
    hash_b = hashlib.sha256(var_b.tobytes()).hexdigest()[:12]
    print(f"  Variant B (CLAHE + 3×3 dil): hash={hash_b}")

    var_c = _preprocess_variant_c(base_img, NORMALIZED_WIDTH)
    hash_c = hashlib.sha256(var_c.tobytes()).hexdigest()[:12]
    print(f"  Variant C (CLAHE + no dil):   hash={hash_c}")

    t_preprocess = time.time() - t0
    print(f"  Preprocessing: {t_preprocess:.1f}s")

    # ── Step 2: Run Surya OCR on variant A (primary) ──
    print(f"\n─── Step 2: Surya OCR — Variant A (primary) ───")
    t0 = time.time()
    tokens_a = _run_surya_on_image(var_a, doc, PAGE_INDEX, OCR_DPI)
    t_a = time.time() - t0
    n_num_a = sum(1 for t in tokens_a.tokens if t.token_type == "NUMERIC")
    print(f"  {len(tokens_a.tokens)} tokens ({n_num_a} numeric), "
          f"{len(tokens_a.lines)} lines  ({t_a:.1f}s)")

    # ── Step 3: Run Surya OCR on variant B ──
    print(f"\n─── Step 3: Surya OCR — Variant B ───")
    t0 = time.time()
    tokens_b = _run_surya_on_image(var_b, doc, PAGE_INDEX, OCR_DPI)
    t_b = time.time() - t0
    n_num_b = sum(1 for t in tokens_b.tokens if t.token_type == "NUMERIC")
    print(f"  {len(tokens_b.tokens)} tokens ({n_num_b} numeric), "
          f"{len(tokens_b.lines)} lines  ({t_b:.1f}s)")

    # ── Step 4: Run Surya OCR on variant C ──
    print(f"\n─── Step 4: Surya OCR — Variant C ───")
    t0 = time.time()
    tokens_c = _run_surya_on_image(var_c, doc, PAGE_INDEX, OCR_DPI)
    t_c = time.time() - t0
    n_num_c = sum(1 for t in tokens_c.tokens if t.token_type == "NUMERIC")
    print(f"  {len(tokens_c.tokens)} tokens ({n_num_c} numeric), "
          f"{len(tokens_c.lines)} lines  ({t_c:.1f}s)")

    # ── Step 5: Multi-Render Voting ──
    print(f"\n─── Step 5: Multi-Render Majority Vote ───")
    primary = tokens_a  # Use Variant A as primary
    voted_tokens, vote_stats = multi_render_vote(
        primary, [tokens_b, tokens_c], VOTE_QUORUM)

    print(f"  Numeric tokens: {vote_stats['total_numeric']}")
    print(f"  Voted (quorum met): {vote_stats['voted']}")
    print(f"  Promoted/Corrected: {vote_stats['promoted']}")
    print(f"  Unchanged: {vote_stats['unchanged']}")

    # Show vote details for interesting cases
    for d in vote_stats.get('details', []):
        if d.get('action', '').startswith('PROMOTED') or d.get('action', '').startswith('CORRECTED'):
            print(f"    → {d['action']}  digits={d['digits']}  "
                  f"readings={d['readings']}  conf={d['conf']:.3f}")

    page_tokens = voted_tokens

    # Dump raw tokens
    n_tokens = len(page_tokens.tokens)
    n_numeric = sum(1 for t in page_tokens.tokens if t.token_type == "NUMERIC")
    n_text = n_tokens - n_numeric
    n_lines = len(page_tokens.lines)

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

    # ── Step 6: Normalize Layout ──
    print(f"\n─── Step 6: Layout Normalization ───")
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

    # ── Step 7: Numeric Validation ──
    print(f"\n─── Step 7: Numeric Validation (Discrete Trust) ───")
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

    # ── Step 8: Render HTML → PDF ──
    print(f"\n─── Step 8: Render Page → PDF ───")
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

    # ── Step 9: QA Reports ──
    print(f"\n─── Step 9: QA Reports ───")
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

    # ── Step 10: Numeric Token Detail ──
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

    # ── Step 11: Log ──
    elapsed = time.time() - t_start
    log_data = {
        "stage": "stage2_multi_render_vote",
        "description": "3 preprocessing variants + majority digit voting",
        "page": PAGE_NUM,
        "time_s": round(elapsed, 1),
        "ocr_times": {
            "variant_a": round(t_a, 1),
            "variant_b": round(t_b, 1),
            "variant_c": round(t_c, 1),
        },
        "render_time_s": round(t_render, 1),
        "tokens": n_tokens,
        "numeric_tokens": n_numeric,
        "text_tokens": n_text,
        "lines": n_lines,
        "image_hashes": {"a": hash_a, "b": hash_b, "c": hash_c},
        "vote_stats": {
            "total_numeric": vote_stats["total_numeric"],
            "voted": vote_stats["voted"],
            "promoted": vote_stats["promoted"],
            "unchanged": vote_stats["unchanged"],
        },
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
        "baseline_comparison": {
            "baseline_trust_pct": 94.2,
            "stage1_trust_pct": 95.6,
            "baseline_untrusted": 8,
            "stage1_untrusted": 6,
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
    trust_pct = ts.get('pct_trusted', 0)
    delta_base = trust_pct - 94.2
    delta_s1 = trust_pct - 95.6
    print(f"\n{'='*64}")
    print(f"  ✓ Stage 2 done in {elapsed:.1f}s")
    print(f"  ✓ {n_tokens} tokens ({n_numeric} numeric)")
    print(f"  Trust: {trust_pct}%  "
          f"(Δ {'+' if delta_base>=0 else ''}{delta_base:.1f}pp from baseline, "
          f"Δ {'+' if delta_s1>=0 else ''}{delta_s1:.1f}pp from Stage 1)")
    print(f"    🔒 LOCKED: {ts.get('locked',0)} | "
          f"⚠ UNTRUSTED: {ts.get('untrusted',0)}")
    n_anom = len(all_anomalies[0])
    print(f"  Columns: {len(columns)} | Anomalies: {n_anom}")
    print(f"  Vote: {vote_stats['promoted']} promoted, "
          f"{vote_stats['voted']} voted, "
          f"{vote_stats['unchanged']} confirmed")
    print(f"  Output: {PREFIX}_*")
    print(f"{'='*64}")

    meets_target = trust_pct > 98 and n_anom < (len(numeric_values) * 0.01)
    if meets_target:
        print(f"\n  🎯 TARGET MET! trust > 98% AND anomalies < 1%")
    else:
        print(f"\n  ⚠ Target NOT met. Proceeding to Stage 3...")


if __name__ == "__main__":
    main()
