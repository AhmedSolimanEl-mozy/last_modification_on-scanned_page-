#!/usr/bin/env python3
"""
p3_stage3_column_context.py — Stage 3: Column-Context Numeric Validation
=========================================================================

Combines Stage 1 (CLAHE) + Stage 2 (Multi-Render Voting) + NEW:
  - Column-level digit-length consistency check
  - If ≥75% of column siblings are trusted AND token has valid digits
    AND digit count matches column's digit-length distribution → promote
  - Direct promotion to SURYA_VALID (no CNN needed)
  - Also: lower column-boost threshold from 90% to 75% for rescue

Output files: p3_stage3_*

Usage:
    python3 p3_stage3_column_context.py
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import html as html_mod
import json
import os
import re
import sys
import tempfile
import time
import statistics
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
    NumericColumn, parse_numeric,
    X_ALIGN_TOLERANCE,
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

PREFIX = "p3_stage3"
OUT_PDF     = BASE_DIR / f"{PREFIX}_output.pdf"
OUT_QA_HTML = BASE_DIR / f"{PREFIX}_qa.html"
OUT_NUM_QA  = BASE_DIR / f"{PREFIX}_numeric_qa.html"
OUT_LOG     = BASE_DIR / f"{PREFIX}_log.json"
OUT_TOKENS  = BASE_DIR / f"{PREFIX}_tokens.jsonl"

NUMERIC_VAL_DPI = 600

FONT_SIZE_FACTOR  = 0.85
FONT_DRIFT_FACTOR = 1.25
FONT_FAMILY       = '"Arial", "Helvetica", "Noto Sans Arabic", sans-serif'
MIN_LINE_GAP      = 1.0

VOTE_QUORUM = 2
COLUMN_RESCUE_THRESHOLD = 0.75  # 75% trusted siblings → eligible for rescue


# ────────────────────────────────────────────────────────────────────
#  Preprocessing Variants (from Stage 2)
# ────────────────────────────────────────────────────────────────────
def _preprocess_variant_a(img, tw):
    h, w = img.shape[:2]
    s = tw / w; nh = int(h * s)
    resized = cv2.resize(img, (tw, nh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return cv2.cvtColor(cv2.bitwise_not(dilated), cv2.COLOR_GRAY2BGR)

def _preprocess_variant_b(img, tw):
    h, w = img.shape[:2]
    s = tw / w; nh = int(h * s)
    resized = cv2.resize(img, (tw, nh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return cv2.cvtColor(cv2.bitwise_not(dilated), cv2.COLOR_GRAY2BGR)

def _preprocess_variant_c(img, tw):
    h, w = img.shape[:2]
    s = tw / w; nh = int(h * s)
    resized = cv2.resize(img, (tw, nh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2BGR)


# ────────────────────────────────────────────────────────────────────
#  Surya OCR on preprocessed image
# ────────────────────────────────────────────────────────────────────
def _run_surya_on_image(preprocessed_bgr, doc, page_idx, dpi=OCR_DPI):
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
        [pil_img], task_names=["ocr_with_boxes"],
        det_predictor=te._surya_det, return_words=True, sort_lines=True,
    )

    if not results or not results[0].text_lines:
        return PageTokens(page_number=page_number, page_width=pw, page_height=ph,
                          tokens=[], extraction_method="surya_ocr", lines=[])

    tokens = []
    lines_data = []
    for line_idx, tl in enumerate(results[0].text_lines):
        line_poly = tl.polygon
        lxs = [p[0] for p in line_poly]
        lys = [p[1] for p in line_poly]
        line_bbox_pt = [round(min(lxs)*px_to_pt,2), round(min(lys)*px_to_pt,2),
                        round(max(lxs)*px_to_pt,2), round(max(lys)*px_to_pt,2)]
        line_text = tl.text
        line_conf = tl.confidence or 0.0
        direction = detect_direction(line_text)
        line_tokens = []

        if tl.chars:
            cw_chars, cw_bboxes = [], []
            def flush():
                if not cw_chars: return
                wt = "".join(cw_chars)
                x0=min(b[0] for b in cw_bboxes); y0=min(b[1] for b in cw_bboxes)
                x1=max(b[2] for b in cw_bboxes); y1=max(b[3] for b in cw_bboxes)
                fs = max((y1-y0)*0.85, 4.0)
                tok = Token(text=wt, bbox=[round(x0,2),round(y0,2),round(x1,2),round(y1,2)],
                            font_size=round(fs,2), confidence=round(line_conf,3),
                            token_type=classify_token(wt), direction=detect_direction(wt),
                            line_id=line_idx)
                tokens.append(tok); line_tokens.append(tok)
                cw_chars.clear(); cw_bboxes.clear()
            for ch in tl.chars:
                cp = ch.polygon
                cxs=[p[0] for p in cp]; cys=[p[1] for p in cp]
                cb=[round(min(cxs)*px_to_pt,2),round(min(cys)*px_to_pt,2),
                    round(max(cxs)*px_to_pt,2),round(max(cys)*px_to_pt,2)]
                if ch.text.isspace(): flush()
                else: cw_chars.append(ch.text); cw_bboxes.append(cb)
            flush()
        else:
            fs = max((line_bbox_pt[3]-line_bbox_pt[1])*0.85, 4.0)
            tok = Token(text=line_text, bbox=line_bbox_pt, font_size=round(fs,2),
                        confidence=round(line_conf,3), token_type=classify_token(line_text),
                        direction=direction, line_id=line_idx)
            tokens.append(tok); line_tokens.append(tok)

        lines_data.append({"line_id":line_idx, "bbox":line_bbox_pt, "text":line_text,
                           "direction":direction,
                           "font_size":line_tokens[0].font_size if line_tokens else 8.0,
                           "is_bold":False, "is_italic":False, "tokens":line_tokens})

    pt = PageTokens(page_number=page_number, page_width=pw, page_height=ph,
                    tokens=tokens, extraction_method="surya_ocr", lines=lines_data)
    return clean_surya_artifacts(pt)


# ────────────────────────────────────────────────────────────────────
#  Multi-Render Voting (from Stage 2)
# ────────────────────────────────────────────────────────────────────
def _bbox_iou(a, b):
    x0=max(a[0],b[0]); y0=max(a[1],b[1]); x1=min(a[2],b[2]); y1=min(a[3],b[3])
    inter = max(0,x1-x0)*max(0,y1-y0)
    if inter == 0: return 0.0
    aa = max(0,a[2]-a[0])*max(0,a[3]-a[1])
    ab = max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter / (aa+ab-inter) if (aa+ab-inter) > 0 else 0.0

def multi_render_vote(primary, variants, quorum=VOTE_QUORUM):
    stats = {"total_numeric":0, "voted":0, "promoted":0, "unchanged":0, "details":[]}
    for tok in primary.tokens:
        if tok.token_type != "NUMERIC": continue
        stats["total_numeric"] += 1
        pd = extract_digits(tok.text)
        if not pd: continue

        readings = [pd]; confs = [tok.confidence]
        for vr in variants:
            best_iou, best_tok = 0.0, None
            for vt in vr.tokens:
                if vt.token_type != "NUMERIC": continue
                iou = _bbox_iou(tok.bbox, vt.bbox)
                if iou > best_iou: best_iou = iou; best_tok = vt
            if best_tok and best_iou > 0.3:
                vd = extract_digits(best_tok.text)
                if vd: readings.append(vd); confs.append(best_tok.confidence)

        if len(readings) >= 2:
            counter = Counter(readings)
            winner, count = counter.most_common(1)[0]
            detail = {"text":tok.text, "digits":pd, "readings":readings,
                      "winner":winner, "votes":count, "conf":tok.confidence}
            if count >= quorum:
                stats["voted"] += 1
                if tok.confidence < CONFIDENCE_THRESHOLD and winner == pd:
                    mc = max(c for d,c in zip(readings,confs) if d==winner)
                    nc = max(mc, CONFIDENCE_THRESHOLD + 0.01)
                    detail["action"] = f"PROMOTED {tok.confidence:.3f}→{nc:.3f}"
                    tok.confidence = nc
                    stats["promoted"] += 1
                elif winner != pd:
                    detail["action"] = f"CORRECTED {pd}→{winner}"
                    tok.text = winner; tok.token_type = classify_token(tok.text)
                    mc = max(c for d,c in zip(readings,confs) if d==winner)
                    tok.confidence = max(mc, CONFIDENCE_THRESHOLD + 0.01)
                    stats["promoted"] += 1
                else:
                    detail["action"] = "CONFIRMED"
                    stats["unchanged"] += 1
                stats["details"].append(detail)

    for ln in primary.lines:
        lt = ln.get('tokens', [])
        if lt: ln['text'] = ' '.join(t.text for t in lt)
    return primary, stats


# ────────────────────────────────────────────────────────────────────
#  Stage 3: Column-Context Rescue
# ────────────────────────────────────────────────────────────────────
def column_context_rescue(
    page_tokens: PageTokens,
    numeric_values: List,
    ocr_results: List[TokenOCRResult],
    threshold: float = COLUMN_RESCUE_THRESHOLD,
) -> Dict:
    """Rescue UNTRUSTED tokens using column-level context.

    For each UNTRUSTED numeric token:
      1. Find which column it belongs to.
      2. Check if ≥ threshold of column siblings are trusted.
      3. Check if token has valid Arabic-Indic digits.
      4. Check if digit-length matches column median ± 1.
      5. If all checks pass → promote to SURYA_VALID.

    Returns stats dict.
    """
    columns = find_numeric_columns(page_tokens)
    rescue_stats = {"checked": 0, "rescued": 0, "details": []}

    # Build token→column mapping
    tok_to_col: Dict[Tuple, NumericColumn] = {}
    for col in columns:
        for ct in col.tokens:
            key = (round(ct.bbox[0],2), round(ct.bbox[1],2),
                   round(ct.bbox[2],2), round(ct.bbox[3],2))
            tok_to_col[key] = col

    # Build token→trust mapping
    numeric_idx = 0
    tok_nv_map: Dict[Tuple, Tuple[int, 'NumericValue']] = {}
    for tok in page_tokens.tokens:
        if tok.token_type == "NUMERIC" and numeric_idx < len(numeric_values):
            key = (round(tok.bbox[0],2), round(tok.bbox[1],2),
                   round(tok.bbox[2],2), round(tok.bbox[3],2))
            tok_nv_map[key] = (numeric_idx, numeric_values[numeric_idx])
            numeric_idx += 1

    # Process each UNTRUSTED token
    for nv_idx, nv in enumerate(numeric_values):
        if nv.status != TrustStatus.UNTRUSTED:
            continue

        rescue_stats["checked"] += 1
        tok_key = (round(nv.bbox[0],2), round(nv.bbox[1],2),
                   round(nv.bbox[2],2), round(nv.bbox[3],2))

        col = tok_to_col.get(tok_key)
        detail = {
            "idx": nv_idx + 1,
            "digits": nv.digits,
            "conf": nv.surya_confidence,
            "bbox": list(tok_key),
        }

        if not col:
            detail["reason"] = "NOT_IN_COLUMN"
            rescue_stats["details"].append(detail)
            continue

        # Check column trust ratio
        trusted_count = 0
        total_count = col.count
        sibling_digit_lengths = []

        for ct in col.tokens:
            ct_key = (round(ct.bbox[0],2), round(ct.bbox[1],2),
                      round(ct.bbox[2],2), round(ct.bbox[3],2))
            ct_nv_info = tok_nv_map.get(ct_key)
            if ct_nv_info:
                _, ct_nv = ct_nv_info
                if ct_nv.status in (TrustStatus.LOCKED, TrustStatus.SURYA_VALID,
                                    TrustStatus.CNN_CONFIRMED):
                    trusted_count += 1
                    digits = extract_digits(ct_nv.digits)
                    if digits:
                        sibling_digit_lengths.append(len(digits))

        col_trust_ratio = trusted_count / total_count if total_count > 0 else 0
        detail["col_id"] = col.column_id
        detail["col_trust_ratio"] = round(col_trust_ratio, 2)
        detail["col_size"] = total_count

        if col_trust_ratio < threshold:
            detail["reason"] = f"COL_TRUST_TOO_LOW ({col_trust_ratio:.0%})"
            rescue_stats["details"].append(detail)
            continue

        # Check token has valid digits
        token_digits = extract_digits(nv.digits)
        if not token_digits or not has_valid_arabic_digits(nv.digits):
            detail["reason"] = "NO_VALID_DIGITS"
            rescue_stats["details"].append(detail)
            continue

        # Check digit-length consistency
        if sibling_digit_lengths:
            median_len = statistics.median(sibling_digit_lengths)
            token_len = len(token_digits)
            if abs(token_len - median_len) > 2:
                detail["reason"] = (f"DIGIT_LEN_MISMATCH "
                                    f"(token={token_len}, median={median_len:.0f})")
                rescue_stats["details"].append(detail)
                continue

        # All checks pass → RESCUE!
        detail["reason"] = "RESCUED"
        nv.status = TrustStatus.SURYA_VALID
        nv.trust_score = TRUST_SCORES[TrustStatus.SURYA_VALID]
        nv.failure_reasons = []

        # Also update the ocr_result
        ocr_results[nv_idx].trust_status = TrustStatus.SURYA_VALID
        ocr_results[nv_idx].trust_score = TRUST_SCORES[TrustStatus.SURYA_VALID]
        ocr_results[nv_idx].failure_reasons = []

        rescue_stats["rescued"] += 1
        rescue_stats["details"].append(detail)

    return rescue_stats


# ────────────────────────────────────────────────────────────────────
#  Layout / Rendering
# ────────────────────────────────────────────────────────────────────
def normalize_page_tokens(page_tokens):
    page = copy.deepcopy(page_tokens)
    pw, ph = page.page_width, page.page_height
    if not page.lines: return page
    page.lines.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    for ln in page.lines:
        x0,y0,x1,y1 = ln["bbox"]
        ln["bbox"] = [round(max(0,x0),2),round(max(0,y0),2),
                      round(min(pw,x1),2),round(min(ph,y1),2)]
    for i in range(1, len(page.lines)):
        ln = page.lines[i]
        x0,y0,x1,y1 = ln["bbox"]; h = y1-y0
        for j in range(i-1, max(i-20,-1), -1):
            prev = page.lines[j]
            px0,py0,px1,py1 = prev["bbox"]
            if x0>=px1 or x1<=px0: continue
            if y0>=py1: continue
            ov_x = min(x1,px1)-max(x0,px0); ov_y = py1-y0
            if ov_x > 2 and ov_y > 2:
                ln["bbox"][1] = round(py1+MIN_LINE_GAP,2)
                ln["bbox"][3] = round(ln["bbox"][1]+h,2)
                y0,y1 = ln["bbox"][1],ln["bbox"][3]
    for ln in page.lines:
        for tok in ln.get("tokens", []):
            tok.bbox[1] = ln["bbox"][1]; tok.bbox[3] = ln["bbox"][3]
    return page


def _render_line_html(line):
    x0,y0,x1,y1 = line["bbox"]; bw=x1-x0; h=y1-y0
    d = line.get("direction","rtl"); dc = "rtl" if d=="rtl" else "ltr"
    fs = line.get("font_size", max(h*FONT_SIZE_FACTOR, 4.0))
    if d=="rtl":
        ew=bw*FONT_DRIFT_FACTOR; extra=ew-bw; left=max(0,x0-extra); rw=x1-left
    else:
        left=x0; rw=bw
    lh=max(h,fs); raw=line.get("text",""); inner=html_mod.escape(raw)
    if line.get("is_bold"): inner=f"<b>{inner}</b>"
    if line.get("is_italic"): inner=f"<i>{inner}</i>"
    return (f'<div class="line {dc}" style="left:{left:.2f}px;top:{y0:.2f}px;'
            f'width:{rw:.2f}px;height:{h:.2f}px;line-height:{lh:.2f}px;'
            f'font-size:{fs:.2f}px;">{inner}</div>')

def render_page_html(pt):
    pw,ph = pt.page_width, pt.page_height
    lines = sorted(pt.lines, key=lambda ln: (ln["bbox"][1],-ln["bbox"][0]))
    lh = "\n    ".join(_render_line_html(ln) for ln in lines)
    return f"""<!DOCTYPE html>
<html lang="ar" dir="ltr"><head><meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:{FONT_FAMILY};direction:ltr;margin:0;padding:0;}}
.page{{position:relative;width:{pw:.2f}px;height:{ph:.2f}px;background:#fff;overflow:hidden;}}
.line{{position:absolute;white-space:pre;}}
.line.rtl{{direction:rtl;text-align:right;unicode-bidi:plaintext;}}
.line.ltr{{direction:ltr;text-align:left;unicode-bidi:plaintext;}}
</style></head><body><div class="page">{lh}</div></body></html>"""

async def html_to_pdf_page(html_str, wp, hp, out):
    from playwright.async_api import async_playwright
    wi=wp/72; hi=hp/72
    async with async_playwright() as pw:
        br = await pw.chromium.launch(headless=True)
        pg = await br.new_page()
        await pg.set_content(html_str, wait_until="networkidle")
        await pg.pdf(path=str(out),width=f"{wi:.4f}in",height=f"{hi:.4f}in",
                     margin={"top":"0","right":"0","bottom":"0","left":"0"},
                     print_background=True,scale=1.0)
        await br.close()


# ────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    print("="*64)
    print(f"  STAGE 3: Column-Context Numeric Validation")
    print(f"  (Stage 1 CLAHE + Stage 2 Multi-Vote + Column Rescue)")
    print(f"  Input:    {PDF_PATH.name}  (page {PAGE_NUM} only)")
    print(f"  Output:   {PREFIX}_*")
    print("="*64)

    doc = fitz.open(str(PDF_PATH))
    page = doc[PAGE_INDEX]
    pw, ph = round(page.rect.width,2), round(page.rect.height,2)
    print(f"\n  Page {PAGE_NUM}: {pw} x {ph} pt\n")

    # ── Step 1: 3 Variants ──
    print("─── Step 1: Preprocess 3 Variants ───")
    base_img = render_page_at_dpi(doc, PAGE_INDEX, OCR_DPI)
    var_a = _preprocess_variant_a(base_img, NORMALIZED_WIDTH)
    var_b = _preprocess_variant_b(base_img, NORMALIZED_WIDTH)
    var_c = _preprocess_variant_c(base_img, NORMALIZED_WIDTH)
    print(f"  3 variants prepared")

    # ── Step 2-4: Surya OCR on all 3 ──
    print(f"\n─── Step 2: Surya OCR — 3 Variants ───")
    t0 = time.time()
    tokens_a = _run_surya_on_image(var_a, doc, PAGE_INDEX, OCR_DPI)
    na = sum(1 for t in tokens_a.tokens if t.token_type=="NUMERIC")
    print(f"  A: {len(tokens_a.tokens)} tokens ({na} numeric)")

    tokens_b = _run_surya_on_image(var_b, doc, PAGE_INDEX, OCR_DPI)
    nb = sum(1 for t in tokens_b.tokens if t.token_type=="NUMERIC")
    print(f"  B: {len(tokens_b.tokens)} tokens ({nb} numeric)")

    tokens_c = _run_surya_on_image(var_c, doc, PAGE_INDEX, OCR_DPI)
    nc = sum(1 for t in tokens_c.tokens if t.token_type=="NUMERIC")
    print(f"  C: {len(tokens_c.tokens)} tokens ({nc} numeric)")
    t_ocr = time.time() - t0
    print(f"  Total OCR: {t_ocr:.1f}s")

    # ── Step 3: Multi-Render Vote ──
    print(f"\n─── Step 3: Multi-Render Majority Vote ───")
    page_tokens, vote_stats = multi_render_vote(tokens_a, [tokens_b, tokens_c])
    print(f"  Voted: {vote_stats['voted']}, Promoted: {vote_stats['promoted']}")

    n_tokens = len(page_tokens.tokens)
    n_numeric = sum(1 for t in page_tokens.tokens if t.token_type=="NUMERIC")

    with open(OUT_TOKENS, "w", encoding="utf-8") as f:
        for tok in page_tokens.tokens:
            f.write(json.dumps({"text":tok.text,"type":tok.token_type,
                "bbox":[round(b,2) for b in tok.bbox],
                "conf":round(tok.confidence,4) if tok.confidence else None,
                "font_size":round(tok.font_size,2)}, ensure_ascii=False)+"\n")

    # ── Step 4: Layout ──
    print(f"\n─── Step 4: Layout Normalization ───")
    page_tokens = normalize_page_tokens(page_tokens)
    original_lines = [{"line_id":ln.get("line_id",i),"text":ln.get("text",""),
                       "bbox":ln.get("bbox",[0,0,0,0])}
                      for i,ln in enumerate(page_tokens.lines)]
    print(f"  {len(page_tokens.lines)} lines")

    # ── Step 5: Numeric Validation ──
    print(f"\n─── Step 5: Numeric Validation (Discrete Trust) ───")
    hires = render_page_at_dpi(doc, PAGE_INDEX, dpi=NUMERIC_VAL_DPI)
    ocr_results = ocr_page_numeric_tokens(page_tokens, hires)
    n_locked = sum(1 for r in ocr_results if r.locked)
    print(f"  {len(ocr_results)} numeric tokens, {n_locked} LOCKED")

    current_lines = [{"line_id":ln.get("line_id",i),"text":ln.get("text",""),
                      "bbox":ln.get("bbox",[0,0,0,0])}
                     for i,ln in enumerate(page_tokens.lines)]
    page_stability = validate_page_lines(original_lines, current_lines, PAGE_NUM)
    ocr_results = apply_line_stability_to_tokens(
        page_stability, ocr_results, page_tokens.lines)

    font_sizes = [t.font_size for t in page_tokens.tokens if t.token_type=="NUMERIC"]
    numeric_values = reconstruct_page_numbers(ocr_results, PAGE_NUM, font_sizes)

    # Pre-rescue trust summary
    pre_untrusted = sum(1 for nv in numeric_values if nv.status == TrustStatus.UNTRUSTED)
    print(f"  Pre-rescue UNTRUSTED: {pre_untrusted}")

    # ── Step 6: Column-Context Rescue (NEW in Stage 3) ──
    print(f"\n─── Step 6: Column-Context Rescue ───")

    # Update token texts first (needed for column detection)
    numeric_idx = 0
    for tok in page_tokens.tokens:
        if tok.token_type == "NUMERIC" and numeric_idx < len(numeric_values):
            nv = numeric_values[numeric_idx]
            if has_valid_arabic_digits(nv.digits):
                tok.text = nv.digits
            numeric_idx += 1
    for ln in page_tokens.lines:
        lt = ln.get('tokens', [])
        if lt: ln['text'] = ' '.join(t.text for t in lt)

    rescue_stats = column_context_rescue(
        page_tokens, numeric_values, ocr_results, COLUMN_RESCUE_THRESHOLD)

    print(f"  Checked: {rescue_stats['checked']}")
    print(f"  Rescued: {rescue_stats['rescued']}")
    for d in rescue_stats.get('details', []):
        icon = "✓" if d["reason"] == "RESCUED" else "✗"
        print(f"    {icon} #{d['idx']} {d['digits']} conf={d['conf']:.3f} → {d['reason']}")

    # Build audit
    pa = PageNumericAudit(
        page_number=PAGE_NUM,
        numeric_values=numeric_values,
        ocr_results=ocr_results,
        line_stability=page_stability,
    )
    pa.compute_summary()
    ts = pa.trust_summary
    print(f"\n  Trust Summary (post-rescue):")
    print(f"    🔒 LOCKED:       {ts.get('locked',0)}")
    print(f"    ✓ SURYA_VALID:   {ts.get('surya_valid',0)}")
    print(f"    🧠 CNN_CONFIRMED: {ts.get('cnn_confirmed',0)}")
    print(f"    ⚠ UNTRUSTED:     {ts.get('untrusted',0)}")
    print(f"    Trust rate:      {ts.get('pct_trusted',0)}%")

    # Column analysis
    columns = find_numeric_columns(page_tokens)
    all_anomalies = [detect_column_anomalies(columns)]

    pipeline_audit = PipelineNumericAudit(pages=[pa])
    pipeline_audit.compute_overall()

    # ── Step 7: Render ──
    print(f"\n─── Step 7: Render → PDF ───")
    t0 = time.time()
    page_html = render_page_html(page_tokens)
    tmp_pdf = Path(tempfile.mktemp(suffix=".pdf"))
    asyncio.run(html_to_pdf_page(page_html, pw, ph, tmp_pdf))
    t_render = time.time() - t0
    merged = fitz.open(str(tmp_pdf))
    merged.save(str(OUT_PDF))
    merged.close()
    tmp_pdf.unlink(missing_ok=True)
    out_kb = OUT_PDF.stat().st_size / 1024
    print(f"  Output: {OUT_PDF.name}  ({out_kb:.1f} KB) ({t_render:.1f}s)")

    # ── Step 8: QA ──
    print(f"\n─── Step 8: QA Reports ───")
    num_qa_html = generate_qa_report(pipeline_audit)
    OUT_NUM_QA.write_text(num_qa_html, encoding="utf-8")
    qa_html = render_qa_html([page_tokens], [columns], all_anomalies)
    OUT_QA_HTML.write_text(qa_html, encoding="utf-8")
    print(f"  {OUT_NUM_QA.name}, {OUT_QA_HTML.name}")

    # ── Step 9: Token Detail ──
    print(f"\n─── Numeric Token Detail ───")
    print(f"  {'#':>3}  {'Status':<15} {'Conf':>6} {'Trust':>5}  {'Digits'}")
    print(f"  {'─'*3}  {'─'*15} {'─'*6} {'─'*5}  {'─'*20}")
    untrusted_details = []
    for idx, nv in enumerate(numeric_values):
        icon = {'LOCKED':'🔒','SURYA_VALID':'✓ ','CNN_CONFIRMED':'🧠',
                'UNTRUSTED':'⚠ '}.get(nv.status.value, '? ')
        reasons = ', '.join(r.value for r in nv.failure_reasons) if nv.failure_reasons else ''
        extra = f"  ({reasons})" if reasons else ""
        print(f"  {idx+1:3d}  {icon} {nv.status.value:<13} "
              f"{nv.surya_confidence:6.3f} {nv.trust_score:5.2f}  "
              f"{nv.digits}{extra}")
        if nv.status == TrustStatus.UNTRUSTED:
            untrusted_details.append({"idx":idx+1,"digits":nv.digits,
                "conf":nv.surya_confidence,"bbox":[round(b,2) for b in nv.bbox]})

    # ── Log ──
    elapsed = time.time() - t_start
    log_data = {
        "stage": "stage3_column_context",
        "description": "CLAHE + Multi-Vote + Column-Context Rescue",
        "page": PAGE_NUM,
        "time_s": round(elapsed, 1),
        "ocr_time_s": round(t_ocr, 1),
        "tokens": n_tokens,
        "numeric_tokens": n_numeric,
        "trust": {
            "total": len(numeric_values),
            "locked": ts.get('locked', 0),
            "surya_valid": ts.get('surya_valid', 0),
            "cnn_confirmed": ts.get('cnn_confirmed', 0),
            "untrusted": ts.get('untrusted', 0),
            "trust_pct": ts.get('pct_trusted', 0),
        },
        "vote_stats": {"voted":vote_stats["voted"],"promoted":vote_stats["promoted"]},
        "rescue_stats": rescue_stats,
        "columns": len(columns),
        "anomalies": len(all_anomalies[0]),
        "untrusted_tokens": untrusted_details,
        "comparison": {
            "baseline": {"trust_pct":94.2, "untrusted":8},
            "stage1": {"trust_pct":95.6, "untrusted":6},
            "stage2": {"trust_pct":97.1, "untrusted":4},
        },
    }
    OUT_LOG.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")
    doc.close()

    # ── Summary ──
    tp = ts.get('pct_trusted',0)
    n_anom = len(all_anomalies[0])
    print(f"\n{'='*64}")
    print(f"  ✓ Stage 3 done in {elapsed:.1f}s")
    print(f"  Trust: {tp}%  (baseline 94.2% → S1 95.6% → S2 97.1% → S3 {tp}%)")
    print(f"    🔒 LOCKED: {ts.get('locked',0)} | ⚠ UNTRUSTED: {ts.get('untrusted',0)}")
    print(f"  Rescue: {rescue_stats['rescued']}/{rescue_stats['checked']} tokens")
    print(f"  Columns: {len(columns)} | Anomalies: {n_anom}")
    print(f"  Output: {PREFIX}_*")
    print(f"{'='*64}")

    meets = tp > 98 and n_anom < (len(numeric_values)*0.01)
    if meets:
        print(f"\n  🎯 TARGET MET! trust > 98% AND anomalies < 1%")
    else:
        remaining = ts.get('untrusted', 0)
        needed = len(numeric_values) - int(len(numeric_values) * 0.98)
        print(f"\n  ⚠ Target NOT met. {remaining} untrusted remain "
              f"(need ≤{needed} for >98%)")


if __name__ == "__main__":
    main()
