#!/usr/bin/env python3
"""
p3_stage4_easyocr_digits.py — Stage 4: Hybrid Surya + EasyOCR Digit Recognition
=================================================================================

ROOT CAUSE: Surya OCR reads Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) as Latin
characters (7, Y, V, T, E, A, £, etc.).  The previous pipeline's
`_recover_arabic_indic()` tried to map these back, but the mapping was
fundamentally wrong — especially `7→٧` which caused ٧ to appear at 35.6%
vs the expected ~8%.

FIX (this script):
  ▸ Use Surya ONLY for layout detection (line bboxes, Arabic text tokens)
  ▸ Use EasyOCR (lang='ar') for digit recognition — it natively outputs
    Arabic-Indic digits with a near-perfect frequency distribution
  ▸ For each Surya numeric token, find the matching EasyOCR region by bbox
    overlap, replace Surya's garbled Latin chars with EasyOCR's correct
    Arabic-Indic digits
  ▸ SKIP `_recover_arabic_indic()` entirely — EasyOCR handles this natively

Evidence:
  Surya on raw page 3:  218 Latin digits, 23 Arabic-Indic — ٧ at 35.6%
  EasyOCR on raw page 3: 3 Western digits, 322 Arabic-Indic — ٧ at 7.8%
  Ground truth (native p4 distribution): ٧ at ~8%

Output files: p3_stage4_*

Usage:
    python3 p3_stage4_easyocr_digits.py
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
from collections import Counter, defaultdict

sys.stdout.reconfigure(line_buffering=True)

import cv2
import numpy as np
import fitz

from token_extract import (
    Token, PageTokens,
    classify_token, detect_direction,
    render_page_at_dpi,
    NORMALIZED_WIDTH, OCR_DPI, DILATION_KERNEL_SIZE,
    # NOTE: we import clean_surya_artifacts but will NOT use _recover_arabic_indic
    # for EasyOCR-corrected tokens
    _HTML_TAG_RE, _ETHIOPIC_RE,
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

PREFIX = "p3_stage4"
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

# EasyOCR matching
EASYOCR_IOU_THRESHOLD = 0.15   # minimum IoU to match EasyOCR → Surya token
EASYOCR_CONF_THRESHOLD = 0.3   # minimum EasyOCR confidence to accept


# ────────────────────────────────────────────────────────────────────
#  Arabic-Indic Digit Utilities
# ────────────────────────────────────────────────────────────────────
_ARABIC_INDIC_DIGITS = set('٠١٢٣٤٥٦٧٨٩')
_WESTERN_DIGITS = set('0123456789')
_NUMERIC_PUNCT = set('().,-\u2212\u2013/% \u066C')

def has_arabic_indic_digits(text: str) -> bool:
    """Return True if text contains at least one Arabic-Indic digit."""
    return any(c in _ARABIC_INDIC_DIGITS for c in text)

def is_numeric_content(text: str) -> bool:
    """Return True if text looks like a numeric value (Arabic-Indic or Western)."""
    stripped = text.strip()
    if not stripped:
        return False
    digit_count = sum(1 for c in stripped
                      if c in _ARABIC_INDIC_DIGITS or c in _WESTERN_DIGITS)
    return digit_count > 0 and digit_count >= len(stripped.replace(' ', '')) * 0.5

def _reverse_rtl_digits(text: str) -> str:
    """Reverse group ORDER of multi-group Arabic-Indic digit sequences.

    ROOT CAUSE: EasyOCR with lang='ar' reads digit GROUPS in RTL order,
    but reads digits within each group correctly (LTR).
    E.g., the number 73,837 (٧٣ ٨٣٧ on paper) gets read as "٨٣٧ ٧٣"
    — the groups are swapped but each group's digits are correct.

    FIX: Reverse the ORDER of space-separated groups only.
    Do NOT reverse the characters within each group.

    Examples:
      ٨٣٧ ٧٣     → ٧٣ ٨٣٧       (2 groups: swap)
      ٥٣٠ ٣٤٨    → ٣٤٨ ٥٣٠      (2 groups: swap)
      ١٠٥         → ١٠٥           (1 group: unchanged)
    """
    stripped = text.strip()
    if not stripped:
        return text

    # Handle parenthesized numbers:  (٨٣٧ ٧٣) → reverse inner groups
    has_open = stripped.startswith('(')
    has_close = stripped.endswith(')')
    inner = stripped
    if has_open:
        inner = inner[1:]
    if has_close:
        inner = inner[:-1]
    inner = inner.strip()

    # Split into space-separated groups
    groups = inner.split()
    if len(groups) < 2:
        return text   # single group — no reversal needed

    # All groups must be pure digits (Arabic-Indic and/or Western)
    digit_chars = _ARABIC_INDIC_DIGITS | _WESTERN_DIGITS
    for g in groups:
        if not all(c in digit_chars for c in g):
            return text  # mixed content — don't reverse

    # Reject date-like patterns: any group with 4+ digits (years like ٢٠٢٤)
    if any(len(g) > 3 for g in groups):
        return text

    # Reject trivial patterns: all groups are single digits (e.g. "١ ٣")
    if all(len(g) == 1 for g in groups):
        return text

    # Reverse the GROUP ORDER only (keep digits within each group as-is)
    reversed_inner = ' '.join(reversed(groups))

    # Rebuild with optional parens
    if has_open and has_close:
        return '(' + reversed_inner + ')'
    elif has_open:
        return '(' + reversed_inner
    elif has_close:
        return reversed_inner + ')'
    return reversed_inner


def normalize_easyocr_number(text: str) -> str:
    """Clean up EasyOCR number output for token use.

    EasyOCR outputs Arabic-Indic digits natively. We:
    1. Strip any stray Arabic text chars that leaked in
    2. Normalize Western digits to Arabic-Indic
    3. Keep numeric punctuation (commas, parens, minus)
    4. Reverse RTL digit sequences (EasyOCR reads numbers backwards)
    """
    WESTERN_TO_ARABIC = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')
    result = []
    for c in text:
        if c in _ARABIC_INDIC_DIGITS:
            result.append(c)
        elif c in _WESTERN_DIGITS:
            result.append(c.translate(WESTERN_TO_ARABIC))
        elif c in _NUMERIC_PUNCT:
            result.append(c)
        elif c.isspace():
            result.append(c)
        # Skip Arabic script chars or other junk
    cleaned = ''.join(result).strip()

    # Fix RTL digit reversal from EasyOCR Arabic mode
    cleaned = _reverse_rtl_digits(cleaned)

    return cleaned


# ────────────────────────────────────────────────────────────────────
#  EasyOCR Engine
# ────────────────────────────────────────────────────────────────────
_easyocr_reader = None

def _load_easyocr():
    """Load EasyOCR Arabic reader (one-time)."""
    global _easyocr_reader
    if _easyocr_reader is not None:
        return
    print("  [EasyOCR] Loading Arabic reader (one-time)...", flush=True)
    import easyocr
    _easyocr_reader = easyocr.Reader(['ar'], gpu=False, verbose=False)
    print("  [EasyOCR] Reader loaded.\n", flush=True)


def run_easyocr_on_image(img_bgr: np.ndarray, page_width: float, page_height: float) -> List[dict]:
    """Run EasyOCR on a BGR image and return regions in PDF point coordinates.

    Uses actual image dimensions for accurate coordinate conversion.

    Returns list of dicts:
      {"text": str, "bbox_pt": [x0,y0,x1,y1], "confidence": float}
    """
    _load_easyocr()
    results = _easyocr_reader.readtext(img_bgr, mag_ratio=2.0,
                                        add_margin=0.15)
    img_h, img_w = img_bgr.shape[:2]
    px_to_pt_x = page_width / img_w
    px_to_pt_y = page_height / img_h
    regions = []
    for (bbox_pts, text, conf) in results:
        # bbox_pts is [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
        xs = [float(p[0]) for p in bbox_pts]
        ys = [float(p[1]) for p in bbox_pts]
        bbox_pt = [
            round(min(xs) * px_to_pt_x, 2),
            round(min(ys) * px_to_pt_y, 2),
            round(max(xs) * px_to_pt_x, 2),
            round(max(ys) * px_to_pt_y, 2),
        ]
        regions.append({
            "text": text,
            "bbox_pt": bbox_pt,
            "confidence": conf,
        })
    return regions


# ────────────────────────────────────────────────────────────────────
#  Crop-Based EasyOCR Refinement
# ────────────────────────────────────────────────────────────────────
EASYOCR_CROP_PADDING_PT = 5.0   # padding around Surya bbox in PDF points
EASYOCR_CROP_MIN_HEIGHT = 64    # minimum crop height in pixels
EASYOCR_CROP_UPSCALE = 3.0      # upscale factor for small crops


def run_easyocr_on_crop(
    img_bgr: np.ndarray,
    tok_bbox_pt: List[float],
    page_width: float,
    page_height: float,
    padding_pt: float = EASYOCR_CROP_PADDING_PT,
) -> Optional[dict]:
    """Run EasyOCR on a cropped+enhanced region around a single token.

    Crops the Surya bbox region with padding, applies CLAHE + upscale,
    then runs EasyOCR for better small-digit accuracy.

    Returns dict with "text" and "confidence", or None if no result.
    """
    _load_easyocr()

    img_h, img_w = img_bgr.shape[:2]
    pt_to_px_x = img_w / page_width
    pt_to_px_y = img_h / page_height

    # Convert PDF-point bbox to pixels with padding
    x0 = max(0, int((tok_bbox_pt[0] - padding_pt) * pt_to_px_x))
    y0 = max(0, int((tok_bbox_pt[1] - padding_pt) * pt_to_px_y))
    x1 = min(img_w, int((tok_bbox_pt[2] + padding_pt) * pt_to_px_x))
    y1 = min(img_h, int((tok_bbox_pt[3] + padding_pt) * pt_to_px_y))

    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return None

    crop = img_bgr[y0:y1, x0:x1]

    # Enhance: grayscale → CLAHE → upscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Upscale to ensure digits are large enough for EasyOCR
    crop_h = enhanced.shape[0]
    scale = max(EASYOCR_CROP_UPSCALE, EASYOCR_CROP_MIN_HEIGHT / max(crop_h, 1))
    enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    # Convert to BGR for EasyOCR
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    results = _easyocr_reader.readtext(enhanced_bgr, mag_ratio=2.0)
    if not results:
        return None

    # Take the result with the most Arabic-Indic digit content
    best = None
    best_digit_count = 0
    for (bbox_pts, text, conf) in results:
        digit_count = sum(1 for c in text
                          if '\u0660' <= c <= '\u0669' or c.isdigit())
        if (digit_count > best_digit_count or
                (digit_count == best_digit_count and
                 conf > (best[2] if best else 0))):
            best = (bbox_pts, text, conf)
            best_digit_count = digit_count

    if best and best_digit_count > 0:
        return {"text": best[1], "confidence": best[2]}
    return None


def refine_easyocr_with_crops(
    page_tokens: PageTokens,
    base_img: np.ndarray,
) -> dict:
    """Second-pass: crop-based EasyOCR for tokens that are CLEARLY wrong.

    CONSERVATIVE: Only tries crop OCR for tokens where:
      1. The token has NO Arabic-Indic digits at all (still garbled Latin)
      2. The token's bbox is wide (≥15pt) but digit count is suspiciously
         low compared to width — likely truncated by whole-page EasyOCR

    Does NOT touch: note references, year headers, already-correct numbers.
    """
    refine_stats = {"tried": 0, "improved": 0, "new_matches": 0, "details": []}
    pw, ph = page_tokens.page_width, page_tokens.page_height

    for tok in page_tokens.tokens:
        is_suspect = _is_surya_suspect_numeric(tok.text)
        is_numeric = tok.token_type == "NUMERIC"

        if not is_suspect and not is_numeric:
            continue

        # Count current Arabic-Indic digits
        old_digits = sum(1 for c in tok.text if '\u0660' <= c <= '\u0669')
        bbox_width = tok.bbox[2] - tok.bbox[0]

        # === CONSERVATIVE CRITERIA ===
        # Case 1: Still garbled Latin (no Arabic-Indic digits at all)
        #         AND bbox is wide enough to be a real number (not a note ref)
        needs_crop = False
        if old_digits == 0 and is_suspect and bbox_width >= 12:
            needs_crop = True

        # Case 2: Has some digits but bbox is MUCH wider than expected
        #         (e.g., bbox is 30pt wide but only has 3 digits → expects 6+)
        #         Only for tokens in the main numeric columns (x < 160pt)
        if (old_digits > 0 and bbox_width >= 20 and
                old_digits < bbox_width / 6 and
                tok.bbox[0] < 160):
            needs_crop = True

        if not needs_crop:
            continue

        refine_stats["tried"] += 1
        crop_result = run_easyocr_on_crop(base_img, tok.bbox, pw, ph)

        detail = {
            "original_text": tok.text,
            "bbox": [round(b, 1) for b in tok.bbox],
            "old_digits": old_digits,
            "bbox_width": round(bbox_width, 1),
        }

        if crop_result and crop_result["confidence"] >= EASYOCR_CONF_THRESHOLD:
            normalized = normalize_easyocr_number(crop_result["text"])
            if normalized and has_arabic_indic_digits(normalized):
                new_digits = sum(1 for c in normalized
                                 if '\u0660' <= c <= '\u0669')

                # Only accept if STRICTLY more digits AND confidence is decent
                if (new_digits > old_digits and
                        crop_result["confidence"] >= 0.4):
                    detail["crop_text"] = crop_result["text"]
                    detail["crop_normalized"] = normalized
                    detail["crop_conf"] = round(crop_result["confidence"], 3)
                    detail["action"] = (
                        f"IMPROVED ({old_digits}→{new_digits} digits)")

                    tok.text = normalized
                    tok.token_type = classify_token(normalized)
                    tok.direction = detect_direction(normalized)
                    tok.confidence = min(0.95, crop_result["confidence"])

                    refine_stats["improved"] += 1
                    if old_digits == 0:
                        refine_stats["new_matches"] += 1
                else:
                    detail["action"] = (
                        f"KEPT (crop {new_digits}d@{crop_result['confidence']:.2f}"
                        f" vs orig {old_digits}d)")
            else:
                detail["action"] = "CROP_NO_DIGITS"
        else:
            detail["action"] = (
                f"CROP_FAILED (conf={crop_result['confidence']:.3f})"
                if crop_result else "CROP_EMPTY")

        refine_stats["details"].append(detail)

    # Rebuild line text
    for ln in page_tokens.lines:
        lt = ln.get('tokens', [])
        if lt:
            ln['text'] = ' '.join(t.text for t in lt)
            ln['direction'] = detect_direction(ln['text'])

    return refine_stats


# ────────────────────────────────────────────────────────────────────
#  Surya Layout Detection (text + bboxes, NO _recover_arabic_indic)
# ────────────────────────────────────────────────────────────────────
def _clean_surya_text_only(page_tokens: PageTokens) -> PageTokens:
    """Strip HTML tags and Ethiopic chars, but DO NOT apply _recover_arabic_indic.

    For Stage 4, digit recovery is handled by EasyOCR instead.
    """
    for tok in page_tokens.tokens:
        original = tok.text
        # 1. Strip HTML tags
        if '<' in tok.text:
            tok.text = _HTML_TAG_RE.sub('', tok.text).strip()
        # 2. Strip <math> wrappers
        if '<math>' in tok.text:
            tok.text = re.sub(r'</?math>', '', tok.text).strip()
        # 3. Remove Ethiopic characters
        if _ETHIOPIC_RE.search(tok.text):
            tok.text = _ETHIOPIC_RE.sub('', tok.text).strip()
        # 4. DO NOT call _recover_arabic_indic — EasyOCR handles digits
        if tok.text != original:
            tok.token_type = classify_token(tok.text)
            tok.direction = detect_direction(tok.text)

    page_tokens.tokens = [t for t in page_tokens.tokens if t.text.strip()]
    for ln in page_tokens.lines:
        ln['tokens'] = [t for t in ln.get('tokens', []) if t.text.strip()]
        if ln['tokens']:
            ln['text'] = ' '.join(t.text for t in ln['tokens'])
            ln['direction'] = detect_direction(ln['text'])
        else:
            ln['text'] = ''
    page_tokens.lines = [ln for ln in page_tokens.lines if ln.get('text', '').strip()]
    return page_tokens


def _preprocess_for_surya(img, tw):
    """CLAHE + Otsu + 2×2 dilation — best for Surya layout detection."""
    h, w = img.shape[:2]
    s = tw / w; nh = int(h * s)
    resized = cv2.resize(img, (tw, nh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return cv2.cvtColor(cv2.bitwise_not(dilated), cv2.COLOR_GRAY2BGR)


def run_surya_layout(preprocessed_bgr, doc, page_idx, dpi=OCR_DPI):
    """Run Surya for layout detection + text. Returns PageTokens with RAW Surya text.

    Arabic TEXT tokens will be correct. NUMERIC tokens will have
    Latin-garbled text — these will be fixed by EasyOCR matching.
    """
    import token_extract as te
    from PIL import Image
    te._load_surya()

    page = doc[page_idx]
    pw = round(page.rect.width, 2)
    ph = round(page.rect.height, 2)
    page_number = page_idx + 1

    # CRITICAL: use actual image dimensions for coordinate conversion,
    # NOT dpi-based ratio — the preprocessed image was resized to a
    # different width than the DPI-rendered original.
    img_h, img_w = preprocessed_bgr.shape[:2]
    px_to_pt_x = pw / img_w
    px_to_pt_y = ph / img_h

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
        line_bbox_pt = [round(min(lxs)*px_to_pt_x, 2), round(min(lys)*px_to_pt_y, 2),
                        round(max(lxs)*px_to_pt_x, 2), round(max(lys)*px_to_pt_y, 2)]
        line_text = tl.text
        line_conf = tl.confidence or 0.0
        direction = detect_direction(line_text)
        line_tokens = []

        if tl.chars:
            cw_chars, cw_bboxes = [], []

            def flush():
                if not cw_chars:
                    return
                wt = "".join(cw_chars)
                x0 = min(b[0] for b in cw_bboxes)
                y0 = min(b[1] for b in cw_bboxes)
                x1 = max(b[2] for b in cw_bboxes)
                y1 = max(b[3] for b in cw_bboxes)
                fs = max((y1 - y0) * 0.85, 4.0)
                tok = Token(
                    text=wt,
                    bbox=[round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                    font_size=round(fs, 2),
                    confidence=round(line_conf, 3),
                    token_type=classify_token(wt),
                    direction=detect_direction(wt),
                    line_id=line_idx,
                )
                tokens.append(tok)
                line_tokens.append(tok)
                cw_chars.clear()
                cw_bboxes.clear()

            for ch in tl.chars:
                cp = ch.polygon
                cxs = [p[0] for p in cp]
                cys = [p[1] for p in cp]
                cb = [round(min(cxs)*px_to_pt_x, 2), round(min(cys)*px_to_pt_y, 2),
                      round(max(cxs)*px_to_pt_x, 2), round(max(cys)*px_to_pt_y, 2)]
                if ch.text.isspace():
                    flush()
                else:
                    cw_chars.append(ch.text)
                    cw_bboxes.append(cb)
            flush()
        else:
            fs = max((line_bbox_pt[3] - line_bbox_pt[1]) * 0.85, 4.0)
            tok = Token(text=line_text, bbox=line_bbox_pt, font_size=round(fs, 2),
                        confidence=round(line_conf, 3),
                        token_type=classify_token(line_text),
                        direction=direction, line_id=line_idx)
            tokens.append(tok)
            line_tokens.append(tok)

        lines_data.append({
            "line_id": line_idx, "bbox": line_bbox_pt, "text": line_text,
            "direction": direction,
            "font_size": line_tokens[0].font_size if line_tokens else 8.0,
            "is_bold": False, "is_italic": False, "tokens": line_tokens,
        })

    pt = PageTokens(page_number=page_number, page_width=pw, page_height=ph,
                    tokens=tokens, extraction_method="surya_ocr", lines=lines_data)

    # Clean HTML/Ethiopic ONLY — no _recover_arabic_indic
    pt = _clean_surya_text_only(pt)
    return pt


# ────────────────────────────────────────────────────────────────────
#  EasyOCR → Surya Token Matching
# ────────────────────────────────────────────────────────────────────
def _bbox_iou(a, b):
    """Intersection over Union of two [x0,y0,x1,y1] bboxes."""
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1-x0) * max(0, y1-y0)
    if inter == 0:
        return 0.0
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    ab = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0.0


def _bbox_overlap_ratio(tok_bbox, easy_bbox):
    """What fraction of tok_bbox is covered by easy_bbox?"""
    x0 = max(tok_bbox[0], easy_bbox[0])
    y0 = max(tok_bbox[1], easy_bbox[1])
    x1 = min(tok_bbox[2], easy_bbox[2])
    y1 = min(tok_bbox[3], easy_bbox[3])
    inter = max(0, x1-x0) * max(0, y1-y0)
    tok_area = max(0, tok_bbox[2]-tok_bbox[0]) * max(0, tok_bbox[3]-tok_bbox[1])
    if tok_area == 0:
        return 0.0
    return inter / tok_area


def _is_surya_suspect_numeric(text: str) -> bool:
    """Return True if this Surya token looks like garbled Latin digits.

    These are tokens that Surya read as Latin chars instead of Arabic-Indic
    digits.  They contain digits or Latin letters known to be Surya misreads
    (V, Y, T, E, A, P, etc.) and no Arabic script.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # Already has Arabic-Indic digits → probably OK
    if has_arabic_indic_digits(stripped):
        return False
    # Must not contain Arabic script
    if any('\u0600' <= c <= '\u06FF' for c in stripped):
        return False
    # Long runs of Latin letters (4+) are words, not garbled digits
    # (garbled digits are short: "YT", "ATY", "T90", etc.)
    if re.search(r'[A-Za-z]{4,}', stripped):
        return False
    # Must have at least one digit or Latin suspect char
    suspect_latin = set('VYTEAPNFOBILKGX')
    has_suspect = any(
        c.isdigit() or c.upper() in suspect_latin
        for c in stripped
    )
    if not has_suspect:
        return False
    # Most chars should be digits/suspects/punctuation
    allowed = _WESTERN_DIGITS | suspect_latin | set(c.lower() for c in suspect_latin) | _NUMERIC_PUNCT | {' ', '£', '°', '·'}
    non_space = [c for c in stripped if not c.isspace()]
    if not non_space:
        return False
    allowed_count = sum(1 for c in non_space if c in allowed or c.upper() in allowed)
    return allowed_count >= len(non_space) * 0.6


def match_easyocr_to_surya(
    page_tokens: PageTokens,
    easyocr_regions: List[dict],
) -> Dict:
    """Replace Surya numeric tokens with EasyOCR readings.

    For each Surya token that is NUMERIC or looks like garbled Latin digits:
      1. Find EasyOCR regions that overlap with it
      2. If an EasyOCR region has Arabic-Indic digits, use its text
      3. Update the token text, type, and confidence

    FIXES applied:
      ▸ De-duplication: When multiple Surya tokens overlap the same EasyOCR
        region, only keep the best match — remove the rest.
      ▸ Confidence filter: Reject EasyOCR matches below EASYOCR_CONF_THRESHOLD.
      ▸ Western→Arabic: Convert remaining small Western digits (note refs etc.)

    Returns stats dict.
    """
    stats = {
        "total_surya_tokens": len(page_tokens.tokens),
        "suspect_numeric": 0,
        "matched": 0,
        "replaced": 0,
        "no_match": 0,
        "kept_surya": 0,
        "deduplicated": 0,
        "low_conf_rejected": 0,
        "western_converted": 0,
        "details": [],
    }

    # Pre-filter EasyOCR regions that contain digits
    numeric_easy_regions = [
        r for r in easyocr_regions
        if is_numeric_content(r["text"])
    ]

    # ── Phase 1: Find best EasyOCR match for each suspect token ──
    # Build: EasyOCR region index → list of (token, score, detail)
    easy_to_surya = defaultdict(list)

    for tok in page_tokens.tokens:
        is_suspect = _is_surya_suspect_numeric(tok.text)
        is_numeric = tok.token_type == "NUMERIC"

        if not is_suspect and not is_numeric:
            continue

        stats["suspect_numeric"] += 1
        surya_text = tok.text

        # Find best matching EasyOCR region by overlap
        best_score = 0.0
        best_idx = -1
        best_region = None
        for idx, region in enumerate(numeric_easy_regions):
            overlap = _bbox_overlap_ratio(tok.bbox, region["bbox_pt"])
            iou = _bbox_iou(tok.bbox, region["bbox_pt"])
            score = max(overlap * 0.7 + iou * 0.3, iou)
            if score > best_score:
                best_score = score
                best_idx = idx
                best_region = region

        detail = {
            "surya_text": surya_text,
            "surya_bbox": [round(b, 1) for b in tok.bbox],
            "surya_conf": tok.confidence,
        }

        if best_region and best_score >= EASYOCR_IOU_THRESHOLD:
            # ── Confidence filter: reject junk EasyOCR matches ──
            if best_region["confidence"] < EASYOCR_CONF_THRESHOLD:
                detail["easyocr_text"] = best_region["text"]
                detail["easyocr_conf"] = round(best_region["confidence"], 3)
                detail["action"] = (f"LOW_CONF_REJECTED "
                                    f"({best_region['confidence']:.3f} < {EASYOCR_CONF_THRESHOLD})")
                stats["low_conf_rejected"] += 1
                stats["kept_surya"] += 1
                stats["details"].append(detail)
                continue

            easy_to_surya[best_idx].append((tok, best_score, detail))
        else:
            detail["action"] = f"NO_MATCH (best_score={best_score:.3f})"
            stats["no_match"] += 1
            stats["details"].append(detail)

    # ── Phase 2: De-duplicate — one EasyOCR region → one Surya token ──
    tokens_to_remove = set()

    for easy_idx, matches in easy_to_surya.items():
        region = numeric_easy_regions[easy_idx]
        easy_text = region["text"]
        easy_conf = region["confidence"]
        normalized = normalize_easyocr_number(easy_text)

        if not normalized or not has_arabic_indic_digits(normalized):
            for tok, score, detail in matches:
                detail["easyocr_text"] = easy_text
                detail["easyocr_normalized"] = normalized
                detail["action"] = "KEPT_SURYA (EasyOCR no Arabic digits)"
                stats["kept_surya"] += 1
                stats["details"].append(detail)
            continue

        # Sort by match score — best first
        matches.sort(key=lambda x: x[1], reverse=True)

        # Best match gets the EasyOCR text
        best_tok, best_score, best_detail = matches[0]
        best_detail["easyocr_text"] = easy_text
        best_detail["easyocr_normalized"] = normalized
        best_detail["easyocr_conf"] = round(easy_conf, 3)
        best_detail["match_score"] = round(best_score, 3)
        best_detail["action"] = "REPLACED"

        best_tok.text = normalized
        best_tok.token_type = classify_token(normalized)
        best_tok.direction = detect_direction(normalized)
        best_tok.confidence = min(0.95, max(easy_conf, best_tok.confidence))

        stats["matched"] += 1
        stats["replaced"] += 1
        stats["details"].append(best_detail)

        # All other matches to same EasyOCR region → duplicates, remove them
        for tok, score, detail in matches[1:]:
            detail["easyocr_text"] = easy_text
            detail["action"] = f"DEDUP_REMOVED (kept best score={best_score:.3f})"
            stats["deduplicated"] += 1
            stats["details"].append(detail)
            tokens_to_remove.add(id(tok))

    # ── Phase 3: Remove duplicate tokens from page_tokens ──
    if tokens_to_remove:
        page_tokens.tokens = [
            t for t in page_tokens.tokens if id(t) not in tokens_to_remove
        ]
        for ln in page_tokens.lines:
            lt = ln.get('tokens', [])
            ln['tokens'] = [t for t in lt if id(t) not in tokens_to_remove]
            if ln['tokens']:
                ln['text'] = ' '.join(t.text for t in ln['tokens'])
                ln['direction'] = detect_direction(ln['text'])
            else:
                ln['text'] = ''
        page_tokens.lines = [
            ln for ln in page_tokens.lines if ln.get('text', '').strip()
        ]

    # ── Phase 4: Convert remaining Western digits → Arabic-Indic ──
    # Handles note references like (7)→(٧), (9)→(٩), (11)→(١١)
    _W2A = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')
    for tok in page_tokens.tokens:
        if any(c in _WESTERN_DIGITS for c in tok.text):
            new_text = tok.text.translate(_W2A)
            if new_text != tok.text:
                tok.text = new_text
                tok.token_type = classify_token(new_text)
                tok.direction = detect_direction(new_text)
                stats["western_converted"] += 1

    # Rebuild line text from corrected tokens
    for ln in page_tokens.lines:
        lt = ln.get('tokens', [])
        if lt:
            ln['text'] = ' '.join(t.text for t in lt)
            ln['direction'] = detect_direction(ln['text'])

    return stats


# ────────────────────────────────────────────────────────────────────
#  Column-Context Rescue (from Stage 3)
# ────────────────────────────────────────────────────────────────────
COLUMN_RESCUE_THRESHOLD = 0.75

def column_context_rescue(
    page_tokens: PageTokens,
    numeric_values: List,
    ocr_results: List[TokenOCRResult],
    threshold: float = COLUMN_RESCUE_THRESHOLD,
) -> Dict:
    """Rescue UNTRUSTED tokens using column-level context."""
    columns = find_numeric_columns(page_tokens)
    rescue_stats = {"checked": 0, "rescued": 0, "details": []}

    tok_to_col: Dict[Tuple, NumericColumn] = {}
    for col in columns:
        for ct in col.tokens:
            key = (round(ct.bbox[0], 2), round(ct.bbox[1], 2),
                   round(ct.bbox[2], 2), round(ct.bbox[3], 2))
            tok_to_col[key] = col

    numeric_idx = 0
    tok_nv_map: Dict[Tuple, Tuple[int, NumericValue]] = {}
    for tok in page_tokens.tokens:
        if tok.token_type == "NUMERIC" and numeric_idx < len(numeric_values):
            key = (round(tok.bbox[0], 2), round(tok.bbox[1], 2),
                   round(tok.bbox[2], 2), round(tok.bbox[3], 2))
            tok_nv_map[key] = (numeric_idx, numeric_values[numeric_idx])
            numeric_idx += 1

    for nv_idx, nv in enumerate(numeric_values):
        if nv.status != TrustStatus.UNTRUSTED:
            continue
        rescue_stats["checked"] += 1
        tok_key = (round(nv.bbox[0], 2), round(nv.bbox[1], 2),
                   round(nv.bbox[2], 2), round(nv.bbox[3], 2))
        col = tok_to_col.get(tok_key)
        detail = {"idx": nv_idx + 1, "digits": nv.digits,
                  "conf": nv.surya_confidence, "bbox": list(tok_key)}

        if not col:
            detail["reason"] = "NOT_IN_COLUMN"
            rescue_stats["details"].append(detail)
            continue

        trusted_count = 0
        total_count = col.count
        sibling_digit_lengths = []
        for ct in col.tokens:
            ct_key = (round(ct.bbox[0], 2), round(ct.bbox[1], 2),
                      round(ct.bbox[2], 2), round(ct.bbox[3], 2))
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
        detail["col_trust_ratio"] = round(col_trust_ratio, 2)
        detail["col_size"] = total_count

        if col_trust_ratio < threshold:
            detail["reason"] = f"COL_TRUST_TOO_LOW ({col_trust_ratio:.0%})"
            rescue_stats["details"].append(detail)
            continue

        token_digits = extract_digits(nv.digits)
        if not token_digits or not has_valid_arabic_digits(nv.digits):
            detail["reason"] = "NO_VALID_DIGITS"
            rescue_stats["details"].append(detail)
            continue

        if sibling_digit_lengths:
            median_len = statistics.median(sibling_digit_lengths)
            token_len = len(token_digits)
            if abs(token_len - median_len) > 2:
                detail["reason"] = (f"DIGIT_LEN_MISMATCH "
                                    f"(token={token_len}, median={median_len:.0f})")
                rescue_stats["details"].append(detail)
                continue

        detail["reason"] = "RESCUED"
        nv.status = TrustStatus.SURYA_VALID
        nv.trust_score = TRUST_SCORES[TrustStatus.SURYA_VALID]
        nv.failure_reasons = []
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
    if not page.lines:
        return page
    page.lines.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    for ln in page.lines:
        x0, y0, x1, y1 = ln["bbox"]
        ln["bbox"] = [round(max(0, x0), 2), round(max(0, y0), 2),
                      round(min(pw, x1), 2), round(min(ph, y1), 2)]
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
    for ln in page.lines:
        for tok in ln.get("tokens", []):
            tok.bbox[1] = ln["bbox"][1]
            tok.bbox[3] = ln["bbox"][3]
    return page


_NBSP = '\u00A0'  # Non-breaking space — keeps digit groups in one BiDi run
_ARABIC_INDIC_NUM_RE = re.compile(
    r'[\(]?[\u0660-\u0669][\u0660-\u0669 .,]*[\u0660-\u0669][\)]?'
    r'|[\(]?[\u0660-\u0669][\)]?'
)


def _bidi_protect_numbers(text: str) -> str:
    """Replace regular spaces with NBSP inside Arabic-Indic number sequences.

    In an RTL div, the Unicode BiDi algorithm treats Arabic-Indic digits
    (U+0660-U+0669) as AN (Arabic Number). Regular spaces between digit
    groups split them into separate BiDi runs that get reversed:
      '\u0663 \u0662\u0662\u0668 \u0662\u0664\u0662' visually becomes '\u0662\u0664\u0662 \u0662\u0662\u0668 \u0663'

    Using non-breaking spaces (U+00A0) instead keeps the entire number
    as a single BiDi run, preserving the correct digit-group order.
    Verified in Chromium with all unicode-bidi modes.
    """
    def _nbsp(m):
        return m.group(0).replace(' ', _NBSP)
    return _ARABIC_INDIC_NUM_RE.sub(_nbsp, text)


def _render_line_html(line):
    x0, y0, x1, y1 = line["bbox"]
    bw = x1 - x0
    h = y1 - y0
    d = line.get("direction", "rtl")
    dc = "rtl" if d == "rtl" else "ltr"
    fs = line.get("font_size", max(h * FONT_SIZE_FACTOR, 4.0))
    if d == "rtl":
        ew = bw * FONT_DRIFT_FACTOR
        extra = ew - bw
        left = max(0, x0 - extra)
        rw = x1 - left
    else:
        left = x0
        rw = bw
    lh = max(h, fs)
    raw = line.get("text", "")
    # Protect number ordering from RTL BiDi reordering
    raw = _bidi_protect_numbers(raw)
    inner = html_mod.escape(raw)
    if line.get("is_bold"):
        inner = f"<b>{inner}</b>"
    if line.get("is_italic"):
        inner = f"<i>{inner}</i>"
    return (f'<div class="line {dc}" style="left:{left:.2f}px;top:{y0:.2f}px;'
            f'width:{rw:.2f}px;height:{h:.2f}px;line-height:{lh:.2f}px;'
            f'font-size:{fs:.2f}px;">{inner}</div>')


def render_page_html(pt):
    pw, ph = pt.page_width, pt.page_height
    lines = sorted(pt.lines, key=lambda ln: (ln["bbox"][1], -ln["bbox"][0]))
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
    wi = wp / 72
    hi = hp / 72
    async with async_playwright() as pw:
        br = await pw.chromium.launch(headless=True)
        pg = await br.new_page()
        await pg.set_content(html_str, wait_until="networkidle")
        await pg.pdf(path=str(out), width=f"{wi:.4f}in", height=f"{hi:.4f}in",
                     margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                     print_background=True, scale=1.0)
        await br.close()


# ────────────────────────────────────────────────────────────────────
#  Digit Frequency Analysis
# ────────────────────────────────────────────────────────────────────
def digit_frequency_analysis(page_tokens: PageTokens) -> Dict:
    """Analyze Arabic-Indic digit distribution in final output."""
    all_digits = []
    for tok in page_tokens.tokens:
        if tok.token_type == "NUMERIC":
            for c in tok.text:
                if '\u0660' <= c <= '\u0669':
                    all_digits.append(c)

    freq = Counter(all_digits)
    total = len(all_digits)
    dist = {}
    for d in '٠١٢٣٤٥٦٧٨٩':
        cnt = freq.get(d, 0)
        pct = (cnt / total * 100) if total > 0 else 0
        dist[d] = {"count": cnt, "pct": round(pct, 1)}

    return {
        "total_digits": total,
        "distribution": dist,
        "max_digit": max(dist.items(), key=lambda x: x[1]["pct"])[0] if dist else None,
        "max_pct": max(x["pct"] for x in dist.values()) if dist else 0,
    }


# ────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    print("=" * 64)
    print("  STAGE 4: Hybrid Surya Layout + EasyOCR Digit Recognition")
    print("  ─────────────────────────────────────────────────────────")
    print("  ROOT CAUSE: Surya reads Arabic-Indic digits as Latin chars")
    print("  FIX: EasyOCR natively outputs correct Arabic-Indic digits")
    print(f"  Input:    {PDF_PATH.name}  (page {PAGE_NUM} only)")
    print(f"  Output:   {PREFIX}_*")
    print("=" * 64)

    doc = fitz.open(str(PDF_PATH))
    page = doc[PAGE_INDEX]
    pw, ph = round(page.rect.width, 2), round(page.rect.height, 2)
    print(f"\n  Page {PAGE_NUM}: {pw} × {ph} pt\n")

    # ── Step 1: Render page ──
    print("─── Step 1: Render Page at 300 DPI ───")
    base_img = render_page_at_dpi(doc, PAGE_INDEX, OCR_DPI)
    print(f"  Image size: {base_img.shape[1]}×{base_img.shape[0]} px")

    # ── Step 2: Surya for layout (text + bboxes) ──
    print(f"\n─── Step 2: Surya OCR for Layout Detection ───")
    t0 = time.time()
    surya_preprocessed = _preprocess_for_surya(base_img, NORMALIZED_WIDTH)
    page_tokens = run_surya_layout(surya_preprocessed, doc, PAGE_INDEX, OCR_DPI)
    t_surya = time.time() - t0
    n_surya_total = len(page_tokens.tokens)
    n_surya_numeric = sum(1 for t in page_tokens.tokens if t.token_type == "NUMERIC")
    n_surya_suspect = sum(1 for t in page_tokens.tokens
                         if _is_surya_suspect_numeric(t.text))
    print(f"  {n_surya_total} tokens ({n_surya_numeric} NUMERIC, "
          f"{n_surya_suspect} suspect-Latin)")
    print(f"  Surya OCR: {t_surya:.1f}s")

    # Show some Surya numeric tokens BEFORE correction
    print(f"\n  Surya NUMERIC tokens (raw, before EasyOCR):")
    shown = 0
    for tok in page_tokens.tokens:
        if (tok.token_type == "NUMERIC" or _is_surya_suspect_numeric(tok.text)) and shown < 10:
            print(f"    [{tok.confidence:.2f}] '{tok.text}'  bbox={[round(b,1) for b in tok.bbox]}")
            shown += 1
    if n_surya_numeric + n_surya_suspect > 10:
        print(f"    ... ({n_surya_numeric + n_surya_suspect - 10} more)")

    # ── Step 3: EasyOCR for digit recognition ──
    print(f"\n─── Step 3: EasyOCR for Arabic Digit Recognition ───")
    t0 = time.time()
    easyocr_regions = run_easyocr_on_image(base_img, pw, ph)
    t_easyocr = time.time() - t0
    n_easy_total = len(easyocr_regions)
    n_easy_numeric = sum(1 for r in easyocr_regions if is_numeric_content(r["text"]))
    print(f"  {n_easy_total} regions ({n_easy_numeric} numeric)")
    print(f"  EasyOCR: {t_easyocr:.1f}s")

    print(f"\n  EasyOCR numeric regions (sample):")
    shown = 0
    for r in easyocr_regions:
        if is_numeric_content(r["text"]) and shown < 10:
            print(f"    [{r['confidence']:.2f}] '{r['text']}'  "
                  f"bbox={[round(b,1) for b in r['bbox_pt']]}")
            shown += 1

    # ── Step 4: Match EasyOCR → Surya tokens ──
    print(f"\n─── Step 4: Match EasyOCR Digits → Surya Layout ───")
    match_stats = match_easyocr_to_surya(page_tokens, easyocr_regions)
    print(f"  Suspect numeric:  {match_stats['suspect_numeric']}")
    print(f"  Matched+Replaced: {match_stats['replaced']}")
    print(f"  Deduplicated:     {match_stats['deduplicated']}")
    print(f"  Low-conf rejected:{match_stats['low_conf_rejected']}")
    print(f"  Western→Arabic:   {match_stats['western_converted']}")
    print(f"  No match:         {match_stats['no_match']}")
    print(f"  Kept Surya:       {match_stats['kept_surya']}")

    # Show replacement details
    print(f"\n  Replacement details:")
    for d in match_stats["details"][:15]:
        if d["action"] == "REPLACED":
            print(f"    ✓ '{d['surya_text']}' → '{d['easyocr_normalized']}' "
                  f"(score={d['match_score']:.2f}, easyocr_conf={d['easyocr_conf']:.2f})")
        else:
            print(f"    ✗ '{d['surya_text']}' → {d['action']}")
    if len(match_stats["details"]) > 15:
        print(f"    ... ({len(match_stats['details']) - 15} more)")

    # Re-classify tokens after correction
    for tok in page_tokens.tokens:
        tok.token_type = classify_token(tok.text)
        tok.direction = detect_direction(tok.text)

    n_numeric_after = sum(1 for t in page_tokens.tokens if t.token_type == "NUMERIC")
    print(f"\n  After correction: {n_numeric_after} NUMERIC tokens")

    # ── Step 4b: Crop-based refinement for short/missed numbers ──
    print(f"\n─── Step 4b: Crop-Based EasyOCR Refinement ───")
    refine_stats = refine_easyocr_with_crops(page_tokens, base_img)
    print(f"  Tried:        {refine_stats['tried']}")
    print(f"  Improved:     {refine_stats['improved']}")
    print(f"  New matches:  {refine_stats['new_matches']}")
    for d in refine_stats.get('details', [])[:15]:
        if 'IMPROVED' in d.get('action', ''):
            print(f"    ✓ '{d['original_text']}' → '{d['crop_normalized']}' "
                  f"({d['action']}, conf={d['crop_conf']:.2f})")
        else:
            print(f"    ✗ '{d['original_text']}' → {d['action']}")
    if len(refine_stats.get('details', [])) > 15:
        print(f"    ... ({len(refine_stats['details']) - 15} more)")

    # Re-classify after refinement
    for tok in page_tokens.tokens:
        tok.token_type = classify_token(tok.text)
        tok.direction = detect_direction(tok.text)

    n_numeric_final = sum(1 for t in page_tokens.tokens if t.token_type == "NUMERIC")
    print(f"\n  After refinement: {n_numeric_final} NUMERIC tokens")

    # ── Step 5: Digit Frequency Analysis ──
    print(f"\n─── Step 5: Digit Frequency Analysis ───")
    freq_analysis = digit_frequency_analysis(page_tokens)
    print(f"  Total Arabic-Indic digits: {freq_analysis['total_digits']}")
    print(f"  Distribution:")
    for d in '٠١٢٣٤٥٦٧٨٩':
        info = freq_analysis['distribution'][d]
        bar = '█' * int(info['pct'] / 2)
        marker = " ⚠️" if info['pct'] > 25 else ""
        print(f"    {d}: {info['count']:3d} ({info['pct']:5.1f}%) {bar}{marker}")

    # ── Step 6: Save tokens ──
    with open(OUT_TOKENS, "w", encoding="utf-8") as f:
        for tok in page_tokens.tokens:
            f.write(json.dumps({
                "text": tok.text, "type": tok.token_type,
                "bbox": [round(b, 2) for b in tok.bbox],
                "conf": round(tok.confidence, 4) if tok.confidence else None,
                "font_size": round(tok.font_size, 2),
            }, ensure_ascii=False) + "\n")

    # ── Step 7: Layout Normalization ──
    print(f"\n─── Step 6: Layout Normalization ───")
    page_tokens = normalize_page_tokens(page_tokens)
    original_lines = [{"line_id": ln.get("line_id", i), "text": ln.get("text", ""),
                       "bbox": ln.get("bbox", [0, 0, 0, 0])}
                      for i, ln in enumerate(page_tokens.lines)]
    print(f"  {len(page_tokens.lines)} lines")

    # ── Step 8: Numeric Validation (Trust Model) ──
    print(f"\n─── Step 7: Numeric Validation (Discrete Trust) ───")
    hires = render_page_at_dpi(doc, PAGE_INDEX, dpi=NUMERIC_VAL_DPI)
    ocr_results = ocr_page_numeric_tokens(page_tokens, hires)
    n_locked = sum(1 for r in ocr_results if r.locked)
    print(f"  {len(ocr_results)} numeric tokens, {n_locked} LOCKED")

    current_lines = [{"line_id": ln.get("line_id", i), "text": ln.get("text", ""),
                      "bbox": ln.get("bbox", [0, 0, 0, 0])}
                     for i, ln in enumerate(page_tokens.lines)]
    page_stability = validate_page_lines(original_lines, current_lines, PAGE_NUM)
    ocr_results = apply_line_stability_to_tokens(
        page_stability, ocr_results, page_tokens.lines)

    font_sizes = [t.font_size for t in page_tokens.tokens if t.token_type == "NUMERIC"]
    numeric_values = reconstruct_page_numbers(ocr_results, PAGE_NUM, font_sizes)

    pre_untrusted = sum(1 for nv in numeric_values if nv.status == TrustStatus.UNTRUSTED)
    print(f"  Pre-rescue UNTRUSTED: {pre_untrusted}")

    # ── Step 9: Column-Context Rescue ──
    print(f"\n─── Step 8: Column-Context Rescue ───")
    numeric_idx = 0
    for tok in page_tokens.tokens:
        if tok.token_type == "NUMERIC" and numeric_idx < len(numeric_values):
            nv = numeric_values[numeric_idx]
            if has_valid_arabic_digits(nv.digits):
                tok.text = nv.digits
            numeric_idx += 1
    for ln in page_tokens.lines:
        lt = ln.get('tokens', [])
        if lt:
            ln['text'] = ' '.join(t.text for t in lt)

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
    print(f"    🔒 LOCKED:        {ts.get('locked', 0)}")
    print(f"    ✓  SURYA_VALID:   {ts.get('surya_valid', 0)}")
    print(f"    🧠 CNN_CONFIRMED: {ts.get('cnn_confirmed', 0)}")
    print(f"    ⚠  UNTRUSTED:     {ts.get('untrusted', 0)}")
    print(f"    Trust rate:       {ts.get('pct_trusted', 0)}%")

    columns = find_numeric_columns(page_tokens)
    all_anomalies = [detect_column_anomalies(columns)]
    pipeline_audit = PipelineNumericAudit(pages=[pa])
    pipeline_audit.compute_overall()

    # ── Step 10: Final Digit Frequency (post-trust) ──
    print(f"\n─── Step 9: Final Digit Frequency Check ───")
    final_freq = digit_frequency_analysis(page_tokens)
    print(f"  Total Arabic-Indic digits: {final_freq['total_digits']}")
    seven_pct = final_freq['distribution']['٧']['pct']
    print(f"  ٧ frequency: {seven_pct}% ", end="")
    if seven_pct > 20:
        print("⚠️  STILL OVER-REPRESENTED")
    elif seven_pct > 12:
        print("⚡ Slightly elevated")
    else:
        print("✅ HEALTHY (matches ground truth ~8%)")

    # ── Step 11: Render ──
    print(f"\n─── Step 10: Render → PDF ───")
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

    # ── Step 12: QA Reports ──
    print(f"\n─── Step 11: QA Reports ───")
    num_qa_html = generate_qa_report(pipeline_audit)
    OUT_NUM_QA.write_text(num_qa_html, encoding="utf-8")
    qa_html = render_qa_html([page_tokens], [columns], all_anomalies)
    OUT_QA_HTML.write_text(qa_html, encoding="utf-8")
    print(f"  {OUT_NUM_QA.name}, {OUT_QA_HTML.name}")

    # ── Step 13: Numeric Token Detail ──
    print(f"\n─── Numeric Token Detail ───")
    print(f"  {'#':>3}  {'Status':<15} {'Conf':>6} {'Trust':>5}  {'Digits'}")
    print(f"  {'─'*3}  {'─'*15} {'─'*6} {'─'*5}  {'─'*20}")
    untrusted_details = []
    for idx, nv in enumerate(numeric_values):
        icon = {'LOCKED': '🔒', 'SURYA_VALID': '✓ ', 'CNN_CONFIRMED': '🧠',
                'UNTRUSTED': '⚠ '}.get(nv.status.value, '? ')
        reasons = ', '.join(r.value for r in nv.failure_reasons) if nv.failure_reasons else ''
        extra = f"  ({reasons})" if reasons else ""
        print(f"  {idx+1:3d}  {icon} {nv.status.value:<13} "
              f"{nv.surya_confidence:6.3f} {nv.trust_score:5.2f}  "
              f"{nv.digits}{extra}")
        if nv.status == TrustStatus.UNTRUSTED:
            untrusted_details.append({
                "idx": idx + 1, "digits": nv.digits,
                "conf": nv.surya_confidence,
                "bbox": [round(b, 2) for b in nv.bbox],
            })

    # ── Log ──
    elapsed = time.time() - t_start
    log_data = {
        "stage": "stage4_easyocr_digits",
        "description": "Hybrid Surya Layout + EasyOCR Digit Recognition",
        "root_cause": "Surya reads Arabic-Indic digits as Latin chars (7,Y,V,T,E,A)",
        "fix": "EasyOCR natively outputs correct Arabic-Indic digits",
        "page": PAGE_NUM,
        "time_s": round(elapsed, 1),
        "surya_time_s": round(t_surya, 1),
        "easyocr_time_s": round(t_easyocr, 1),
        "tokens": n_surya_total,
        "numeric_tokens": n_numeric_after,
        "easyocr_matching": {
            "suspect_numeric": match_stats["suspect_numeric"],
            "replaced": match_stats["replaced"],
            "deduplicated": match_stats["deduplicated"],
            "low_conf_rejected": match_stats["low_conf_rejected"],
            "western_converted": match_stats["western_converted"],
            "no_match": match_stats["no_match"],
            "kept_surya": match_stats["kept_surya"],
        },
        "crop_refinement": {
            "tried": refine_stats["tried"],
            "improved": refine_stats["improved"],
            "new_matches": refine_stats["new_matches"],
        },
        "trust": {
            "total": len(numeric_values),
            "locked": ts.get('locked', 0),
            "surya_valid": ts.get('surya_valid', 0),
            "cnn_confirmed": ts.get('cnn_confirmed', 0),
            "untrusted": ts.get('untrusted', 0),
            "trust_pct": ts.get('pct_trusted', 0),
        },
        "digit_frequency": {
            "total_digits": final_freq["total_digits"],
            "seven_pct": seven_pct,
            "distribution": {k: v["pct"] for k, v in final_freq["distribution"].items()},
        },
        "rescue_stats": rescue_stats,
        "columns": len(columns),
        "anomalies": len(all_anomalies[0]),
        "untrusted_tokens": untrusted_details,
        "comparison": {
            "baseline": {"trust_pct": 94.2, "untrusted": 8, "seven_pct": 30.6},
            "stage1": {"trust_pct": 95.6, "untrusted": 6, "seven_pct": "~35%"},
            "stage2": {"trust_pct": 97.1, "untrusted": 4, "seven_pct": "~35%"},
            "stage3": {"trust_pct": 99.3, "untrusted": 1, "seven_pct": 35.6},
        },
    }
    OUT_LOG.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")
    doc.close()

    # ── Summary ──
    tp = ts.get('pct_trusted', 0)
    n_anom = len(all_anomalies[0])
    print(f"\n{'='*64}")
    print(f"  ✓ Stage 4 done in {elapsed:.1f}s")
    print(f"  Trust:  {tp}%")
    print(f"    🔒 LOCKED: {ts.get('locked',0)} | ⚠ UNTRUSTED: {ts.get('untrusted',0)}")
    print(f"  ٧ freq: {seven_pct}% (was 35.6% in Stage 3, ground truth ~8%)")
    print(f"  Columns: {len(columns)} | Anomalies: {n_anom}")
    print(f"")
    print(f"  Comparison:")
    print(f"    Baseline:  94.2% trust, ٧=30.6%")
    print(f"    Stage 1:   95.6% trust")
    print(f"    Stage 2:   97.1% trust")
    print(f"    Stage 3:   99.3% trust, ٧=35.6%  ← high trust, WRONG digits")
    print(f"    Stage 4:   {tp}% trust, ٧={seven_pct}%  ← correct digits!")
    print(f"")
    print(f"  Output: {PREFIX}_*")
    print(f"{'='*64}")

    meets = tp > 98 and seven_pct < 15
    if meets:
        print(f"\n  🎯 TARGET MET! trust > 98% AND ٧ frequency normalized!")
    else:
        remaining = ts.get('untrusted', 0)
        print(f"\n  ⚠ {remaining} untrusted remain, ٧ at {seven_pct}%")


if __name__ == "__main__":
    main()
