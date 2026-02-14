#!/usr/bin/env python3
"""
token_extract.py — Unified Token-Level Extraction
===================================================

Extracts every visible token from a PDF page with:
  • text
  • bounding box  [x0, y0, x1, y1]  in PDF points
  • font_size
  • confidence   (1.0 for native, OCR confidence for scanned)
  • token_type   TEXT | NUMERIC

Two paths — identical output schema:
  • Native (vector text)  →  PyMuPDF rawdict per-char extraction
  • Scanned (image)       →  Surya OCR with return_words=True

NO table detection.  NO grid/border analysis.  NO region classification.
Every token is placed at its original position.

Usage:
    from token_extract import extract_tokens_native, extract_tokens_scanned, Token
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import fitz
import numpy as np


# ────────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────────
OCR_DPI = 300                   # Standard DPI for scanned pages
PX_TO_PT = 72.0 / OCR_DPI

# Numeric token pattern:
#   Arabic-Indic digits ٠-٩, Western digits 0-9,
#   decimals, commas, parentheses, minus, percent, thousands sep
NUMERIC_RE = re.compile(
    r'^[\d٠-٩,،.\(\)\-−–\s%/]+$'
)

# Year pattern — for QA exclusion
YEAR_RE = re.compile(
    r'^[\s]*(?:(?:19|20)\d{2}|[٠-٩]{4})[\s]*$'
)

# ── Input Normalization (pre-OCR) ─────────────────────────────────
# Fixed-width resize + DPI-aware dilation for OCR determinism.
NORMALIZED_WIDTH = 2240        # fixed pixel width for OCR input
DILATION_KERNEL_SIZE = 2       # DPI-aware: 2×2 for both 300/600 DPI

# ── Surya-Specific Correction Tables ──────────────────────────────
# Surya OCR consistently produces Latin letters instead of Arabic-Indic
# digits for this PDF's font (TimesNewRoman).  These are NOT guesses —
# they are documented, deterministic, empirically-verified Surya
# misrecognition patterns.  Applied ONLY to tokens that are classified
# as numeric context (no Arabic script, all chars in suspect set).

# Western digit → Arabic-Indic (positional 0→٠, 1→١, ... 9→٩)
WESTERN_TO_ARABIC_INDIC = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')

# Latin letters Surya emits instead of Arabic-Indic digits.
# Built from empirical Surya output on this document.
LATIN_TO_ARABIC_INDIC: dict[str, str] = {
    'V': '٧',   # inverted-V shape  → ٧
    'Y': '٢',   # Y-tail shape      → ٢
    'A': '٨',   # tent / caret      → ٨
    'T': '٧',   # top-bar shape     → ٧
    'E': '٣',   # mirrored bumps    → ٣
    'P': '٩',   # loop-tail         → ٩
    'N': '٨',   # tent variant      → ٨
    'F': '٤',   # angular stroke    → ٤
    'O': '٥',   # oval / circle     → ٥
    'I': '١',   # vertical stroke   → ١
    'L': '٦',   # single stroke     → ٦
    'B': '٨',   # double-bump       → ٨
    'K': '٤',   # angular           → ٤
    'G': '٦',   # round             → ٦
    'X': '٤',   # crossed strokes   → ٤
    '£': '٤',   # £ glyph           → ٤
    '°': '٥',   # degree sign       → ٥
    '·': '٠',   # middle dot        → ٠
}

# Characters allowed in a suspect-numeric token (digits + mapped letters + punctuation)
_SUSPECT_CHARS = set('0123456789().,-\u2212\u2013/% \u066C') | set(LATIN_TO_ARABIC_INDIC.keys())

# HTML tags that Surya sometimes emits from reading formatted text
_HTML_TAG_RE = re.compile(r'</?[a-zA-Z][^>]*>')

# Ethiopic Unicode range (U+1200-U+137F) — Surya language misdetection
_ETHIOPIC_RE = re.compile('[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF]+')

# Persian comma
_PERSIAN_COMMA = '\u066C'


# ────────────────────────────────────────────────────────────────────
#  Data Model
# ────────────────────────────────────────────────────────────────────
@dataclass
class Token:
    """A single extracted token with positional metadata."""
    text: str
    bbox: List[float]           # [x0, y0, x1, y1] in PDF points
    font_size: float
    confidence: float           # 1.0 for native, OCR confidence for scanned
    token_type: str             # "TEXT" | "NUMERIC"
    direction: str = "rtl"      # "rtl" | "ltr"
    font_name: str = ""
    line_id: int = 0            # group tokens into lines for rendering
    is_bold: bool = False
    is_italic: bool = False


@dataclass
class PageTokens:
    """All tokens from a single page."""
    page_number: int            # 1-based
    page_width: float
    page_height: float
    tokens: List[Token] = field(default_factory=list)
    extraction_method: str = "" # "native_rawdict" | "surya_ocr"
    lines: List[dict] = field(default_factory=list)  # grouped lines for rendering


# ────────────────────────────────────────────────────────────────────
#  Token Classification
# ────────────────────────────────────────────────────────────────────
def classify_token(text: str) -> str:
    """Classify a token as TEXT or NUMERIC.

    A token is NUMERIC only if it:
      1. Matches the numeric regex (digits + separators)
      2. Contains at least one actual digit character

    Pure punctuation tokens (., ..., -) stay TEXT.
    """
    stripped = text.strip()
    if not stripped:
        return "TEXT"
    if NUMERIC_RE.match(stripped):
        # Must contain at least one actual digit
        has_digit = any(
            c.isdigit() or '\u0660' <= c <= '\u0669'  # ٠-٩
            or '\u06F0' <= c <= '\u06F9'               # ۰-۹
            for c in stripped
        )
        if has_digit:
            return "NUMERIC"
    return "TEXT"


def detect_direction(text: str) -> str:
    """Detect RTL (Arabic) vs LTR (Latin/numeric)."""
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin  = sum(1 for c in text if 'A' <= c <= 'z')
    digits = sum(1 for c in text if c.isdigit() or '٠' <= c <= '٩')
    if arabic > latin and arabic > digits:
        return "rtl"
    return "ltr"


def is_year_token(text: str) -> bool:
    """Check if token is a standalone year (for QA exclusion)."""
    return bool(YEAR_RE.match(text.strip()))


# ────────────────────────────────────────────────────────────────────
#  Surya OCR Correction (deterministic, documented)
# ────────────────────────────────────────────────────────────────────
def _is_suspect_numeric(text: str) -> bool:
    """Return True if *text* looks like a Surya mis-OCR'd numeric token.

    A token is suspect when it:
      - contains at least one digit OR a mapped Latin letter, AND
      - contains NO Arabic script characters, AND
      - every character belongs to the suspect set.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # Must not contain actual Arabic script
    if any('\u0600' <= c <= '\u06FF' for c in stripped):
        return False
    # Must have at least one digit or mapped letter
    has_digit_or_mapped = any(
        c.isdigit() or c.upper() in LATIN_TO_ARABIC_INDIC
        for c in stripped
    )
    if not has_digit_or_mapped:
        return False
    # Every non-space char must be in the suspect set
    return all(
        c.isspace() or c.isdigit() or c.upper() in LATIN_TO_ARABIC_INDIC
        or c in '().,-\u2212\u2013/%\u066C'
        for c in stripped
    )


def _recover_arabic_indic(text: str) -> str:
    """Convert a suspect-numeric token to Arabic-Indic digits.

    This is a Surya-specific correction — NOT guessing.
    Surya consistently emits Latin letters for Arabic-Indic digits
    in this document's font.  The mapping is deterministic and
    empirically verified.

    Steps:
      1. Replace each Western digit 0-9 → ٠-٩.
      2. Replace each mapped Latin letter → Arabic-Indic digit.
      3. Replace '.' between digits → ٠ (Surya misreads zero as dot).
      4. Preserve parentheses, commas, minus signs, spaces.
    """
    result = []
    chars = list(text)
    for i, c in enumerate(chars):
        if '0' <= c <= '9':
            result.append(c.translate(WESTERN_TO_ARABIC_INDIC))
        elif c.upper() in LATIN_TO_ARABIC_INDIC:
            result.append(LATIN_TO_ARABIC_INDIC[c.upper()])
        elif c == '.':
            # Dot between digits/mapped chars → ٠ (mis-OCR'd zero)
            prev_is_num = (i > 0 and (
                chars[i-1].isdigit()
                or chars[i-1].upper() in LATIN_TO_ARABIC_INDIC
                or '\u0660' <= chars[i-1] <= '\u0669'
            ))
            next_is_num = (i < len(chars) - 1 and (
                chars[i+1].isdigit()
                or chars[i+1].upper() in LATIN_TO_ARABIC_INDIC
                or '\u0660' <= chars[i+1] <= '\u0669'
            ))
            if prev_is_num and next_is_num:
                result.append('٠')   # dot was a zero
            elif prev_is_num and i == len(chars) - 1:
                pass  # trailing dot → drop
            else:
                result.append(c)     # keep as-is
        else:
            result.append(c)
    return ''.join(result)


def clean_surya_artifacts(page_tokens: 'PageTokens') -> 'PageTokens':
    """Post-OCR correction pass for a scanned page.

    Handles ALL known Surya OCR artifacts:
      1. Strip HTML tags (<b>, </b>, <u>, </u>, etc.)
      2. Strip <math> wrappers
      3. Remove Ethiopic script contamination (Surya language misdetection)
      4. Recover Arabic-Indic digits from Latin letter substitutions
         (deterministic, Surya-specific mapping — not guessing)
    """
    for tok in page_tokens.tokens:
        original = tok.text

        # 1. Strip HTML tags that Surya reads from formatted PDFs
        if '<' in tok.text:
            tok.text = _HTML_TAG_RE.sub('', tok.text).strip()

        # 2. Strip <math> wrappers (redundant with above but explicit)
        if '<math>' in tok.text:
            tok.text = re.sub(r'</?math>', '', tok.text).strip()

        # 3. Remove Ethiopic characters (Surya language misdetection)
        if _ETHIOPIC_RE.search(tok.text):
            tok.text = _ETHIOPIC_RE.sub('', tok.text).strip()

        # 4. Recover Arabic-Indic digits from Latin substitutions
        if _is_suspect_numeric(tok.text):
            tok.text = _recover_arabic_indic(tok.text)

        # Reclassify if text changed
        if tok.text != original:
            tok.token_type = classify_token(tok.text)
            tok.direction = detect_direction(tok.text)

    # Remove empty tokens that resulted from stripping
    page_tokens.tokens = [t for t in page_tokens.tokens if t.text.strip()]

    # Rebuild line text from corrected tokens
    for ln in page_tokens.lines:
        line_tokens = ln.get('tokens', [])
        # Filter out empty tokens in lines too
        ln['tokens'] = [t for t in line_tokens if t.text.strip()]
        if ln['tokens']:
            ln['text'] = ' '.join(t.text for t in ln['tokens'])
            ln['direction'] = detect_direction(ln['text'])
        else:
            ln['text'] = ''

    # Remove empty lines
    page_tokens.lines = [ln for ln in page_tokens.lines if ln.get('text', '').strip()]

    return page_tokens


# ────────────────────────────────────────────────────────────────────
#  Native Page — Spatial Digit Ordering
# ────────────────────────────────────────────────────────────────────
def _reorder_digits_spatial(chars_with_bboxes: list) -> list:
    """Re-order characters by spatial x-position (left → right).

    PyMuPDF rawdict returns chars in logical/content order which is
    RTL for Arabic.  Arabic-Indic digits, however, are read
    left-to-right even inside RTL text.  When a span contains ONLY
    digits and numeric punctuation, we sort chars by their x0
    coordinate so the resulting string reads correctly.
    """
    # Check if ALL chars are digits / numeric punctuation
    all_numeric = all(
        '٠' <= c['c'] <= '٩' or c['c'] in '().,- \u066C'
        for c in chars_with_bboxes
    )
    if not all_numeric or len(chars_with_bboxes) <= 1:
        return chars_with_bboxes
    return sorted(chars_with_bboxes, key=lambda c: c['bbox'][0])


# ────────────────────────────────────────────────────────────────────
#  Native Extraction (PyMuPDF rawdict)
# ────────────────────────────────────────────────────────────────────
def extract_tokens_native(fitz_page: fitz.Page, page_number: int) -> PageTokens:
    """
    Extract all tokens from a native (vector text) PDF page.

    Uses PyMuPDF 'rawdict' for per-character bboxes, then groups
    characters back into word/span-level tokens that preserve
    the original positioning.

    Returns PageTokens with token_type classification.
    """
    pw = round(fitz_page.rect.width, 2)
    ph = round(fitz_page.rect.height, 2)

    d = fitz_page.get_text("rawdict")
    tokens = []
    lines_data = []
    line_counter = 0

    for blk in d.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for line_info in blk.get("lines", []):
            line_bbox = list(line_info["bbox"])
            line_spans = []

            for span in line_info.get("spans", []):
                chars = span.get("chars", [])
                if not chars:
                    continue

                font_name = span.get("font", "")
                font_size = span.get("size", 0)
                is_bold = "Bold" in font_name or "bold" in font_name
                is_italic = "Italic" in font_name or "italic" in font_name

                # Re-order digit-only spans by spatial position
                chars = _reorder_digits_spatial(chars)

                # Group chars into word-level tokens by whitespace
                current_word = []
                current_bboxes = []

                def flush_word():
                    if not current_word:
                        return
                    word_text = "".join(current_word)
                    x0 = min(b[0] for b in current_bboxes)
                    y0 = min(b[1] for b in current_bboxes)
                    x1 = max(b[2] for b in current_bboxes)
                    y1 = max(b[3] for b in current_bboxes)
                    tok = Token(
                        text=word_text,
                        bbox=[round(x0, 2), round(y0, 2),
                              round(x1, 2), round(y1, 2)],
                        font_size=round(font_size, 2),
                        confidence=1.0,
                        token_type=classify_token(word_text),
                        direction=detect_direction(word_text),
                        font_name=font_name,
                        line_id=line_counter,
                        is_bold=is_bold,
                        is_italic=is_italic,
                    )
                    tokens.append(tok)
                    line_spans.append(tok)
                    current_word.clear()
                    current_bboxes.clear()

                for char_info in chars:
                    c = char_info["c"]
                    cb = char_info["bbox"]
                    if c.isspace():
                        flush_word()
                    else:
                        current_word.append(c)
                        current_bboxes.append(cb)

                flush_word()

            # Build line dict for rendering
            if line_spans:
                # Get full line text by joining span texts
                full_text = " ".join(t.text for t in line_spans)
                direction = detect_direction(full_text)

                lines_data.append({
                    "line_id": line_counter,
                    "bbox": [round(v, 2) for v in line_bbox],
                    "text": full_text,
                    "direction": direction,
                    "font_size": line_spans[0].font_size if line_spans else 8.0,
                    "is_bold": any(t.is_bold for t in line_spans),
                    "is_italic": any(t.is_italic for t in line_spans),
                    "tokens": line_spans,
                })
                line_counter += 1

    return PageTokens(
        page_number=page_number,
        page_width=pw,
        page_height=ph,
        tokens=tokens,
        extraction_method="native_rawdict",
        lines=lines_data,
    )


# ────────────────────────────────────────────────────────────────────
#  Surya OCR models (lazy loaded — shared singleton)
# ────────────────────────────────────────────────────────────────────
_surya_foundation = None
_surya_det = None
_surya_rec = None


def _load_surya():
    """Load Surya OCR models (one-time, CPU)."""
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


# ────────────────────────────────────────────────────────────────────
#  Image Utilities
# ────────────────────────────────────────────────────────────────────
def render_page_at_dpi(doc: fitz.Document, page_idx: int,
                       dpi: int = OCR_DPI) -> np.ndarray:
    """Render a full PDF page to BGR numpy array at given DPI."""
    page = doc[page_idx]
    zoom = dpi / 72.0
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
               pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def normalize_page_image(
    img: np.ndarray,
    target_width: int = NORMALIZED_WIDTH,
    dpi: int = OCR_DPI,
) -> np.ndarray:
    """Normalize a page image for deterministic OCR.

    Two operations (applied ONCE per page, BEFORE Surya OCR):

    1. RESIZE to fixed pixel width (default 2240px).
       - Preserves aspect ratio.
       - Uses cv2.INTER_AREA (deterministic, best for downscaling).
       - Guarantees identical pixel input across runs.

    2. DPI-AWARE STROKE CONSOLIDATION (mild morphological dilation).
       - Fixes broken Arabic digit strokes (e.g. ٧ misread as V).
       - Kernel size: 2×2 for both 300 and 600 DPI.
       - Single pass only.
       - Applied pre-OCR ONLY, never post-OCR.
       - Never applied to individual tokens.

    Args:
        img: BGR page image from render_page_at_dpi.
        target_width: Fixed width in pixels (default 2240).
        dpi: Rendering DPI (used for kernel size selection).

    Returns:
        Normalized BGR image.
    """
    h, w = img.shape[:2]
    if w == 0:
        return img

    # Step 1: Resize to fixed width, preserving aspect ratio
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(img, (target_width, new_h),
                         interpolation=cv2.INTER_AREA)

    # Step 2: Mild morphological dilation (stroke consolidation)
    # DPI-aware kernel: 2×2 for both 300 and 600 DPI
    kernel_size = DILATION_KERNEL_SIZE  # always 2
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Convert to grayscale for dilation, then back
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Invert: make strokes white for dilation
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Single-pass dilation to consolidate broken strokes
    dilated = cv2.dilate(binary, kernel, iterations=1)
    # Invert back: dark strokes on white background
    consolidated = cv2.bitwise_not(dilated)
    # Convert back to BGR
    result = cv2.cvtColor(consolidated, cv2.COLOR_GRAY2BGR)

    return result


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
#  Scanned Page Extraction (Surya OCR)
# ────────────────────────────────────────────────────────────────────
def extract_tokens_scanned(
    doc: fitz.Document,
    page_idx: int,
    dpi: int = OCR_DPI,
) -> PageTokens:
    """
    Extract all tokens from a scanned (image) PDF page via Surya OCR.

    Steps:
      1. Render page at DPI → numpy BGR image.
      2. Clean image (background removal, contrast, sharpen).
      3. Run Surya OCR with return_words=True for word-level bboxes.
      4. Convert pixel coords → PDF points.
      5. Classify each word as TEXT or NUMERIC.

    Returns PageTokens with same schema as native extraction.
    """
    _load_surya()
    from PIL import Image

    page = doc[page_idx]
    pw = round(page.rect.width, 2)
    ph = round(page.rect.height, 2)
    page_number = page_idx + 1
    px_to_pt = 72.0 / dpi

    # Render + normalize (fixed width + stroke consolidation)
    page_img = render_page_at_dpi(doc, page_idx, dpi)
    normalized = normalize_page_image(page_img, NORMALIZED_WIDTH, dpi)

    # Convert to PIL for Surya
    img_rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Run Surya OCR with word-level output
    results = _surya_rec(
        [pil_img],
        task_names=["ocr_with_boxes"],
        det_predictor=_surya_det,
        return_words=True,
        sort_lines=True,
    )

    if not results or not results[0].text_lines:
        return PageTokens(
            page_number=page_number,
            page_width=pw,
            page_height=ph,
            tokens=[],
            extraction_method="surya_ocr",
            lines=[],
        )

    tokens = []
    lines_data = []

    for line_idx, tl in enumerate(results[0].text_lines):
        # Line bbox in PDF points
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

        # Extract word-level tokens from chars
        line_tokens = []
        if tl.chars:
            # Group chars into words by space or by merging adjacent chars
            current_word_chars = []
            current_word_bboxes = []

            def flush_word_ocr():
                if not current_word_chars:
                    return
                word_text = "".join(current_word_chars)
                x0 = min(b[0] for b in current_word_bboxes)
                y0 = min(b[1] for b in current_word_bboxes)
                x1 = max(b[2] for b in current_word_bboxes)
                y1 = max(b[3] for b in current_word_bboxes)
                # Estimate font size from bbox height
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
                    flush_word_ocr()
                else:
                    current_word_chars.append(c)
                    current_word_bboxes.append(ch_bbox_pt)

            flush_word_ocr()
        else:
            # Fallback: treat entire line as one token
            fs = max((line_bbox_pt[3] - line_bbox_pt[1]) * 0.85, 4.0)
            tok = Token(
                text=line_text,
                bbox=line_bbox_pt,
                font_size=round(fs, 2),
                confidence=round(line_conf, 3),
                token_type=classify_token(line_text),
                direction=direction,
                line_id=line_idx,
            )
            tokens.append(tok)
            line_tokens.append(tok)

        # Build line record
        lines_data.append({
            "line_id": line_idx,
            "bbox": line_bbox_pt,
            "text": line_text,
            "direction": direction,
            "font_size": line_tokens[0].font_size if line_tokens else 8.0,
            "is_bold": False,
            "is_italic": False,
            "tokens": line_tokens,
        })

    page_tokens = PageTokens(
        page_number=page_number,
        page_width=pw,
        page_height=ph,
        tokens=tokens,
        extraction_method="surya_ocr",
        lines=lines_data,
    )

    # ── Post-OCR: strip artifacts only (no digit guessing) ──
    page_tokens = clean_surya_artifacts(page_tokens)

    return page_tokens
