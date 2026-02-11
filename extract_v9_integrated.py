#!/usr/bin/env python3
"""
extract_v9_integrated.py — Integrated Arabic Financial PDF Extraction
=====================================================================
Addresses five critical concerns for Arabic financial document processing:

1. MODEL SELECTION
   - PaddleOCR v3 (PP-OCRv5) with built-in Differentiable Binarization (DB)
     text detector — trained on multilingual data including Arabic script.
   - Tesseract LSTM (--oem 3) as fallback with Arabic language pack.
   - PyMuPDF native vector-text extraction for digital pages (zero OCR needed).

2. LAYOUT ANALYSIS
   - Consumes table bounding boxes from detect_tables_cv.py's spatial_data.json
     (3-layer hybrid: morphological + PyMuPDF band analysis + OpenCV scan).
   - Every page is partitioned into TABLE regions (from spatial_data.json) and
     TEXT regions (everything outside table boxes).
   - No reliance on Detectron2 for table detection (PubLayNet is trained on
     English academic papers — poor recall on Arabic borderless financial tables).

3. SPATIAL CORRELATION
   - Maps paragraph text-line bounding boxes (x, y) to the nearest table row's
     Y-coordinate within ±15 pt.
   - Produces TABLE_ANNOTATION units that carry a back-reference to the
     specific table row they annotate (footnotes, notes, sub-labels).

4. PRE-PROCESSING  (for scanned / image-only pages)
   - CLAHE contrast equalisation (clipLimit=2.0, 8×8 grid)
   - Adaptive Gaussian binarisation (blockSize=31, C=10)
   - Morphological closing (2×2) to repair broken Arabic ligatures
   - Deskew via cv2.minAreaRect angle detection
   - DPI normalisation: upscale low-res scans to ≥300 DPI (INTER_CUBIC)
   - PaddleOCR receives the *original* image (its DB detector is trained
     end-to-end on noisy inputs); Tesseract receives the pre-processed binary.

5. RTL / BiDi LOGIC
   - Unicode Bidirectional Algorithm via python-bidi.
   - Span ordering within each visual row: RTL (rightmost first).
   - Column ordering in structured tables: RTL (col 0 = rightmost = labels,
     col 1+ = numeric value columns going left).
   - Output: logical order with U+200F Right-to-Left Mark on Arabic lines
     for correct rendering in any viewer.
   - Mixed Arabic text + LTR numbers handled structurally: label column
     is separated from value columns via x-coordinate clustering.

Pipeline
--------
  0. Load spatial_data.json (table bounding boxes from detect_tables_cv.py)
  1. For each page:
     a) Classify: scanned (OCR) vs native (vector text)
     b) Collect all text spans (PyMuPDF native or PaddleOCR + Tesseract)
     c) Partition spans into TABLE regions and TEXT regions using bbox overlap
     d) TABLE spans → split merged → column detection → structured rows
     e) TEXT spans → group into visual lines → BiDi-aware rendering
     f) Spatial correlation: link text lines to nearest table rows (±15 pt)
  2. Output:
     - extracted_v9.txt   — human-readable structured text
     - extracted_v9.jsonl  — one JSON object per Information Unit (for RAG)

Usage
-----
    # Step 1: run table detection (if not already done)
    python detect_tables_cv.py

    # Step 2: run integrated extraction
    python extract_v9_integrated.py
"""

import os
import re
import sys
import io
import json
import warnings

warnings.filterwarnings("ignore")

# ── Paddle / oneDNN environment (must precede any paddle import) ─────
os.environ['FLAGS_enable_onednn'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['GLOG_minloglevel'] = '3'
os.environ['FLAGS_logtostderr'] = '0'

import fitz            # PyMuPDF
import numpy as np
import cv2
from PIL import Image

# BiDi — Unicode Bidirectional Algorithm
try:
    from bidi.algorithm import get_display as _bidi_get_display
    HAS_BIDI = True
except ImportError:
    HAS_BIDI = False

# ====================================================================
#  Configuration
# ====================================================================
PDF_FILE       = 'el-bankalahly .pdf'
SPATIAL_JSON   = 'table_extraction/spatial_data.json'
OUTPUT_TXT     = 'extracted_v9.txt'
OUTPUT_JSONL   = 'extracted_v9.jsonl'

OCR_ZOOM              = 3       # zoom for region OCR on native pages
SCAN_OCR_ZOOM         = 4       # higher zoom for scanned pages
NATIVE_CHAR_THRESHOLD = 20      # fewer native chars → scanned page
Y_GROUP_TOLERANCE     = 5       # ±pt for same-line grouping
SPATIAL_LINK_PT       = 15      # ±pt to link text line → table row
MIN_COL_GAP           = 20      # x-distance between column clusters (pt)
MIN_COL_SPANS         = 2       # minimum spans to form a column
CLOSE_SPAN_PT         = 5       # gap below which spans merge (same phrase)
RLM                   = '\u200F'  # Right-to-Left Mark


# ====================================================================
#  OCR engine initialisation
# ====================================================================
paddle_ocr = None
try:
    from paddleocr import PaddleOCR
    paddle_ocr = PaddleOCR(
        use_textline_orientation=True,
        use_doc_orientation_classify=True,
        lang='ar',
    )
except Exception:
    pass

HAS_TESSERACT = False
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    HAS_TESSERACT = True
except Exception:
    pass


# ====================================================================
#  Arabic normalisation  (light — numerals untouched)
# ====================================================================
_DIACRITICS_RE = re.compile(r'[\u064B-\u0652\u0640]')


def normalize_arabic(text: str) -> str:
    if not text:
        return text
    text = _DIACRITICS_RE.sub('', text)
    text = text.replace('\u0623', '\u0627')   # أ → ا
    text = text.replace('\u0625', '\u0627')   # إ → ا
    text = text.replace('\u0622', '\u0627')   # آ → ا
    text = text.replace('\u0649', '\u064A')   # ى → ي
    return text


def has_arabic(text: str) -> bool:
    return any('\u0600' <= ch <= '\u06FF' for ch in text)


# ====================================================================
#  BiDi utility
# ====================================================================
def bidi_display(text: str) -> str:
    """Convert logical-order text to visual-order using the Unicode BiDi
    algorithm.  Falls back to identity if python-bidi is unavailable."""
    if not text or not HAS_BIDI:
        return text
    try:
        return _bidi_get_display(text)
    except Exception:
        return text


# ====================================================================
#  Image pre-processing  (for OCR of scanned / image regions)
# ====================================================================
def preprocess_for_ocr(img_np, *, use_clahe=False):
    """Full pre-processing pipeline:
    1. Upscale low-res images to ≥300 DPI (INTER_CUBIC)
    2. Grayscale conversion
    3. CLAHE contrast equalisation (scanned pages only)
    4. Gaussian blur 5×5
    5. Adaptive Gaussian binarisation (blockSize=31, C=10)
    6. Morphological closing 2×2 (repair broken Arabic ligatures)
    7. Deskew via minAreaRect angle detection
    """
    img = img_np.copy()
    h, w = img.shape[:2]
    scale = 1.0

    # 1. Upscale
    if w < 1000:
        scale = 2.0
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    if len(img.shape) == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalisation)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # 4. Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 5. Adaptive threshold
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=10)

    # 6. Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 7. Deskew
    binary = _deskew(binary)

    return binary, scale


def _deskew(binary_img):
    coords = cv2.findNonZero(cv2.bitwise_not(binary_img))
    if coords is None or len(coords) < 100:
        return binary_img
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) < 0.3 or abs(angle) > 10:
        return binary_img
    h, w = binary_img.shape[:2]
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    return cv2.warpAffine(binary_img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# ====================================================================
#  OCR engines
# ====================================================================
def ocr_image(img_np, *, is_scanned=False):
    """Two-engine OCR strategy:
    - PaddleOCR receives the *original* image (DB detector is trained on raw inputs)
    - Tesseract receives pre-processed binary (it needs clean black-on-white)
    """
    results = _run_paddle_ocr(img_np)
    if not results and HAS_TESSERACT:
        processed, scale = preprocess_for_ocr(img_np, use_clahe=is_scanned)
        results = _run_tesseract(processed, scale)
    return results


def _run_paddle_ocr(img_np):
    results = []
    if paddle_ocr is None:
        return results
    try:
        for pred in paddle_ocr.predict(img_np):
            if pred is None:
                continue
            items = pred if isinstance(pred, list) else [pred]
            for item in items:
                try:
                    _parse_paddle_item(item, results)
                except (TypeError, ValueError, IndexError):
                    continue
    except Exception:
        pass
    return results


def _run_tesseract(binary_img, scale=1.0):
    results = []
    if not HAS_TESSERACT:
        return results
    try:
        pil_img = Image.fromarray(binary_img)
        data = pytesseract.image_to_data(
            pil_img, lang='ara', config=r'--oem 3 --psm 6',
            output_type=pytesseract.Output.DICT)
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i]) if str(data['conf'][i]) != '-1' else 0
            if conf > 25 and text:
                x = data['left'][i]   / scale
                y = data['top'][i]    / scale
                w = data['width'][i]  / scale
                h = data['height'][i] / scale
                results.append(dict(text=text, x0=x, y0=y,
                                    x1=x + w, y1=y + h))
    except Exception:
        pass
    return results


def _parse_paddle_item(item, out: list):
    if isinstance(item, dict):
        text = item.get('text', '').strip()
        conf = item.get('score', item.get('confidence', 0))
        bbox = item.get('text_region', item.get('bbox', None))
        if bbox and conf > 0.3 and text:
            if isinstance(bbox[0], (list, tuple)):
                x0, y0 = min(p[0] for p in bbox), min(p[1] for p in bbox)
                x1, y1 = max(p[0] for p in bbox), max(p[1] for p in bbox)
            else:
                x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
            out.append(dict(text=text, x0=x0, y0=y0, x1=x1, y1=y1))

    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        bbox_pts, text_info = item[0], item[1]
        if not isinstance(text_info, (list, tuple)):
            return
        text = str(text_info[0]).strip()
        conf = float(text_info[1])
        if conf > 0.3 and text:
            if isinstance(bbox_pts[0], (list, tuple)):
                x0, y0 = min(p[0] for p in bbox_pts), min(p[1] for p in bbox_pts)
                x1, y1 = max(p[0] for p in bbox_pts), max(p[1] for p in bbox_pts)
            else:
                x0, y0 = bbox_pts[0], bbox_pts[1]
                x1 = bbox_pts[2] if len(bbox_pts) >= 4 else x0 + 10
                y1 = bbox_pts[3] if len(bbox_pts) >= 4 else y0 + 10
            out.append(dict(text=text, x0=x0, y0=y0, x1=x1, y1=y1))


# ====================================================================
#  Load table regions from detect_tables_cv.py output
# ====================================================================
def load_table_regions(json_path: str) -> dict:
    """Load table bounding boxes from spatial_data.json.

    Returns dict: page_number (1-indexed) → list of
        {x, y, w, h, table_id, detection}   (all in PDF points).
    """
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        data = json.load(f)
    regions: dict[int, list] = {}
    for table in data.get('tables', []):
        pn = table['page_number']
        bp = table['bbox_points']
        regions.setdefault(pn, []).append(dict(
            x=bp['x'], y=bp['y'], w=bp['w'], h=bp['h'],
            table_id=table['table_id'],
            detection=table['detection'],
        ))
    return regions


# ====================================================================
#  Span utilities
# ====================================================================
def _group_into_rows(spans, tolerance=None):
    """Group spans into visual rows by Y-coordinate ±tolerance."""
    if tolerance is None:
        tolerance = Y_GROUP_TOLERANCE
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: s['y0'])
    rows = []
    cur_row = [sorted_spans[0]]
    anchor_y = sorted_spans[0]['y0']
    for sp in sorted_spans[1:]:
        if abs(sp['y0'] - anchor_y) <= tolerance:
            cur_row.append(sp)
        else:
            rows.append(cur_row)
            cur_row = [sp]
            anchor_y = sp['y0']
    if cur_row:
        rows.append(cur_row)
    return rows


def _remove_duplicate_spans(spans):
    """Remove substring and spatially-overlapping duplicate spans."""
    if len(spans) <= 1:
        return spans
    texts = [s['text'] for s in spans]
    keep = []
    for i, sp in enumerate(spans):
        t = texts[i]
        is_sub = any(j != i and len(texts[j]) > len(t) and t in texts[j]
                     for j in range(len(texts)))
        if not is_sub:
            keep.append(sp)
    if len(keep) > 1:
        keep2 = []
        for i, sp in enumerate(keep):
            suppressed = False
            for j, sp2 in enumerate(keep):
                if i == j:
                    continue
                ox0 = max(sp['x0'], sp2['x0'])
                ox1 = min(sp['x1'], sp2['x1'])
                overlap = max(0, ox1 - ox0)
                w1 = sp['x1'] - sp['x0']
                if w1 > 0 and overlap / w1 > 0.5:
                    if (len(sp2['text']) > len(sp['text'])
                            or (len(sp2['text']) == len(sp['text']) and j < i)):
                        suppressed = True
                        break
            if not suppressed:
                keep2.append(sp)
        keep = keep2
    return keep


def _is_numeric_cell(text: str) -> bool:
    stripped = text.strip().strip('()*').strip()
    if not stripped:
        return False
    digits = sum(1 for c in stripped if c.isdigit() or '\u0660' <= c <= '\u0669')
    non_sp = len(stripped.replace(' ', '').replace('\u00A0', ''))
    return non_sp > 0 and digits / non_sp > 0.4


def _is_financial_value(text: str) -> bool:
    """True if text looks like a financial data value (≥5 digits, high ratio).
    Excludes standalone years (4-digit) and dates."""
    text = text.strip()
    if not text or text == '-':
        return False
    clean = text.replace('(', '').replace(')', '').replace('*', '')
    digits = sum(1 for c in clean if c.isdigit() or '\u0660' <= c <= '\u0669')
    if digits < 5:
        return False
    non_sp = len(clean.replace(' ', '').replace('\u00A0', ''))
    return non_sp > 0 and digits / non_sp > 0.5


# ====================================================================
#  Span splitting — break merged label+number spans
# ====================================================================
_ANY_DIGIT = r'[\u0660-\u06690-9]'
_ARABIC_LETTER = r'[\u0600-\u065F\u066A-\u06FF]'

_TRAILING_PAREN_NUM_RE = re.compile(
    r'([\u0600-\u06FF][\u0600-\u06FF\s/]*)'
    r'(\(' + _ANY_DIGIT + r'{1,3}'
    r'(?:[\s\u00A0]+' + _ANY_DIGIT + r'{3})*'
    r'\))'
    r'(' + _ANY_DIGIT + r'{1,3}'
    r'(?:[\s\u00A0]+' + _ANY_DIGIT + r'{3})*'
    r')?\s*$')

_DOUBLE_DECIMAL_RE = re.compile(
    r'^(' + _ANY_DIGIT + r'+\.' + _ANY_DIGIT + r'+)'
    r'[\s\u00A0]*'
    r'(' + _ANY_DIGIT + r'+\.' + _ANY_DIGIT + r'+)$')

_PAREN_THEN_NUM_RE = re.compile(
    r'^(\(' + _ANY_DIGIT + r'{1,3}'
    r'(?:[\s\u00A0]+' + _ANY_DIGIT + r'{3})*'
    r'\))'
    r'(' + _ANY_DIGIT + r'{1,3}'
    r'(?:[\s\u00A0]+' + _ANY_DIGIT + r'{3})*'
    r')\s*$')


def _split_merged_spans(spans, page_width):
    """Split spans that contain merged label+number or multi-number content.
    RTL layout: labels on the right, numbers on the left."""
    result = []
    for sp in spans:
        text = sp['text']
        x0, y0, x1, y1 = sp['x0'], sp['y0'], sp['x1'], sp['y1']
        span_w = x1 - x0
        char_w = span_w / max(len(text), 1)
        split_done = False

        # P1: Arabic label + trailing parenthesised number ± plain number
        m = _TRAILING_PAREN_NUM_RE.match(text)
        if m:
            label = m.group(1).strip()
            paren_num = m.group(2).strip()
            trail_num = (m.group(3) or '').strip()
            if label and paren_num:
                combined = paren_num + (' ' + trail_num if trail_num else '')
                num_frac = len(combined) / max(len(text), 1)
                num_w = span_w * num_frac
                result.append(dict(text=label, x0=x0 + num_w, y0=y0,
                                   x1=x1, y1=y1, src=sp.get('src', 'native')))
                if trail_num:
                    pf = len(paren_num) / max(len(combined), 1)
                    pw = num_w * pf
                    result.append(dict(text=paren_num, x0=x0 + num_w - pw, y0=y0,
                                       x1=x0 + num_w, y1=y1, src=sp.get('src', 'native')))
                    result.append(dict(text=trail_num, x0=x0, y0=y0,
                                       x1=x0 + num_w - pw, y1=y1, src=sp.get('src', 'native')))
                else:
                    result.append(dict(text=paren_num, x0=x0, y0=y0,
                                       x1=x0 + num_w, y1=y1, src=sp.get('src', 'native')))
                split_done = True

        # P2: Dash-separated multi-numbers (e.g. "٥٠١ ٠٠٠-٢٤ ٣٣٧")
        if not split_done and '-' in text:
            parts = [p.strip() for p in text.split('-')]
            non_empty = [p for p in parts if p]
            if (len(non_empty) >= 2
                    and all(_is_numeric_cell(p) or p == '' for p in parts)):
                total_len = sum(len(p) for p in parts) + len(parts) - 1
                cur_x = x0
                for p in parts:
                    ps = p.strip() or '-'
                    pf = max(len(p), 1) / max(total_len, 1)
                    pw = span_w * pf
                    result.append(dict(text=ps, x0=cur_x, y0=y0,
                                       x1=cur_x + pw, y1=y1,
                                       src=sp.get('src', 'native')))
                    cur_x += pw + char_w
                split_done = True

        # P3a: Two decimal numbers glued ("0.٥٧0.40")
        if not split_done:
            m3 = _DOUBLE_DECIMAL_RE.match(text.strip())
            if m3:
                n1, n2 = m3.group(1), m3.group(2)
                frac = len(n1) / max(len(n1) + len(n2), 1)
                mid_x = x0 + span_w * frac
                result.append(dict(text=n1.strip(), x0=mid_x, y0=y0,
                                   x1=x1, y1=y1, src=sp.get('src', 'native')))
                result.append(dict(text=n2.strip(), x0=x0, y0=y0,
                                   x1=mid_x, y1=y1, src=sp.get('src', 'native')))
                split_done = True

        # P3b: (parenthesised number)(plain number)
        if not split_done:
            m3b = _PAREN_THEN_NUM_RE.match(text.strip())
            if m3b:
                n1, n2 = m3b.group(1), m3b.group(2)
                frac = len(n1) / max(len(n1) + len(n2), 1)
                mid_x = x0 + span_w * frac
                result.append(dict(text=n1.strip(), x0=mid_x, y0=y0,
                                   x1=x1, y1=y1, src=sp.get('src', 'native')))
                result.append(dict(text=n2.strip(), x0=x0, y0=y0,
                                   x1=mid_x, y1=y1, src=sp.get('src', 'native')))
                split_done = True

        # P4: Arabic label glued directly to number
        if not split_done:
            m4 = re.match(
                r'(' + _ARABIC_LETTER + r'[\u0600-\u06FF\s]*' + _ARABIC_LETTER + r')'
                r'(' + _ANY_DIGIT + r'{1,3}'
                r'(?:[\s\u00A0]+' + _ANY_DIGIT + r'{3})*'
                r'(?:\.' + _ANY_DIGIT + r'+)?)\s*$', text)
            if m4:
                lbl = m4.group(1).strip()
                num = m4.group(2).strip()
                if lbl and num:
                    nf = len(num) / max(len(text), 1)
                    nw = span_w * nf
                    result.append(dict(text=lbl, x0=x0 + nw, y0=y0,
                                       x1=x1, y1=y1, src=sp.get('src', 'native')))
                    result.append(dict(text=num, x0=x0, y0=y0,
                                       x1=x0 + nw, y1=y1, src=sp.get('src', 'native')))
                    split_done = True

        if not split_done:
            result.append(sp)

    return result


# ====================================================================
#  Column detection  (within table bounds)
# ====================================================================
def _detect_columns(spans, page_width):
    """Cluster x0 positions to find table column boundaries.
    Returns (col_anchors, col_ranges) — both RTL-ordered."""
    if not spans:
        return [], {}
    x_positions = sorted(set(round(sp['x0'], 0) for sp in spans))
    if len(x_positions) < 2:
        return [], {}

    clusters = []
    cur_cl = [x_positions[0]]
    for x in x_positions[1:]:
        if x - cur_cl[-1] <= MIN_COL_GAP:
            cur_cl.append(x)
        else:
            clusters.append(cur_cl)
            cur_cl = [x]
    clusters.append(cur_cl)

    col_anchors = []
    for cl in clusters:
        centre = sum(cl) / len(cl)
        count = sum(1 for sp in spans if abs(sp['x0'] - centre) <= MIN_COL_GAP)
        col_anchors.append((centre, count))

    col_anchors = [(c, n) for c, n in col_anchors if n >= MIN_COL_SPANS]
    col_anchors.sort(key=lambda a: -a[0])   # RTL: rightmost first

    if len(col_anchors) < 2:
        return [], {}

    col_ranges = {}
    for idx, (centre, _) in enumerate(col_anchors):
        left_bound = 0
        right_bound = page_width
        if idx > 0:
            right_bound = (centre + col_anchors[idx - 1][0]) / 2
        if idx < len(col_anchors) - 1:
            left_bound = (centre + col_anchors[idx + 1][0]) / 2
        col_ranges[idx] = (left_bound, right_bound, centre)

    return col_anchors, col_ranges


def _assign_column(span, col_ranges):
    cx = span['x0']
    best_col = -1
    best_dist = float('inf')
    for idx, (lo, hi, _) in col_ranges.items():
        if lo <= cx <= hi:
            return idx
        dist = min(abs(cx - lo), abs(cx - hi))
        if dist < best_dist:
            best_dist = dist
            best_col = idx
    return best_col


# ====================================================================
#  Table structure extraction
# ====================================================================
def extract_table_structure(spans, page_width):
    """Build structured table from spans: column headers + data rows.

    Returns dict with column_headers, rows [{sentence, values, y_coord}],
    label_col — or None if structure can't be recovered.
    """
    if len(spans) < 4:
        return None

    spans = _split_merged_spans(spans, page_width)
    col_anchors, col_ranges = _detect_columns(spans, page_width)
    if not col_ranges or len(col_anchors) < 2:
        return None

    num_cols = len(col_anchors)
    raw_rows = _group_into_rows(spans)
    raw_rows = [_remove_duplicate_spans(r) for r in raw_rows]

    # ── Build grid ───────────────────────────────────────────────────
    grid = []
    grid_y = []
    for row_spans in raw_rows:
        cells = [''] * num_cols
        for sp in row_spans:
            ci = _assign_column(sp, col_ranges)
            if 0 <= ci < num_cols:
                cells[ci] = (cells[ci] + ' ' + sp['text']).strip()
        if any(c.strip() for c in cells):
            grid.append(cells)
            grid_y.append(sum(sp['y0'] for sp in row_spans) / len(row_spans))

    if len(grid) < 2:
        return None

    # ── Identify label column (composite score) ─────────────────────
    scores = [0.0] * num_cols
    for row in grid:
        for ci, cell in enumerate(row):
            cell_s = cell.strip()
            if not cell_s or cell_s == '-':
                continue
            arabic_chars = sum(1 for ch in cell_s if '\u0600' <= ch <= '\u06FF')
            if _is_financial_value(cell_s):
                scores[ci] -= 3
            elif arabic_chars > 2:
                scores[ci] += 2 + arabic_chars / 20.0
            elif _is_numeric_cell(cell_s):
                scores[ci] -= 1
            else:
                scores[ci] += 1
    # RTL bias: rightmost column gets small bonus
    for idx in range(num_cols):
        if idx < len(col_anchors):
            scores[idx] += col_anchors[idx][0] / (page_width * 10)
    label_col = scores.index(max(scores))

    # ── Find header rows (no financial values in data columns) ──────
    header_end = 0
    for i, row in enumerate(grid[:6]):
        fin_count = sum(1 for ci in range(num_cols)
                        if ci != label_col and _is_financial_value(row[ci]))
        if fin_count == 0:
            header_end = i + 1
        else:
            break

    # ── Build column names from headers ─────────────────────────────
    col_names = {}
    if header_end > 0:
        hrow = grid[header_end - 1]
        for ci in range(num_cols):
            if ci == label_col:
                continue
            val = hrow[ci].strip()
            if val and val != '-':
                col_names[ci] = val

    unnamed_idx = 1
    for ci in range(num_cols):
        if ci == label_col:
            continue
        if ci not in col_names:
            # Detect note-reference columns (small numbers like 1-99)
            data_cells = [grid[r][ci].strip()
                          for r in range(header_end, len(grid))
                          if grid[r][ci].strip() and grid[r][ci].strip() != '-']
            all_small = all(
                len(c.replace(' ', '').replace('\u00A0', '')) <= 3
                and _is_numeric_cell(c)
                for c in data_cells
            ) if data_cells else False
            if all_small and len(data_cells) >= 2:
                col_names[ci] = "ايضاح"
            else:
                col_names[ci] = f"عمود {unnamed_idx}"
                unnamed_idx += 1

    # ── Build data rows ─────────────────────────────────────────────
    data_rows = []
    for ri in range(header_end, len(grid)):
        sentence = grid[ri][label_col].strip()
        if not sentence or sentence == '-':
            continue
        values = {}
        for ci in range(num_cols):
            if ci == label_col:
                continue
            cell = grid[ri][ci].strip()
            if cell and cell != '-':
                values[col_names.get(ci, f'col_{ci}')] = cell
        data_rows.append(dict(
            sentence=sentence,
            values=values if values else None,
            y_coord=grid_y[ri],
        ))

    if not data_rows:
        return None

    # ── Ordered column headers ──────────────────────────────────────
    used_cols = set()
    for dr in data_rows:
        if dr['values']:
            used_cols.update(dr['values'].keys())
    ordered_headers = [col_names[ci]
                       for ci in sorted(col_names.keys())
                       if col_names[ci] in used_cols]

    return dict(
        column_headers=ordered_headers,
        rows=data_rows,
        label_col=label_col,
    )


# ====================================================================
#  Text rendering  (RTL-aware)
# ====================================================================
def _render_text_line_rtl(spans, page_width):
    """Render one visual row as a single output line with RTL ordering.
    Close spans (< 5 pt gap) join with space; far spans join with tab."""
    if not spans:
        return ''
    # Sort RTL: rightmost span first
    spans_sorted = sorted(spans, key=lambda sp: page_width - sp['x1'])
    parts = [spans_sorted[0]['text']]
    for i in range(1, len(spans_sorted)):
        prev = spans_sorted[i - 1]
        curr = spans_sorted[i]
        gap = prev['x0'] - curr['x1']
        if gap < 0:
            gap = curr['x1'] - prev['x0']
        parts.append('\t' if gap > CLOSE_SPAN_PT else ' ')
        parts.append(curr['text'])
    return ''.join(parts).strip()


# ====================================================================
#  Spatial partitioning  (table boxes vs rest-of-page)
# ====================================================================
def _spans_in_box(spans, box):
    """Partition spans into those inside and outside a table box (pt coords).
    Uses center-point containment with ±5 pt margin."""
    inside, outside = [], []
    bx0 = box['x'] - 5
    by0 = box['y'] - 5
    bx1 = box['x'] + box['w'] + 5
    by1 = box['y'] + box['h'] + 5
    for sp in spans:
        cx = (sp['x0'] + sp['x1']) / 2
        cy = (sp['y0'] + sp['y1']) / 2
        if bx0 <= cx <= bx1 and by0 <= cy <= by1:
            inside.append(sp)
        else:
            outside.append(sp)
    return inside, outside


# ====================================================================
#  Spatial correlation  (paragraph lines ↔ table rows)
# ====================================================================
def _correlate_text_to_table(text_spans, table_rows, page_width):
    """Link text lines to table rows by Y-coordinate proximity.

    A text line whose Y-centre is within ±SPATIAL_LINK_PT of a table
    row's Y-centre is tagged as an annotation of that row.

    Returns (annotations, free_text) — both lists of rendered strings.
    """
    text_rows = _group_into_rows(text_spans)
    annotations = []   # list of (text, related_row_sentence)
    free_text = []     # list of text strings (no table link)

    for row_spans in text_rows:
        row_spans = _remove_duplicate_spans(row_spans)
        if not row_spans:
            continue
        text_y = sum(sp['y0'] for sp in row_spans) / len(row_spans)
        line = _render_text_line_rtl(row_spans, page_width)
        if not line.strip():
            continue

        # Find nearest table row
        best_row = None
        best_dist = float('inf')
        for tr in table_rows:
            dist = abs(text_y - tr['y_coord'])
            if dist < best_dist:
                best_dist = dist
                best_row = tr

        if best_row is not None and best_dist <= SPATIAL_LINK_PT:
            annotations.append((line, best_row['sentence']))
        else:
            free_text.append(line)

    return annotations, free_text


# ====================================================================
#  Collect native text spans from a PyMuPDF page
# ====================================================================
def _collect_native_spans(page):
    """Extract all native vector-text spans from a fitz page."""
    text_dict = page.get_text("dict")
    spans = []
    for block in text_dict.get("blocks", []):
        if block.get("type", -1) != 0:
            continue
        for line_obj in block.get("lines", []):
            for span in line_obj.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                bbox = span.get("bbox", (0, 0, 0, 0))
                spans.append(dict(
                    text=normalize_arabic(text),
                    x0=bbox[0], y0=bbox[1],
                    x1=bbox[2], y1=bbox[3],
                    src='native'))
    return spans, text_dict


def _collect_image_ocr_spans(page, text_dict):
    """OCR embedded images on native-text pages (e.g. scanned sub-regions)."""
    spans = []
    for block in text_dict.get("blocks", []):
        if block.get("type", -1) != 1:
            continue
        bbox = block.get("bbox")
        if bbox is None:
            continue
        iw, ih = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if iw < 80 or ih < 50:
            continue
        try:
            clip = fitz.Rect(bbox)
            mat = fitz.Matrix(OCR_ZOOM, OCR_ZOOM)
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img_np = np.array(img)
            for r in ocr_image(img_np):
                spans.append(dict(
                    text=normalize_arabic(r['text']),
                    x0=r['x0'] / OCR_ZOOM + bbox[0],
                    y0=r['y0'] / OCR_ZOOM + bbox[1],
                    x1=r['x1'] / OCR_ZOOM + bbox[0],
                    y1=r['y1'] / OCR_ZOOM + bbox[1],
                    src='ocr_img'))
        except Exception:
            pass
    return spans


def _fullpage_ocr_spans(page, zoom):
    """Full-page OCR for scanned pages.  Returns spans in page-point coords."""
    spans = []
    try:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img_np = np.array(img)
        for r in ocr_image(img_np, is_scanned=True):
            spans.append(dict(
                text=normalize_arabic(r['text']),
                x0=r['x0'] / zoom, y0=r['y0'] / zoom,
                x1=r['x1'] / zoom, y1=r['y1'] / zoom,
                src='ocr_full'))
    except Exception:
        pass
    return spans


# ====================================================================
#  Per-page processing
# ====================================================================
def process_page(doc, page_num: int, table_regions: dict) -> dict:
    """Process one page with integrated table + text extraction.

    Returns dict: page_num, is_scanned, paragraphs[], table_count.
    """
    page = doc[page_num]
    page_width = page.rect.width
    if page.rotation in (90, 270):
        page_width = page.rect.height
    if page_width <= 0:
        page_width = 595

    # ── Page classification ──────────────────────────────────────────
    native_spans, text_dict = _collect_native_spans(page)
    native_chars = sum(len(sp['text']) for sp in native_spans)
    is_scanned = native_chars < NATIVE_CHAR_THRESHOLD

    # ── Collect all spans ────────────────────────────────────────────
    if is_scanned:
        all_spans = _fullpage_ocr_spans(page, SCAN_OCR_ZOOM)
    else:
        all_spans = native_spans + _collect_image_ocr_spans(page, text_dict)

    if not all_spans:
        return dict(page_num=page_num + 1, is_scanned=is_scanned,
                    paragraphs=[], table_count=0)

    # ── Get table boxes for this page (1-indexed) ────────────────────
    page_tables = table_regions.get(page_num + 1, [])
    paragraphs = []
    table_all_rows = []     # accumulate all table rows for spatial link

    # ── Process each table region ────────────────────────────────────
    remaining_spans = all_spans
    for tbox in page_tables:
        table_spans, remaining_spans = _spans_in_box(remaining_spans, tbox)
        if not table_spans:
            continue

        tbl = extract_table_structure(table_spans, page_width)
        if tbl and tbl['rows']:
            paragraphs.append(dict(
                type='TABLE_BACKED',
                units=[dict(sentence=r['sentence'],
                            numeric_data=r['values'])
                       for r in tbl['rows']],
                column_headers=tbl['column_headers'],
                table_id=tbox.get('table_id'),
            ))
            table_all_rows.extend(tbl['rows'])
        else:
            # Couldn't structure as table → emit as flat text
            rows = _group_into_rows(table_spans)
            lines = []
            for row in rows:
                row = _remove_duplicate_spans(row)
                line = _render_text_line_rtl(row, page_width)
                if line.strip():
                    lines.append(line)
            if lines:
                paragraphs.append(dict(
                    type='TABLE_REGION_TEXT',
                    units=[dict(sentence=l, numeric_data=None) for l in lines],
                    column_headers=None,
                ))

    # ── Process remaining (non-table) spans ──────────────────────────
    if remaining_spans:
        if table_all_rows:
            # Spatial correlation: link text lines to table rows
            annotations, free_text = _correlate_text_to_table(
                remaining_spans, table_all_rows, page_width)

            if annotations:
                paragraphs.append(dict(
                    type='TABLE_ANNOTATION',
                    units=[dict(sentence=text,
                                numeric_data=None,
                                annotation_for=row_label)
                           for text, row_label in annotations],
                    column_headers=None,
                ))
            if free_text:
                paragraphs.append(dict(
                    type='TEXT_ONLY',
                    units=[dict(sentence=t, numeric_data=None)
                           for t in free_text],
                    column_headers=None,
                ))
        else:
            # No table context — all text is free
            rows = _group_into_rows(remaining_spans)
            lines = []
            for row in rows:
                row = _remove_duplicate_spans(row)
                line = _render_text_line_rtl(row, page_width)
                if line.strip():
                    lines.append(line)
            if lines:
                paragraphs.append(dict(
                    type='TEXT_ONLY',
                    units=[dict(sentence=l, numeric_data=None) for l in lines],
                    column_headers=None,
                ))

    return dict(
        page_num=page_num + 1,
        is_scanned=is_scanned,
        paragraphs=paragraphs,
        table_count=len(page_tables),
    )


# ====================================================================
#  Output formatting  (human-readable TXT)
# ====================================================================
def format_output(all_pages, pdf_name, total_pages):
    lines = []
    lines.append('=' * 60)
    lines.append('  ARABIC FINANCIAL PDF — INTEGRATED EXTRACTION v9')
    lines.append('=' * 60)
    lines.append(f'  Entity:   {RLM}البنك الاهلي المصري')
    lines.append(f'  Document: {RLM}القوائم الماليه المستقله')
    lines.append(f'  Source:   {pdf_name}')
    lines.append(f'  Pages:    {total_pages}')
    lines.append('=' * 60)
    lines.append('')

    for page_data in all_pages:
        pn = page_data['page_num']
        lines.append(f'{"─" * 60}')
        lines.append(f'  Page {pn}')
        lines.append(f'{"─" * 60}')
        lines.append('')

        if not page_data['paragraphs']:
            lines.append('  [No content detected on this page]')
            lines.append('')
            continue

        for pi, para in enumerate(page_data['paragraphs'], start=1):
            ptype = para['type']
            lines.append(f'  [{ptype}] Paragraph {pi}')

            if ptype == 'TABLE_BACKED' and para.get('column_headers'):
                hdrs = ' | '.join(para['column_headers'])
                lines.append(f'  Column Headers: {RLM}{hdrs}')

            lines.append('')
            for ui, unit in enumerate(para['units'], start=1):
                sentence = unit['sentence']
                # Add RLM for Arabic-heavy sentences
                if has_arabic(sentence):
                    sentence = RLM + sentence

                nd = unit.get('numeric_data')
                if nd:
                    vals = ', '.join(f'{k}: {v}' for k, v in nd.items())
                    lines.append(f'    {ui:>3}. {sentence}')
                    lines.append(f'         → {vals}')
                elif unit.get('annotation_for'):
                    lines.append(f'    {ui:>3}. {sentence}')
                    lines.append(f'         ↳ annotation for: {RLM}{unit["annotation_for"]}')
                else:
                    lines.append(f'    {ui:>3}. {sentence}')

            lines.append('')

    # ── Technical Architecture ───────────────────────────────────────
    lines.append('')
    lines.append('=' * 60)
    lines.append('  TECHNICAL ARCHITECTURE')
    lines.append('=' * 60)
    lines.append('')
    lines.append('  1. MODEL SELECTION')
    lines.append('     Primary:  PaddleOCR v3 (PP-OCRv5, lang=ar)')
    lines.append('       - Built-in DB (Differentiable Binarization) text detector')
    lines.append('       - CRNN recognition head trained on Arabic script')
    lines.append('       - Processes raw images (no pre-binarisation needed)')
    lines.append('     Fallback: Tesseract LSTM (--oem 3 --psm 6 -l ara)')
    lines.append('       - Receives pre-processed binary image (binarised + deskewed)')
    lines.append('     Native:   PyMuPDF vector text for digital pages (no OCR needed)')
    lines.append('')
    lines.append('  2. LAYOUT ANALYSIS')
    lines.append('     Table detection: 3-layer hybrid pipeline (detect_tables_cv.py)')
    lines.append('       Layer 1: Morphological line isolation (bordered tables)')
    lines.append('       Layer 2: PyMuPDF word-band analysis (borderless digital tables)')
    lines.append('       Layer 3: OpenCV adaptive threshold (scanned page fallback)')
    lines.append('     Region classification: spatial_data.json bounding boxes')
    lines.append('     Non-table regions: everything outside detected table boxes')
    lines.append('')
    lines.append('  3. SPATIAL CORRELATION')
    lines.append('     Method: Y-coordinate proximity matching')
    lines.append(f'     Threshold: ±{SPATIAL_LINK_PT} pt')
    lines.append('     Text lines within threshold of a table row →')
    lines.append('       tagged as TABLE_ANNOTATION with back-reference')
    lines.append('')
    lines.append('  4. PRE-PROCESSING (for scanned pages)')
    lines.append('     a) DPI normalisation — upscale to ≥300 DPI (INTER_CUBIC)')
    lines.append('     b) CLAHE contrast enhancement (clipLimit=2.0, 8×8 grid)')
    lines.append('     c) Gaussian blur (5×5) — scanner noise reduction')
    lines.append('     d) Adaptive Gaussian binarisation (blockSize=31, C=10)')
    lines.append('     e) Morphological closing (2×2) — repair Arabic ligatures')
    lines.append('     f) Deskew via minAreaRect angle detection (±10° range)')
    lines.append('')
    lines.append('  5. RTL / BiDi LOGIC')
    lines.append(f'     Library: python-bidi ({"✓ loaded" if HAS_BIDI else "✗ not available"})')
    lines.append('     Span ordering: RTL (rightmost first per visual row)')
    lines.append('     Column ordering: RTL (col 0 = rightmost = labels)')
    lines.append('     Mixed content: Arabic text + LTR numbers separated')
    lines.append('       structurally by column detection, not character reordering')
    lines.append('     Output: logical order with U+200F (RLM) for correct rendering')
    lines.append('')

    # ── Statistics ───────────────────────────────────────────────────
    scanned = [p['page_num'] for p in all_pages if p['is_scanned']]
    tbl_pg = [p['page_num'] for p in all_pages
              if any(pa['type'] == 'TABLE_BACKED' for pa in p['paragraphs'])]
    txt_pg = [p['page_num'] for p in all_pages
              if all(pa['type'] in ('TEXT_ONLY', 'TABLE_ANNOTATION')
                     for pa in p['paragraphs'])
              and p['paragraphs']]
    total_u = sum(len(pa['units']) for p in all_pages
                  for pa in p['paragraphs'])
    tbl_u = sum(len(pa['units']) for p in all_pages
                for pa in p['paragraphs'] if pa['type'] == 'TABLE_BACKED')
    ann_u = sum(len(pa['units']) for p in all_pages
                for pa in p['paragraphs'] if pa['type'] == 'TABLE_ANNOTATION')

    lines.append('  STATISTICS')
    lines.append(f'    Total pages:        {total_pages}')
    lines.append(f'    Scanned (OCR):      {scanned or "none"}')
    lines.append(f'    Table pages:        {tbl_pg or "none"}')
    lines.append(f'    Text-only pages:    {txt_pg or "none"}')
    lines.append(f'    Total units:        {total_u}')
    lines.append(f'    Table-backed:       {tbl_u}')
    lines.append(f'    Table annotations:  {ann_u}')
    lines.append(f'    Text-only:          {total_u - tbl_u - ann_u}')
    lines.append('')
    lines.append('=' * 60)

    return '\n'.join(lines)


# ====================================================================
#  JSONL output  (for RAG / downstream ingestion)
# ====================================================================
def write_jsonl(all_pages, output_path):
    """Write one JSON object per Information Unit for RAG ingestion."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for page_data in all_pages:
            for para in page_data['paragraphs']:
                for unit in para['units']:
                    record = dict(
                        page=page_data['page_num'],
                        type=para['type'],
                        sentence=unit['sentence'],
                        numeric_data=unit.get('numeric_data'),
                    )
                    if para.get('column_headers'):
                        record['column_headers'] = para['column_headers']
                    if unit.get('annotation_for'):
                        record['annotation_for'] = unit['annotation_for']
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ====================================================================
#  Main pipeline
# ====================================================================
def main():
    print('=' * 60)
    print('  ARABIC FINANCIAL PDF — INTEGRATED EXTRACTION v9')
    print('  Model: PaddleOCR v3 + PyMuPDF native')
    print('  Layout: 3-Layer Hybrid (detect_tables_cv.py)')
    print('  BiDi:  python-bidi | Preproc: Adaptive + CLAHE')
    print('=' * 60)

    pdf_file = PDF_FILE
    if not os.path.exists(pdf_file):
        print(f'\n  [ERROR] PDF not found: {pdf_file}')
        sys.exit(1)

    # ── Load table regions ───────────────────────────────────────────
    print(f'\n  Loading table regions from {SPATIAL_JSON} ...')
    table_regions = load_table_regions(SPATIAL_JSON)
    if not table_regions:
        print(f'  [WARN] No table regions found.')
        print(f'         Run  python detect_tables_cv.py  first.')
        print(f'         Proceeding without table detection ...')
    else:
        total_tables = sum(len(v) for v in table_regions.values())
        pages_with = sorted(table_regions.keys())
        print(f'  → {total_tables} tables on pages {pages_with}')

    # ── Engine status ────────────────────────────────────────────────
    print(f'\n  OCR Engines:')
    print(f'    PaddleOCR: {"✓ ready (Arabic, PP-OCRv5)" if paddle_ocr else "✗ not available"}')
    print(f'    Tesseract: {"✓ available (LSTM, ara)" if HAS_TESSERACT else "✗ not available"}')
    print(f'    BiDi:      {"✓ python-bidi loaded" if HAS_BIDI else "✗ not available"}')

    # ── Process pages ────────────────────────────────────────────────
    doc = fitz.open(pdf_file)
    total = len(doc)
    print(f'\n  Processing {total} pages ...\n')

    all_pages = []
    for pn in range(total):
        print(f'  Page {pn + 1:>2}/{total} ...', end=' ', flush=True)
        page_data = process_page(doc, pn, table_regions)
        all_pages.append(page_data)

        n_para = len(page_data['paragraphs'])
        n_tbl = sum(1 for p in page_data['paragraphs']
                    if p['type'] == 'TABLE_BACKED')
        n_ann = sum(1 for p in page_data['paragraphs']
                    if p['type'] == 'TABLE_ANNOTATION')
        n_units = sum(len(p['units']) for p in page_data['paragraphs'])
        src = 'OCR' if page_data['is_scanned'] else 'native'
        tc = page_data.get('table_count', 0)

        print(f'[{src}] {n_para} para (T:{n_tbl} A:{n_ann}) '
              f'{n_units} units, {tc} table regions ✓')

    doc.close()

    # ── Write outputs ────────────────────────────────────────────────
    output_text = format_output(all_pages, os.path.basename(pdf_file), total)
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write(output_text)

    write_jsonl(all_pages, OUTPUT_JSONL)

    # ── Summary ──────────────────────────────────────────────────────
    total_units = sum(len(pa['units']) for p in all_pages
                      for pa in p['paragraphs'])
    tbl_units = sum(len(pa['units']) for p in all_pages
                    for pa in p['paragraphs'] if pa['type'] == 'TABLE_BACKED')
    ann_units = sum(len(pa['units']) for p in all_pages
                    for pa in p['paragraphs'] if pa['type'] == 'TABLE_ANNOTATION')

    print(f'\n{"=" * 60}')
    print(f'  ✓ TXT output:  {OUTPUT_TXT}')
    print(f'  ✓ JSONL output: {OUTPUT_JSONL}')
    print(f'  ✓ Total units:  {total_units}')
    print(f'    Table-backed: {tbl_units}')
    print(f'    Annotations:  {ann_units}')
    print(f'    Text-only:    {total_units - tbl_units - ann_units}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
