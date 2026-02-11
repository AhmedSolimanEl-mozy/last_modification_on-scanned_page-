#!/usr/bin/env python3
"""
extract_v8_units.py — Document Understanding Layer v8
=====================================================
Information-Unit extraction for Arabic financial documents.

Pipeline (follows the required spec exactly):
  1. PAGE & LAYOUT ANALYSIS   — Detectron2 PubLayNet
  2. TEXT EXTRACTION           — PyMuPDF native + PaddleOCR / Tesseract
  3. TABLE EXTRACTION          — column detection + structured rows
  4. PARAGRAPH CLASSIFICATION  — TEXT_ONLY vs TABLE_BACKED
  5. INFORMATION UNIT BUILD    — sentence ↔ table-row alignment
  6. OUTPUT                    — UTF-8 TXT with explicit units

Key design decisions:
  • Arabic numerals (٠-٩) are PRESERVED — no conversion.
  • Numbers kept as strings — no float casting.
  • Light Arabic normalisation only (diacritics, alef, ya).
  • Noise regions (stamps, signatures, logos) excluded.
  • Each table row → one Information Unit with sentence + numeric data.
  • Uncertainty → omit data rather than guess.
"""

import os
import re
import sys
import io
import warnings
import urllib.request
from collections import defaultdict

warnings.filterwarnings("ignore")

os.environ['FLAGS_enable_onednn'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['GLOG_minloglevel'] = '3'
os.environ['FLAGS_logtostderr'] = '0'

import fitz
import numpy as np
import cv2
from PIL import Image

# ====================================================================
#  Constants
# ====================================================================
OCR_ZOOM              = 3
LAYOUT_ZOOM           = 2
MIN_IMG_W             = 80
MIN_IMG_H             = 50
NATIVE_CHAR_THRESHOLD = 20
Y_GROUP_TOLERANCE     = 5     # ±5 px vertical row grouping
MIN_COL_GAP           = 20
MIN_COL_SPANS         = 2

OUTPUT_FILE = "extracted_units.txt"

# ====================================================================
#  Detectron2 Layout Model  (PubLayNet Faster R-CNN R-50-FPN)
# ====================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(SCRIPT_DIR, 'publaynet_model')
CONFIG_PATH = os.path.join(MODEL_DIR, 'config.yml')
WEIGHT_PATH = os.path.join(MODEL_DIR, 'model_final.pth')

HF_BASE = ("https://huggingface.co/nlpconnect/"
           "PubLayNet-faster_rcnn_R_50_FPN_3x/resolve/main")

LAYOUT_LABEL_MAP = {0: 'Text', 1: 'Title', 2: 'List',
                    3: 'Table', 4: 'Figure'}

LAYOUT_SCORE_THRESH = 0.5
_layout_predictor = None


def _ensure_model_files():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        print(f"[INFO] Downloading PubLayNet config → {CONFIG_PATH}")
        urllib.request.urlretrieve(f"{HF_BASE}/config.yml", CONFIG_PATH)
    if not os.path.exists(WEIGHT_PATH):
        print(f"[INFO] Downloading PubLayNet weights → {WEIGHT_PATH}")
        urllib.request.urlretrieve(f"{HF_BASE}/model_final.pth", WEIGHT_PATH)


def _get_layout_predictor():
    global _layout_predictor
    if _layout_predictor is not None:
        return _layout_predictor
    _ensure_model_files()
    import detectron2.config
    import detectron2.engine
    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(CONFIG_PATH)
    cfg.MODEL.WEIGHTS = WEIGHT_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = LAYOUT_SCORE_THRESH
    cfg.MODEL.DEVICE = 'cpu'
    _layout_predictor = detectron2.engine.DefaultPredictor(cfg)
    print("[INFO] Detectron2 layout model loaded (PubLayNet R-50-FPN)")
    return _layout_predictor


# ====================================================================
#  Layout detection
# ====================================================================
def detect_layout(img_np):
    predictor = _get_layout_predictor()
    img_bgr   = img_np[:, :, ::-1].copy() if img_np.ndim == 3 else img_np.copy()
    outputs   = predictor(img_bgr)
    instances = outputs['instances'].to('cpu')
    regions   = []
    for i in range(len(instances)):
        cls   = instances.pred_classes[i].item()
        score = instances.scores[i].item()
        box   = instances.pred_boxes[i].tensor[0].tolist()
        regions.append(dict(
            type=LAYOUT_LABEL_MAP.get(cls, f'cls_{cls}'),
            score=score,
            x0=box[0], y0=box[1], x1=box[2], y1=box[3],
        ))
    return regions


def _remove_overlapping_regions(regions):
    if len(regions) <= 1:
        return regions

    def _iou(a, b):
        ix0 = max(a['x0'], b['x0'])
        iy0 = max(a['y0'], b['y0'])
        ix1 = min(a['x1'], b['x1'])
        iy1 = min(a['y1'], b['y1'])
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        area_a = (a['x1'] - a['x0']) * (a['y1'] - a['y0'])
        area_b = (b['x1'] - b['x0']) * (b['y1'] - b['y0'])
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0

    regions = sorted(regions, key=lambda r: -r['score'])
    keep = []
    for r in regions:
        if not any(_iou(r, k) > 0.5 for k in keep):
            keep.append(r)
    return keep


# ====================================================================
#  OCR engines
# ====================================================================
paddle_ocr = None
try:
    from paddleocr import PaddleOCR
    paddle_ocr = PaddleOCR(
        use_textline_orientation=True,
        use_doc_orientation_classify=True,
        lang='ar',
    )
    print("[INFO] PaddleOCR (Arabic) ready")
except Exception as exc:
    print(f"[WARN] PaddleOCR not available: {exc}")

HAS_TESSERACT = False
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    HAS_TESSERACT = True
    print("[INFO] Tesseract OCR available")
except Exception:
    pass


# ====================================================================
#  Arabic normalisation  (light — no numeral conversion)
# ====================================================================
_DIACRITICS_RE = re.compile(r'[\u064B-\u0652\u0640]')


def normalize_arabic(text: str) -> str:
    """Light normalisation: diacritics, alef variants, ya.  Numerals untouched."""
    if not text:
        return text
    text = _DIACRITICS_RE.sub('', text)
    text = text.replace('\u0623', '\u0627')   # أ → ا
    text = text.replace('\u0625', '\u0627')   # إ → ا
    text = text.replace('\u0622', '\u0627')   # آ → ا
    text = text.replace('\u0649', '\u064A')   # ى → ي
    return text


# ====================================================================
#  OpenCV pre-processing  (for OCR of scanned regions)
# ====================================================================
TARGET_DPI  = 300
SRC_DPI_EST = 150


def preprocess_for_ocr(img_np, *, target_dpi=TARGET_DPI):
    img   = img_np.copy()
    h, w  = img.shape[:2]
    scale = 1.0
    if w < 1000:
        scale = target_dpi / SRC_DPI_EST
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    if len(img.shape) == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary  = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = _deskew(binary)
    return binary, scale


def _deskew(binary_img):
    coords = cv2.findNonZero(cv2.bitwise_not(binary_img))
    if coords is None or len(coords) < 100:
        return binary_img
    rect  = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) < 0.3 or abs(angle) > 10:
        return binary_img
    h, w   = binary_img.shape[:2]
    centre = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(centre, angle, 1.0)
    return cv2.warpAffine(binary_img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# ====================================================================
#  OCR helpers
# ====================================================================
def ocr_image_to_elements(img_np):
    results = _run_paddle_ocr(img_np)
    if not results and HAS_TESSERACT:
        processed, scale = preprocess_for_ocr(img_np)
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
                x0 = min(p[0] for p in bbox)
                y0 = min(p[1] for p in bbox)
                x1 = max(p[0] for p in bbox)
                y1 = max(p[1] for p in bbox)
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
                x0 = min(p[0] for p in bbox_pts)
                y0 = min(p[1] for p in bbox_pts)
                x1 = max(p[0] for p in bbox_pts)
                y1 = max(p[1] for p in bbox_pts)
            else:
                x0, y0 = bbox_pts[0], bbox_pts[1]
                x1 = bbox_pts[2] if len(bbox_pts) >= 4 else x0 + 10
                y1 = bbox_pts[3] if len(bbox_pts) >= 4 else y0 + 10
            out.append(dict(text=text, x0=x0, y0=y0, x1=x1, y1=y1))


# ====================================================================
#  Utility:  page width
# ====================================================================
def _effective_page_width(page):
    rot = page.rotation
    if rot in (90, 270):
        return page.rect.height
    return page.rect.width


# ====================================================================
#  Span grouping
# ====================================================================
def _group_into_rows(spans, tolerance=None):
    """Group spans into visual rows by Y-coordinate ±tolerance."""
    if tolerance is None:
        tolerance = Y_GROUP_TOLERANCE
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: s['y0'])
    rows = []
    cur_row  = [sorted_spans[0]]
    anchor_y = sorted_spans[0]['y0']
    for sp in sorted_spans[1:]:
        if abs(sp['y0'] - anchor_y) <= tolerance:
            cur_row.append(sp)
        else:
            rows.append(cur_row)
            cur_row  = [sp]
            anchor_y = sp['y0']
    if cur_row:
        rows.append(cur_row)
    return rows


# ====================================================================
#  De-duplication
# ====================================================================
def _remove_duplicate_spans(spans):
    """Remove substring and spatially-overlapping duplicate spans."""
    if len(spans) <= 1:
        return spans
    texts = [s['text'] for s in spans]
    keep  = []
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
                ox0     = max(sp['x0'], sp2['x0'])
                ox1     = min(sp['x1'], sp2['x1'])
                overlap = max(0, ox1 - ox0)
                w1      = sp['x1'] - sp['x0']
                if w1 > 0 and overlap / w1 > 0.5:
                    if (len(sp2['text']) > len(sp['text'])
                            or (len(sp2['text']) == len(sp['text']) and j < i)):
                        suppressed = True
                        break
            if not suppressed:
                keep2.append(sp)
        keep = keep2
    return keep


# ====================================================================
#  Numeric detection  (handles both ٠-٩ and 0-9)
# ====================================================================
_ANY_DIGIT = r'[\u0660-\u06690-9]'


def _is_numeric_cell(text):
    """True if text is primarily a number (ignoring parens/spaces)."""
    stripped = text.strip().strip('()*').strip()
    if not stripped:
        return False
    digits = sum(1 for c in stripped if c.isdigit() or '\u0660' <= c <= '\u0669')
    non_sp = len(stripped.replace(' ', '').replace('\u00A0', ''))
    return non_sp > 0 and digits / non_sp > 0.4


def _is_financial_value(text):
    """True if text looks like a financial data value (≥5 digits, high ratio).

    Excludes standalone years (4-digit) and dates with some digits,
    so that header rows with year labels are NOT treated as data rows.
    """
    text = text.strip()
    if not text or text == '-':
        return False
    clean = text.replace('(', '').replace(')', '').replace('*', '')
    digits = sum(1 for c in clean if c.isdigit() or '\u0660' <= c <= '\u0669')
    if digits < 5:                 # years have 4 digits → not financial
        return False
    non_sp = len(clean.replace(' ', '').replace('\u00A0', ''))
    if non_sp == 0:
        return False
    return digits / non_sp > 0.5   # strict: mostly digits


def _has_arabic(text):
    return any('\u0600' <= ch <= '\u06FF' for ch in text)


# ====================================================================
#  Span splitting — break merged label+number spans
# ====================================================================
_NUM_TOKEN_RE = re.compile(
    r'(\(?' + _ANY_DIGIT + r'{1,3}'
    r'(?:[\s\u00A0]+' + _ANY_DIGIT + r'{3})*'
    r'(?:\.' + _ANY_DIGIT + r'+)?'
    r'\)?)')

_TRAILING_PAREN_NUM_RE = re.compile(
    r'([\u0600-\u06FF][\u0600-\u06FF\s/]*)'
    r'(\(' + _ANY_DIGIT + r'{1,3}'
    r'(?:[\s\u00A0]+' + _ANY_DIGIT + r'{3})*'
    r'\))'
    r'(' + _ANY_DIGIT + r'{1,3}'
    r'(?:[\s\u00A0]+' + _ANY_DIGIT + r'{3})*'
    r')?'
    r'\s*$')

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
    """Split spans containing merged label+number or multi-number content."""
    result = []
    for sp in spans:
        text = sp['text']
        x0, y0, x1, y1 = sp['x0'], sp['y0'], sp['x1'], sp['y1']
        span_w = x1 - x0
        char_w = span_w / max(len(text), 1)
        split_done = False

        # P1: Arabic label + trailing parenthesised number
        m = _TRAILING_PAREN_NUM_RE.match(text)
        if m:
            label     = m.group(1).strip()
            paren_num = m.group(2).strip()
            trail_num = (m.group(3) or '').strip()
            if label and paren_num:
                combined  = paren_num + (' ' + trail_num if trail_num else '')
                num_frac  = len(combined) / max(len(text), 1)
                num_w     = span_w * num_frac
                result.append(dict(text=label, x0=x0+num_w, y0=y0,
                                   x1=x1, y1=y1, src=sp.get('src','native'),
                                   region=sp.get('region','')))
                if trail_num:
                    pf = len(paren_num) / max(len(combined), 1)
                    pw = num_w * pf
                    result.append(dict(text=paren_num, x0=x0+num_w-pw, y0=y0,
                                       x1=x0+num_w, y1=y1,
                                       src=sp.get('src','native'),
                                       region=sp.get('region','')))
                    result.append(dict(text=trail_num, x0=x0, y0=y0,
                                       x1=x0+num_w-pw, y1=y1,
                                       src=sp.get('src','native'),
                                       region=sp.get('region','')))
                else:
                    result.append(dict(text=paren_num, x0=x0, y0=y0,
                                       x1=x0+num_w, y1=y1,
                                       src=sp.get('src','native'),
                                       region=sp.get('region','')))
                split_done = True

        # P2: Dash-separated multi-numbers
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
                                       x1=cur_x+pw, y1=y1,
                                       src=sp.get('src','native'),
                                       region=sp.get('region','')))
                    cur_x += pw + char_w
                split_done = True

        # P3a: Two decimal numbers glued
        if not split_done:
            m3 = _DOUBLE_DECIMAL_RE.match(text.strip())
            if m3:
                n1, n2 = m3.group(1), m3.group(2)
                frac   = len(n1) / max(len(n1)+len(n2), 1)
                mid_x  = x0 + span_w * frac
                result.append(dict(text=n1.strip(), x0=mid_x, y0=y0,
                                   x1=x1, y1=y1, src=sp.get('src','native'),
                                   region=sp.get('region','')))
                result.append(dict(text=n2.strip(), x0=x0, y0=y0,
                                   x1=mid_x, y1=y1, src=sp.get('src','native'),
                                   region=sp.get('region','')))
                split_done = True

        # P3b: (paren-num)plain-num
        if not split_done:
            m3b = _PAREN_THEN_NUM_RE.match(text.strip())
            if m3b:
                n1, n2 = m3b.group(1), m3b.group(2)
                frac   = len(n1) / max(len(n1)+len(n2), 1)
                mid_x  = x0 + span_w * frac
                result.append(dict(text=n1.strip(), x0=mid_x, y0=y0,
                                   x1=x1, y1=y1, src=sp.get('src','native'),
                                   region=sp.get('region','')))
                result.append(dict(text=n2.strip(), x0=x0, y0=y0,
                                   x1=mid_x, y1=y1, src=sp.get('src','native'),
                                   region=sp.get('region','')))
                split_done = True

        # P4: Arabic label glued to number
        if not split_done:
            _AL = r'[\u0600-\u065F\u066A-\u06FF]'
            m4 = re.match(
                r'(' + _AL + r'[\u0600-\u06FF\s]*' + _AL + r')'
                r'(' + _ANY_DIGIT + r'{1,3}'
                r'(?:[\s\u00A0]+' + _ANY_DIGIT + r'{3})*'
                r'(?:\.' + _ANY_DIGIT + r'+)?)\s*$', text)
            if m4:
                lbl = m4.group(1).strip()
                num = m4.group(2).strip()
                if lbl and num:
                    nf = len(num) / max(len(text), 1)
                    nw = span_w * nf
                    result.append(dict(text=lbl, x0=x0+nw, y0=y0,
                                       x1=x1, y1=y1,
                                       src=sp.get('src','native'),
                                       region=sp.get('region','')))
                    result.append(dict(text=num, x0=x0, y0=y0,
                                       x1=x0+nw, y1=y1,
                                       src=sp.get('src','native'),
                                       region=sp.get('region','')))
                    split_done = True

        if not split_done:
            result.append(sp)
    return result


# ====================================================================
#  Column detection
# ====================================================================
def _detect_columns(spans, page_width):
    """Cluster x0 positions to find table column boundaries."""
    if not spans:
        return [], {}
    x_positions = sorted(set(round(sp['x0'], 0) for sp in spans))
    if len(x_positions) < 2:
        return [], {}

    clusters = []
    cur_cl   = [x_positions[0]]
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
        count  = sum(1 for sp in spans
                     if abs(sp['x0'] - centre) <= MIN_COL_GAP)
        col_anchors.append((centre, count))

    col_anchors = [(c, n) for c, n in col_anchors if n >= MIN_COL_SPANS]
    col_anchors.sort(key=lambda x: -x[0])   # RTL: rightmost first

    if len(col_anchors) < 2:
        return [], {}

    col_ranges = {}
    for idx, (centre, _) in enumerate(col_anchors):
        left_bound  = 0
        right_bound = page_width
        if idx > 0:
            right_bound = (centre + col_anchors[idx - 1][0]) / 2
        if idx < len(col_anchors) - 1:
            left_bound = (centre + col_anchors[idx + 1][0]) / 2
        col_ranges[idx] = (left_bound, right_bound, centre)

    return col_anchors, col_ranges


def _assign_column(span, col_ranges):
    cx = span['x0']
    best_col  = -1
    best_dist = float('inf')
    for idx, (lo, hi, _centre) in col_ranges.items():
        if lo <= cx <= hi:
            return idx
        dist = min(abs(cx - lo), abs(cx - hi))
        if dist < best_dist:
            best_dist = dist
            best_col  = idx
    return best_col


# ====================================================================
#  Native-span collection
# ====================================================================
def _native_spans_in_box(text_dict, bx0, by0, bx1, by1):
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
                cx   = (bbox[0] + bbox[2]) / 2
                cy   = (bbox[1] + bbox[3]) / 2
                if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                    spans.append(dict(
                        text=normalize_arabic(text),
                        x0=bbox[0], y0=bbox[1],
                        x1=bbox[2], y1=bbox[3],
                        src='native'))
    return spans


# ====================================================================
#  TABLE STRUCTURE EXTRACTION
# ====================================================================
def _extract_table_structure(spans, page_width):
    """Extract structured table data from a set of spans.

    Returns dict with keys:
        column_headers : list[str]     – header names (RTL order)
        rows           : list[dict]    – each {sentence, values}
        label_col_name : str           – name assigned to label column
    or None if the spans don't form a recognisable table.
    """
    if len(spans) < 4:
        return None

    # Split merged label+number spans
    spans = _split_merged_spans(spans, page_width)

    # Detect columns
    col_anchors, col_ranges = _detect_columns(spans, page_width)
    if not col_ranges or len(col_anchors) < 2:
        return None

    num_cols = len(col_anchors)

    # Group into rows
    raw_rows = _group_into_rows(spans)
    raw_rows = [_remove_duplicate_spans(r) for r in raw_rows]

    # Build grid
    grid = []
    for row_spans in raw_rows:
        cells = [''] * num_cols
        for sp in row_spans:
            ci = _assign_column(sp, col_ranges)
            if 0 <= ci < num_cols:
                cells[ci] = (cells[ci] + ' ' + sp['text']).strip()
        if any(c.strip() for c in cells):
            grid.append(cells)

    if len(grid) < 2:
        return None

    # Identify the label column using a composite score:
    #  +2 for each non-numeric non-empty cell (labels are text)
    #  +1 per 20 Arabic characters in the column
    #  +1 bonus for rightmost position (RTL convention)
    #  -3 for each cell that IS a financial value
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
                scores[ci] -= 1    # small number (year, ref)
            else:
                scores[ci] += 1
    # RTL bias: rightmost column gets a small bonus
    if col_anchors:
        for idx in range(num_cols):
            anchor_x = col_anchors[idx][0] if idx < len(col_anchors) else 0
            scores[idx] += anchor_x / (page_width * 10)  # small positional boost
    label_col = scores.index(max(scores))

    # Find header rows — rows whose VALUE columns have no financial data.
    # Uses _is_financial_value (≥5 digits, >50% digit ratio) so that
    # year labels like "٤٢٠٢" (4 digits) are treated as headers, not data.
    header_end = 0
    for i, row in enumerate(grid[:6]):      # check first 6 rows max
        fin_count = sum(1 for ci in range(num_cols)
                        if ci != label_col
                        and _is_financial_value(row[ci]))
        if fin_count == 0:
            header_end = i + 1
        else:
            break

    # Build column names from last header row
    col_names = {}
    if header_end > 0:
        hrow = grid[header_end - 1]
        for ci in range(num_cols):
            if ci == label_col:
                continue
            val = hrow[ci].strip()
            if val and val != '-':
                col_names[ci] = val

    # Fallback names for unnamed value columns
    # Detect note-reference columns: if most data cells have small numbers (1-99)
    unnamed_idx = 1
    for ci in range(num_cols):
        if ci == label_col:
            continue
        if ci not in col_names:
            # Check if this column is a note-reference column
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

    # Build structured data rows
    data_rows = []
    for row in grid[header_end:]:
        sentence = row[label_col].strip()
        if not sentence or sentence == '-':
            continue

        values = {}
        for ci in range(num_cols):
            if ci == label_col:
                continue
            cell = row[ci].strip()
            if cell and cell != '-':
                values[col_names[ci]] = cell

        data_rows.append(dict(sentence=sentence, values=values if values else None))

    if not data_rows:
        return None

    # Build ordered column header list — only include columns that
    # have at least one value across all data rows
    used_cols = set()
    for dr in data_rows:
        if dr['values']:
            used_cols.update(dr['values'].keys())

    ordered_headers = []
    for ci in sorted(col_names.keys()):
        name = col_names[ci]
        if name in used_cols:
            ordered_headers.append(name)

    return dict(
        column_headers=ordered_headers,
        rows=data_rows,
        label_col=label_col,
    )


# ====================================================================
#  TEXT LINE EXTRACTION (for non-table regions)
# ====================================================================
def _extract_text_lines(spans, page_width):
    """Group spans into visual rows and render each as a text line (RTL)."""
    if not spans:
        return []
    rows  = _group_into_rows(spans)
    lines = []
    for row in rows:
        row = _remove_duplicate_spans(row)
        # Sort RTL (rightmost first)
        row.sort(key=lambda sp: -sp['x1'])
        parts = []
        for sp in row:
            parts.append(sp['text'])
        line = ' '.join(parts).strip()
        if line:
            lines.append(line)
    return lines


# ====================================================================
#  NOISE DETECTION
# ====================================================================
def _is_noise_region(region, page_width, page_height, is_scanned, text_dict):
    """Return True if a region is noise (stamp, logo, signature)."""
    rtype = region['type']
    if rtype != 'Figure':
        return False
    # On native pages, small figures without text → noise
    if not is_scanned:
        pdf_x0 = region['x0'] / LAYOUT_ZOOM
        pdf_y0 = region['y0'] / LAYOUT_ZOOM
        pdf_x1 = region['x1'] / LAYOUT_ZOOM
        pdf_y1 = region['y1'] / LAYOUT_ZOOM
        fig_area  = (pdf_x1 - pdf_x0) * (pdf_y1 - pdf_y0)
        page_area = page_width * page_height
        if fig_area < page_area * 0.15:
            return True   # small figure → logo/stamp
        # Large figure: check if it has native text
        probe = _native_spans_in_box(text_dict, pdf_x0, pdf_y0, pdf_x1, pdf_y1)
        if not probe:
            return True   # large image with no text → signature/graphic
    return False


# ====================================================================
#  PARAGRAPH CLASSIFICATION & UNIT CONSTRUCTION
# ====================================================================
def _build_table_units(table_data):
    """Build Information Units from structured table data."""
    units = []
    for row in table_data['rows']:
        unit = dict(
            sentence=row['sentence'],
            numeric_data=row['values'],     # dict or None
        )
        units.append(unit)
    return units


def _build_text_units(text_lines):
    """Build Information Units from plain text lines."""
    units = []
    for line in text_lines:
        units.append(dict(
            sentence=line,
            numeric_data=None,
        ))
    return units


# ====================================================================
#  PER-PAGE PROCESSING
# ====================================================================
def process_page(pdf_path: str, page_num: int):
    """Process one page and return structured paragraph data.

    Returns dict with:
        page_num    : int
        is_scanned  : bool
        paragraphs  : list[dict]   each {type, units, region_label, col_headers}
    """
    doc  = fitz.open(pdf_path)
    page = doc[page_num]
    page_width  = _effective_page_width(page)
    page_height = page.rect.height
    if page_width <= 0:
        page_width = 595

    # ── 1. Render for layout detection ───────────────────────────────
    lz_mat = fitz.Matrix(LAYOUT_ZOOM, LAYOUT_ZOOM)
    lz_pix = page.get_pixmap(matrix=lz_mat)
    lz_img = Image.open(io.BytesIO(lz_pix.tobytes("png")))
    lz_np  = np.array(lz_img)

    # ── 2. Layout analysis ───────────────────────────────────────────
    regions = detect_layout(lz_np)
    regions = _remove_overlapping_regions(regions)
    regions.sort(key=lambda r: r['y0'])   # reading order top→bottom

    # ── 3. Page classification ───────────────────────────────────────
    text_dict    = page.get_text("dict")
    native_chars = sum(
        len(sp.get("text", "").strip())
        for blk in text_dict.get("blocks", []) if blk.get("type") == 0
        for ln in blk.get("lines", [])
        for sp in ln.get("spans", []))
    is_scanned = native_chars < NATIVE_CHAR_THRESHOLD

    # ── 4. Collect ALL spans from layout regions ────────────────────
    all_spans   = []
    seen_spans  = set()
    noise_count = 0

    for reg in regions:
        pdf_x0 = reg['x0'] / LAYOUT_ZOOM
        pdf_y0 = reg['y0'] / LAYOUT_ZOOM
        pdf_x1 = reg['x1'] / LAYOUT_ZOOM
        pdf_y1 = reg['y1'] / LAYOUT_ZOOM

        # ── Noise filter ─────────────────────────────────────────────
        if _is_noise_region(reg, page_width, page_height,
                            is_scanned, text_dict):
            noise_count += 1
            continue

        # ── Collect spans for this region ────────────────────────────
        if not is_scanned:
            native = _native_spans_in_box(text_dict,
                                          pdf_x0, pdf_y0, pdf_x1, pdf_y1)
            for sp in native:
                key = (round(sp['y0'], 1), round(sp['x0'], 1), sp['text'])
                if key not in seen_spans:
                    seen_spans.add(key)
                    sp['region'] = reg['type']
                    all_spans.append(sp)
        else:
            # OCR the region
            try:
                clip = fitz.Rect(pdf_x0, pdf_y0, pdf_x1, pdf_y1)
                mat  = fitz.Matrix(OCR_ZOOM, OCR_ZOOM)
                pix  = page.get_pixmap(matrix=mat, clip=clip)
                img  = Image.open(io.BytesIO(pix.tobytes("png")))
                img_np = np.array(img)
                for r in ocr_image_to_elements(img_np):
                    ntxt = normalize_arabic(r['text'])
                    ox0  = r['x0'] / OCR_ZOOM + pdf_x0
                    oy0  = r['y0'] / OCR_ZOOM + pdf_y0
                    key  = (round(oy0, 1), round(ox0, 1), ntxt)
                    if key not in seen_spans:
                        seen_spans.add(key)
                        all_spans.append(dict(
                            text=ntxt, x0=ox0, y0=oy0,
                            x1=r['x1'] / OCR_ZOOM + pdf_x0,
                            y1=r['y1'] / OCR_ZOOM + pdf_y0,
                            src='ocr', region=reg['type']))
            except Exception:
                pass

    # ── 5. Catch-all: native spans not in any region ─────────────────
    if not is_scanned:
        for block in text_dict.get("blocks", []):
            if block.get("type", -1) != 0:
                continue
            for line_obj in block.get("lines", []):
                for span in line_obj.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    bbox = span.get("bbox", (0, 0, 0, 0))
                    ntxt = normalize_arabic(text)
                    key  = (round(bbox[1], 1), round(bbox[0], 1), ntxt)
                    if key not in seen_spans:
                        seen_spans.add(key)
                        all_spans.append(dict(
                            text=ntxt, x0=bbox[0], y0=bbox[1],
                            x1=bbox[2], y1=bbox[3],
                            src='native', region='uncovered'))

    if not all_spans:
        # empty — try full-page fallback below
        pass

    # ── 6. Whole-page table test ─────────────────────────────────────
    #   Merge ALL collected spans and decide if the page is table-backed.
    #   This avoids the problem of many small Text regions each having
    #   too few spans to trigger table detection individually.
    paragraphs = []

    if all_spans:
        # Split merged label+number spans
        all_spans = _split_merged_spans(all_spans, page_width)

        rows_all = _group_into_rows(all_spans)
        num_rows_all = sum(1 for r in rows_all
                          if any(_is_numeric_cell(sp['text']) for sp in r))

        if num_rows_all >= 5 and not is_scanned:
            # ── PAGE IS TABLE-BACKED ────────────────────────────────
            # First, extract any header / title text that precedes
            # the numeric rows (those are TEXT_ONLY paragraphs).
            # Then the numeric block becomes one TABLE_BACKED paragraph.
            header_lines = []
            table_spans  = []
            first_num_y  = None

            for row in rows_all:
                has_num = any(_is_numeric_cell(sp['text']) for sp in row)
                if has_num and first_num_y is None:
                    first_num_y = min(sp['y0'] for sp in row)

            if first_num_y is not None:
                for sp in all_spans:
                    if sp['y0'] < first_num_y - Y_GROUP_TOLERANCE:
                        header_lines.append(sp)
                    else:
                        table_spans.append(sp)
            else:
                table_spans = all_spans

            # Emit header text as TEXT_ONLY
            if header_lines:
                hlines = _extract_text_lines(header_lines, page_width)
                if hlines:
                    paragraphs.append(dict(
                        type='TEXT_ONLY',
                        units=_build_text_units(hlines),
                        region_label='header',
                        column_headers=None,
                    ))

            # Emit table as TABLE_BACKED — but validate quality first.
            # A real table should have ≥2 value columns carrying data
            # in at least 30% of rows.  Otherwise it's a text page
            # with scattered numbers.
            tbl = _extract_table_structure(table_spans, page_width)
            table_valid = False
            if tbl and tbl['rows']:
                # Count how many distinct column names actually carry
                # financial values across the data rows
                col_value_counts = {}
                for row in tbl['rows']:
                    if row['values']:
                        for cname, val in row['values'].items():
                            if _is_financial_value(val) or _is_numeric_cell(val):
                                col_value_counts[cname] = \
                                    col_value_counts.get(cname, 0) + 1
                n_data_rows = len(tbl['rows'])
                # How many columns have data in ≥30% of rows?
                active_cols = sum(1 for cnt in col_value_counts.values()
                                  if cnt >= max(2, n_data_rows * 0.3))
                table_valid = active_cols >= 2

            if tbl and table_valid:
                paragraphs.append(dict(
                    type='TABLE_BACKED',
                    units=_build_table_units(tbl),
                    region_label='Table',
                    column_headers=tbl['column_headers'],
                ))
            if not (tbl and table_valid):
                # Fallback: render as text lines (table not valid)
                tlines = _extract_text_lines(
                    table_spans if table_spans else all_spans, page_width)
                if tlines:
                    paragraphs.append(dict(
                        type='TEXT_ONLY',
                        units=_build_text_units(tlines),
                        region_label='fallback_text',
                        column_headers=None,
                    ))
        else:
            # ── PAGE IS TEXT-ONLY ───────────────────────────────────
            lines = _extract_text_lines(all_spans, page_width)
            if lines:
                paragraphs.append(dict(
                    type='TEXT_ONLY',
                    units=_build_text_units(lines),
                    region_label='text',
                    column_headers=None,
                ))

    # ── 6. Full-page OCR fallback ────────────────────────────────────
    if not paragraphs and is_scanned:
        try:
            mat    = fitz.Matrix(OCR_ZOOM, OCR_ZOOM)
            pix    = page.get_pixmap(matrix=mat)
            img    = Image.open(io.BytesIO(pix.tobytes("png")))
            img_np = np.array(img)
            elems  = []
            for r in ocr_image_to_elements(img_np):
                elems.append(dict(
                    text=normalize_arabic(r['text']),
                    x0=r['x0'] / OCR_ZOOM, y0=r['y0'] / OCR_ZOOM,
                    x1=r['x1'] / OCR_ZOOM, y1=r['y1'] / OCR_ZOOM,
                    src='ocr_full', region='full_page'))
            if elems:
                lines = _extract_text_lines(elems, page_width)
                if lines:
                    paragraphs.append(dict(
                        type='TEXT_ONLY',
                        units=_build_text_units(lines),
                        region_label='full_page_ocr',
                        column_headers=None,
                    ))
        except Exception:
            pass

    doc.close()
    return dict(
        page_num=page_num + 1,   # 1-indexed for output
        is_scanned=is_scanned,
        paragraphs=paragraphs,
        noise_skipped=noise_count,
    )


# ====================================================================
#  OUTPUT FORMATTING
# ====================================================================
def format_output(all_pages, pdf_name, total_pages):
    """Format all pages into the specified TXT output."""
    lines = []

    lines.append('=' * 50)
    lines.append('  DOCUMENT UNDERSTANDING OUTPUT')
    lines.append(f'  Entity: البنك الاهلي المصري')
    lines.append(f'  Document: القوائم الماليه المستقله')
    lines.append(f'  Source: {pdf_name}')
    lines.append(f'  Pages: {total_pages}')
    lines.append('=' * 50)
    lines.append('')

    for page_data in all_pages:
        pn = page_data['page_num']
        lines.append(f'===== Page {pn} =====')
        lines.append('')

        if not page_data['paragraphs']:
            lines.append('[No content detected on this page]')
            lines.append('')
            continue

        for pi, para in enumerate(page_data['paragraphs'], start=1):
            ptype = para['type']
            lines.append(f'[Paragraph {pi}] ({ptype})')
            lines.append('')

            # If TABLE_BACKED, show column headers
            if ptype == 'TABLE_BACKED' and para.get('column_headers'):
                lines.append(f'  Column Headers: {" | ".join(para["column_headers"])}')
                lines.append('')

            for ui, unit in enumerate(para['units'], start=1):
                lines.append(f'  [Unit {ui}]')
                lines.append(f'  Sentence:')
                lines.append(f'  {unit["sentence"]}')
                lines.append('')

                nd = unit.get('numeric_data')
                if nd:
                    lines.append(f'  Numeric Data:')
                    for col_name, val in nd.items():
                        lines.append(f'  - {col_name}: {val}')
                else:
                    lines.append(f'  Numeric Data:')
                    lines.append(f'  None')

                lines.append('')
                lines.append('  ' + '-' * 38)
                lines.append('')

        lines.append('')

    # ── Technical Note ───────────────────────────────────────────────
    lines.append('=' * 50)
    lines.append('  TECHNICAL NOTE')
    lines.append('=' * 50)
    lines.append('')
    lines.append('  OCR Tools Used:')
    lines.append('    - PaddleOCR (Arabic, PP-OCRv5) — primary OCR engine')
    lines.append('    - Tesseract (--oem 3 --psm 6 -l ara) — fallback')
    lines.append('    - OpenCV adaptive threshold + deskew — preprocessing')
    lines.append('')
    lines.append('  Layout Analysis:')
    lines.append('    - Detectron2 with PubLayNet Faster R-CNN R-50-FPN')
    lines.append('    - Regions classified: Text, Title, List, Table, Figure')
    lines.append('    - Figure regions filtered as noise (stamps/logos/signatures)')
    lines.append('')
    lines.append('  Table-Sentence Alignment:')
    lines.append('    - Spans grouped into visual rows (±5px Y-tolerance)')
    lines.append('    - Columns detected by clustering X-positions (gap ≥20px)')
    lines.append('    - Label column identified by composite score')
    lines.append('      (Arabic text weight, non-numeric penalty, RTL position bias)')
    lines.append('    - Header rows: first rows without financial-value cells (≥5 digits)')
    lines.append('    - Each data row: label cell → Sentence, value cells → Numeric Data')
    lines.append('    - Merged label+number spans split using regex patterns')
    lines.append('    - Table validation: ≥2 columns with consistent numeric data')
    lines.append('')
    lines.append('  Known Limitations:')
    lines.append('    - Pages 1-3 (scanned) have lower OCR accuracy')
    lines.append('    - Complex multi-level headers may not fully decompose')
    lines.append('    - Some OCR artifacts from stamps/signatures may persist')
    lines.append('    - Arabic numerals (٠-٩) preserved as-is (no conversion)')
    lines.append('    - Numbers kept as original strings (spaces between digit groups preserved)')
    lines.append('')

    scanned_pages = [p['page_num'] for p in all_pages if p['is_scanned']]
    table_pages   = [p['page_num'] for p in all_pages
                     if any(pa['type'] == 'TABLE_BACKED' for pa in p['paragraphs'])]
    text_pages    = [p['page_num'] for p in all_pages
                     if all(pa['type'] == 'TEXT_ONLY' for pa in p['paragraphs'])
                     and p['paragraphs']]
    total_units   = sum(len(u) for p in all_pages
                        for pa in p['paragraphs']
                        for u in [pa['units']])
    table_units   = sum(len(pa['units']) for p in all_pages
                        for pa in p['paragraphs']
                        if pa['type'] == 'TABLE_BACKED')
    text_units    = total_units - table_units

    lines.append('  Statistics:')
    lines.append(f'    - Total pages processed: {total_pages}')
    lines.append(f'    - Scanned pages (OCR): {scanned_pages or "none"}')
    lines.append(f'    - Pages with tables: {table_pages or "none"}')
    lines.append(f'    - Text-only pages: {text_pages or "none"}')
    lines.append(f'    - Total Information Units: {total_units}')
    lines.append(f'    - Table-backed units: {table_units}')
    lines.append(f'    - Text-only units: {text_units}')
    lines.append('')
    lines.append('=' * 50)

    return '\n'.join(lines)


# ====================================================================
#  MAIN
# ====================================================================
def main():
    pdf_file = 'el-bankalahly .pdf'
    output_file = OUTPUT_FILE

    print('=' * 60)
    print('  ARABIC PDF — Document Understanding Layer v8')
    print('  Information Unit Extraction')
    print('=' * 60)
    print(f'\n  Input:  {pdf_file}')
    print(f'  Output: {output_file}\n')

    if not os.path.exists(pdf_file):
        print(f'[ERROR] PDF not found: {pdf_file}')
        sys.exit(1)

    try:
        doc   = fitz.open(pdf_file)
        total = len(doc)
        doc.close()
    except Exception as exc:
        print(f'[ERROR] Cannot open PDF: {exc}')
        return

    print(f'  Pages: {total}\n')
    _get_layout_predictor()
    print()

    all_pages = []
    for pn in range(total):
        print(f'  Page {pn + 1:>2}/{total} …', end=' ', flush=True)
        page_data = process_page(pdf_file, pn)
        all_pages.append(page_data)

        n_para  = len(page_data['paragraphs'])
        n_table = sum(1 for p in page_data['paragraphs']
                      if p['type'] == 'TABLE_BACKED')
        n_text  = n_para - n_table
        n_units = sum(len(p['units']) for p in page_data['paragraphs'])
        noise   = page_data.get('noise_skipped', 0)
        src     = 'OCR' if page_data['is_scanned'] else 'native'

        print(f'[{src}] {n_para} paragraphs '
              f'(T:{n_table} P:{n_text}) '
              f'{n_units} units '
              f'(noise:{noise}) ✓')

    # ── Write output ─────────────────────────────────────────────────
    output_text = format_output(all_pages, os.path.basename(pdf_file), total)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text)

    # ── Summary ──────────────────────────────────────────────────────
    total_units = sum(len(u) for p in all_pages
                      for pa in p['paragraphs']
                      for u in [pa['units']])
    table_units = sum(len(pa['units']) for p in all_pages
                      for pa in p['paragraphs']
                      if pa['type'] == 'TABLE_BACKED')

    print(f"\n{'=' * 60}")
    print(f'  ✓ Output: {output_file}')
    print(f'  ✓ Pages: {total}')
    print(f'  ✓ Total Information Units: {total_units}')
    print(f'  ✓ Table-backed units: {table_units}')
    print(f'  ✓ Text-only units: {total_units - table_units}')
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
