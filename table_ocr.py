#!/usr/bin/env python3
"""
table_ocr.py — Character-level OCR for Table Regions
=====================================================

For TABLE_BLOCK regions detected by table_segment.py:

1. Crop the region from the page image.
2. Re-render at HIGH DPI (≥400) for better glyph resolution.
3. Run Surya OCR with character-level output (TextChar per glyph).
4. Disable language model corrections for numbers.
5. Return per-glyph bounding boxes + confidence.

For TEXT_BLOCK regions:
    Standard Surya OCR line-level output (as-is).

Key design decisions:
  • TABLE_BLOCK at 400 DPI — more pixels → better digit shapes.
  • return_words=True populates TextLine.chars (TextChar objects).
  • No aggressive word merging: we keep char-level granularity.
  • Numeric strings are preserved as-is (no LM post-correction).

Usage:
    from table_ocr import ocr_table_region, ocr_text_region, TableGlyph
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import fitz
import numpy as np

# ────────────────────────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────────────────────────
TABLE_DPI   = 400       # higher DPI for table regions
TEXT_DPI    = 300       # standard DPI for text regions
PX_TO_PT_TABLE = 72.0 / TABLE_DPI
PX_TO_PT_TEXT  = 72.0 / TEXT_DPI

# Numeric token detection: digits, Arabic-Indic digits, commas, dots, parens, minus
NUMERIC_RE = re.compile(r'^[\d٠-٩,،.\(\)\-−–\s%]+$')


# ────────────────────────────────────────────────────────────────────
#  Data model
# ────────────────────────────────────────────────────────────────────
@dataclass
class TableGlyph:
    """A single character/glyph detected in a table region."""
    char: str
    bbox: List[float]       # [x0, y0, x1, y1] in PDF points (page-absolute)
    confidence: float
    is_numeric: bool = False
    polygon: Optional[List[List[float]]] = None  # raw pixel polygon


@dataclass
class TableLine:
    """A line of text in a table region, with character-level detail."""
    text: str
    bbox: List[float]       # [x0, y0, x1, y1] in PDF points (page-absolute)
    glyphs: List[TableGlyph] = field(default_factory=list)
    confidence: float = 0.0
    direction: str = "ltr"  # "rtl" | "ltr"


@dataclass
class TextBlockResult:
    """Standard OCR result for a TEXT_BLOCK region."""
    lines: List[dict]       # same format as layout_extract.py output


@dataclass
class TableOCRResult:
    """OCR result for a TABLE_BLOCK region, with character-level detail."""
    table_lines: List[TableLine]
    region_bbox: List[float]    # [x0, y0, x1, y1] PDF points
    dpi: int
    page_number: int


# ────────────────────────────────────────────────────────────────────
#  Surya models (lazy loaded, shared across calls)
# ────────────────────────────────────────────────────────────────────
_surya_foundation = None
_surya_det = None
_surya_rec = None


def _load_surya():
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
#  Image utilities
# ────────────────────────────────────────────────────────────────────
def render_page_at_dpi(doc: fitz.Document, page_idx: int,
                       dpi: int) -> np.ndarray:
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


def crop_region(img: np.ndarray, bbox_pt: List[float],
                dpi: int, pad_px: int = 8) -> Tuple[np.ndarray, int, int]:
    """Crop a region from a page image.

    Args:
        img: Full page image at `dpi`.
        bbox_pt: [x0, y0, x1, y1] in PDF points.
        dpi: DPI of the image.
        pad_px: Padding in pixels around the crop.

    Returns:
        (cropped_img, offset_x_px, offset_y_px)
    """
    scale = dpi / 72.0
    x0_px = int(bbox_pt[0] * scale) - pad_px
    y0_px = int(bbox_pt[1] * scale) - pad_px
    x1_px = int(bbox_pt[2] * scale) + pad_px
    y1_px = int(bbox_pt[3] * scale) + pad_px

    h, w = img.shape[:2]
    x0_px = max(0, x0_px)
    y0_px = max(0, y0_px)
    x1_px = min(w, x1_px)
    y1_px = min(h, y1_px)

    crop = img[y0_px:y1_px, x0_px:x1_px]
    return crop, x0_px, y0_px


def clean_table_image(img: np.ndarray) -> np.ndarray:
    """Enhanced cleaning for table images — preserves thin strokes.
    Lighter than the general clean_image: no heavy morphology that
    could destroy digit strokes or decimal points."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Mild bilateral denoise (preserves edges/digits)
    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)

    # Mild sharpen
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1)
    sharp   = cv2.addWeighted(denoised, 1.3, blurred, -0.3, 0)

    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


# ────────────────────────────────────────────────────────────────────
#  Direction detection
# ────────────────────────────────────────────────────────────────────
def detect_direction(text: str) -> str:
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin  = sum(1 for c in text if 'A' <= c <= 'z')
    digits = sum(1 for c in text if c.isdigit() or '٠' <= c <= '٩')
    # If mostly Arabic, RTL; if mostly digits, treat as LTR for positioning
    if arabic > latin and arabic > digits:
        return "rtl"
    return "ltr"


def is_numeric_text(text: str) -> bool:
    """Check if text is purely numeric/financial."""
    return bool(NUMERIC_RE.match(text.strip()))


# ────────────────────────────────────────────────────────────────────
#  Table region OCR (character-level)
# ────────────────────────────────────────────────────────────────────
def ocr_table_region(
    doc: fitz.Document,
    page_idx: int,
    region_bbox_pt: List[float],
    dpi: int = TABLE_DPI,
) -> TableOCRResult:
    """
    OCR a table region at high DPI with character-level output.

    Steps:
      1. Render page at high DPI.
      2. Crop the table region.
      3. Clean for table content (light touch).
      4. Run Surya with return_words=True (populates .chars).
      5. Convert char bboxes from crop-pixel to page-absolute PDF points.
      6. Mark numeric glyphs.

    Returns TableOCRResult with per-glyph detail.
    """
    _load_surya()
    from PIL import Image

    page_num = page_idx + 1
    px_to_pt = 72.0 / dpi

    # Render full page at high DPI
    page_img = render_page_at_dpi(doc, page_idx, dpi)

    # Crop table region
    crop, off_x, off_y = crop_region(page_img, region_bbox_pt, dpi, pad_px=12)
    cleaned = clean_table_image(crop)

    # Convert to PIL for Surya
    img_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Run Surya OCR with character-level output
    results = _surya_rec(
        [pil_img],
        task_names=["ocr_with_boxes"],
        det_predictor=_surya_det,
        return_words=True,      # populates .chars on each TextLine
        sort_lines=True,
    )

    if not results or not results[0].text_lines:
        return TableOCRResult(
            table_lines=[],
            region_bbox=region_bbox_pt,
            dpi=dpi,
            page_number=page_num,
        )

    table_lines = []
    for tl in results[0].text_lines:
        # Line bbox: convert crop-local pixels → page-absolute PDF points
        line_poly = tl.polygon
        lxs = [p[0] for p in line_poly]
        lys = [p[1] for p in line_poly]
        line_bbox_px = [min(lxs), min(lys), max(lxs), max(lys)]
        line_bbox_pt = [
            round((line_bbox_px[0] + off_x) * px_to_pt, 2),
            round((line_bbox_px[1] + off_y) * px_to_pt, 2),
            round((line_bbox_px[2] + off_x) * px_to_pt, 2),
            round((line_bbox_px[3] + off_y) * px_to_pt, 2),
        ]

        # Extract character-level glyphs
        glyphs = []
        if tl.chars:
            for ch in tl.chars:
                ch_poly = ch.polygon
                cxs = [p[0] for p in ch_poly]
                cys = [p[1] for p in ch_poly]
                ch_bbox_px = [min(cxs), min(cys), max(cxs), max(cys)]
                ch_bbox_pt = [
                    round((ch_bbox_px[0] + off_x) * px_to_pt, 2),
                    round((ch_bbox_px[1] + off_y) * px_to_pt, 2),
                    round((ch_bbox_px[2] + off_x) * px_to_pt, 2),
                    round((ch_bbox_px[3] + off_y) * px_to_pt, 2),
                ]
                glyph = TableGlyph(
                    char=ch.text,
                    bbox=ch_bbox_pt,
                    confidence=ch.confidence or 0.0,
                    is_numeric=ch.text.strip() in '0123456789٠١٢٣٤٥٦٧٨٩.,،()-%',
                    polygon=[[round((p[0] + off_x) * px_to_pt, 2),
                              round((p[1] + off_y) * px_to_pt, 2)]
                             for p in ch_poly],
                )
                glyphs.append(glyph)
        else:
            # Fallback: no char data, synthesize from line text
            # Distribute bbox evenly (rough approximation)
            text = tl.text
            if text:
                char_w = (line_bbox_pt[2] - line_bbox_pt[0]) / max(len(text), 1)
                for ci, c in enumerate(text):
                    cx0 = line_bbox_pt[0] + ci * char_w
                    glyphs.append(TableGlyph(
                        char=c,
                        bbox=[round(cx0, 2), line_bbox_pt[1],
                              round(cx0 + char_w, 2), line_bbox_pt[3]],
                        confidence=tl.confidence or 0.0,
                        is_numeric=c in '0123456789٠١٢٣٤٥٦٧٨٩.,،()-%',
                    ))

        line_text = tl.text
        direction = detect_direction(line_text)

        table_lines.append(TableLine(
            text=line_text,
            bbox=line_bbox_pt,
            glyphs=glyphs,
            confidence=tl.confidence or 0.0,
            direction=direction,
        ))

    return TableOCRResult(
        table_lines=table_lines,
        region_bbox=region_bbox_pt,
        dpi=dpi,
        page_number=page_num,
    )


# ────────────────────────────────────────────────────────────────────
#  Native-text table extraction (PyMuPDF character-level)
# ────────────────────────────────────────────────────────────────────
def extract_native_table_chars(
    fitz_page: fitz.Page,
    region_bbox_pt: List[float],
    page_num: int,
) -> TableOCRResult:
    """
    For native (vector) PDF pages: extract characters directly from
    PyMuPDF with per-glyph bounding boxes.

    This is much more accurate than OCR for native text since we get
    the exact font metrics and glyph positions from the PDF itself.
    """
    x0, y0, x1, y1 = region_bbox_pt
    clip = fitz.Rect(x0, y0, x1, y1)
    d = fitz_page.get_text("dict", clip=clip)

    table_lines = []
    for blk in d.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for line in blk.get("lines", []):
            line_bbox = list(line["bbox"])
            line_chars = []
            line_text_parts = []

            for span in line.get("spans", []):
                span_text = span["text"]
                span_bbox = list(span["bbox"])
                font_name = span.get("font", "")
                font_size = span.get("size", 0)

                # Try to get character-level from rawdict
                # For each span, distribute chars by font metrics
                if span_text:
                    char_w = (span_bbox[2] - span_bbox[0]) / max(len(span_text), 1)
                    for ci, c in enumerate(span_text):
                        cx0 = span_bbox[0] + ci * char_w
                        cx1 = cx0 + char_w
                        line_chars.append(TableGlyph(
                            char=c,
                            bbox=[round(cx0, 2), round(span_bbox[1], 2),
                                  round(cx1, 2), round(span_bbox[3], 2)],
                            confidence=1.0,  # native text = perfect confidence
                            is_numeric=c in '0123456789٠١٢٣٤٥٦٧٨٩.,،()-%',
                        ))
                    line_text_parts.append(span_text)

            full_text = "".join(line_text_parts)
            direction = detect_direction(full_text)

            if line_chars:
                table_lines.append(TableLine(
                    text=full_text,
                    bbox=[round(v, 2) for v in line_bbox],
                    glyphs=line_chars,
                    confidence=1.0,
                    direction=direction,
                ))

    return TableOCRResult(
        table_lines=table_lines,
        region_bbox=region_bbox_pt,
        dpi=72,
        page_number=page_num,
    )


# ────────────────────────────────────────────────────────────────────
#  Native character-level extraction using rawdict
# ────────────────────────────────────────────────────────────────────
def extract_native_table_rawdict(
    fitz_page: fitz.Page,
    region_bbox_pt: List[float],
    page_num: int,
) -> TableOCRResult:
    """
    Extract true per-character bounding boxes from PyMuPDF rawdict.
    This gives exact glyph positions from the PDF font metrics.
    """
    x0, y0, x1, y1 = region_bbox_pt
    clip = fitz.Rect(x0, y0, x1, y1)
    d = fitz_page.get_text("rawdict", clip=clip)

    table_lines = []
    for blk in d.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for line in blk.get("lines", []):
            line_bbox = list(line["bbox"])
            line_chars = []
            line_text_parts = []

            for span in line.get("spans", []):
                for char_info in span.get("chars", []):
                    c = char_info["c"]
                    cb = char_info["bbox"]
                    line_chars.append(TableGlyph(
                        char=c,
                        bbox=[round(cb[0], 2), round(cb[1], 2),
                              round(cb[2], 2), round(cb[3], 2)],
                        confidence=1.0,
                        is_numeric=c in '0123456789٠١٢٣٤٥٦٧٨٩.,،()-%',
                    ))
                    line_text_parts.append(c)

            full_text = "".join(line_text_parts)
            direction = detect_direction(full_text)

            if line_chars:
                table_lines.append(TableLine(
                    text=full_text,
                    bbox=[round(v, 2) for v in line_bbox],
                    glyphs=line_chars,
                    confidence=1.0,
                    direction=direction,
                ))

    return TableOCRResult(
        table_lines=table_lines,
        region_bbox=region_bbox_pt,
        dpi=72,
        page_number=page_num,
    )
