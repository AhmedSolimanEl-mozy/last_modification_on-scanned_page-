#!/usr/bin/env python3
"""
table_segment.py — Page Segmentation: TABLE_BLOCK vs TEXT_BLOCK
================================================================

Architectural decision: EXPLICIT ROUTING by page type.

  IF page has native text layer (PyMuPDF word count > threshold):
      → Native-text band analysis  (PyMuPDF word-gap detection)
  ELSE (scanned / image-only page):
      → CV-based detection from detect_tables_cv.py
        (morphological lines + adaptive threshold band analysis
         + numeric cluster heuristic)

No fallbacks. No mixing signals. Each path emits the same schema:
    PageRegion(type, bbox, confidence, detection_method, ...)

Downstream modules (table_ocr, table_reconstruct, table_render)
do NOT know or care how the table was detected.

Usage:
    from table_segment import segment_page, PageRegion
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import cv2
import fitz
import numpy as np

# Import CV-scan detection from detect_tables_cv.py
# This is the canonical scanned-page table detector.
from detect_tables_cv import (
    detect_lined_tables   as cv_detect_lined_tables,
    detect_tables_cv_scan as cv_detect_tables_cv_scan,
    PX_TO_PT              as CV_PX_TO_PT,
)


# ────────────────────────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────────────────────────
DPI           = 300
PX_TO_PT      = 72.0 / DPI
PT_TO_PX      = DPI / 72.0

# ── Native path: word-band analysis ─────────────────────────────────
BAND_PT       = 6
BIG_GAP_PT    = 20
SPAN_RATIO    = 0.30
CLUSTER_TOL   = 10
MIN_TAB_BANDS = 4
HEADER_EXTEND = 5
FIN_NUM_RE    = re.compile(r'[\d,\.\(\)٠-٩]{2,}')

# ── Scanned path: CV detection thresholds (overrides for tolerance) ─
# These are used by the NEW numeric-cluster heuristic below.
NUM_CLUSTER_MIN_COLS    = 2       # ≥2 vertically-aligned numeric groups
NUM_CLUSTER_MIN_ROWS    = 3       # across ≥3 Y-bands
NUM_CLUSTER_X_TOL_PX    = 40      # X-center tolerance for alignment (px)
NUM_CLUSTER_BAND_PX     = 40      # Y-band height (px)
NUM_CLUSTER_MIN_FRAG_W  = 15      # minimum fragment width (px)

# ── Page type threshold ─────────────────────────────────────────────
NATIVE_WORD_THRESHOLD   = 5       # ≥5 native words → native path


# ────────────────────────────────────────────────────────────────────
#  Data model  (shared schema for ALL detection methods)
# ────────────────────────────────────────────────────────────────────
@dataclass
class PageRegion:
    """Normalized output from any detection method.
    Downstream modules consume this without caring about `source`."""
    region_type: str            # "TABLE_BLOCK" | "TEXT_BLOCK"
    bbox: List[float]           # [x0, y0, x1, y1] in PDF points
    confidence: float = 1.0
    detection_method: str = ""  # "native_band" | "cv_scan" | "cv_morph" | "cv_numcluster"
    has_grid_lines: bool = False
    numeric_density: float = 0.0
    page_number: int = 0


# ────────────────────────────────────────────────────────────────────
#  NATIVE PATH: Word-band gap analysis (PyMuPDF, borderless tables)
# ────────────────────────────────────────────────────────────────────
def detect_tables_native_bands(fitz_page: fitz.Page) -> List[Tuple[float, float, float, float]]:
    """Detect borderless tables via word-band gap analysis on native pages.
    Returns list of (x0, y0, x1, y1) in PDF points."""
    pw = fitz_page.rect.width
    ph = fitz_page.rect.height
    words = fitz_page.get_text('words')
    if not words:
        return []

    row_bands: dict = {}
    for w in words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        band = int(y0 // BAND_PT)
        row_bands.setdefault(band, []).append((x0, y0, x1, y1, text))

    band_info: dict = {}
    for band in sorted(row_bands.keys()):
        items = row_bands[band]
        if len(items) < 2:
            continue
        sorted_items = sorted(items, key=lambda it: it[0])
        big_gaps = 0
        for i in range(1, len(sorted_items)):
            gap = sorted_items[i][0] - sorted_items[i - 1][2]
            if gap > BIG_GAP_PT:
                big_gaps += 1
        has_num = any(FIN_NUM_RE.search(it[4]) for it in items)
        span = sorted_items[-1][2] - sorted_items[0][0]
        band_info[band] = dict(big_gaps=big_gaps, has_num=has_num,
                                span=span, n_words=len(items))

    tabular = sorted(
        b for b, info in band_info.items()
        if info['big_gaps'] >= 2 and info['has_num']
        and info['span'] > pw * SPAN_RATIO
    )
    if not tabular:
        return []

    clusters = [[tabular[0]]]
    for b in tabular[1:]:
        if b - clusters[-1][-1] <= CLUSTER_TOL:
            clusters[-1].append(b)
        else:
            clusters.append([b])

    tables = []
    for cluster in clusters:
        if len(cluster) < MIN_TAB_BANDS:
            continue
        first_tab, last_tab = cluster[0], cluster[-1]

        start = first_tab
        for b in range(first_tab - 1, max(first_tab - HEADER_EXTEND - 1, 0), -1):
            if b in band_info and band_info[b]['span'] > pw * 0.2:
                start = b
            else:
                break

        end = last_tab
        for b in range(last_tab + 1, last_tab + HEADER_EXTEND + 1):
            if b in band_info and band_info[b]['span'] > pw * 0.2:
                end = b
            else:
                break

        all_words = []
        for b in range(start, end + 1):
            all_words.extend(row_bands.get(b, []))
        if not all_words:
            continue

        x0 = min(w[0] for w in all_words)
        x1 = max(w[2] for w in all_words)
        y0 = max(0, start * BAND_PT - BAND_PT)
        y1 = min(ph, (end + 1) * BAND_PT + BAND_PT)
        tables.append((round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)))

    return tables


# ────────────────────────────────────────────────────────────────────
#  SCANNED PATH: Numeric cluster heuristic (Part 3 requirement)
# ────────────────────────────────────────────────────────────────────
def detect_numeric_clusters_cv(page_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect table candidates on scanned pages by finding vertically
    aligned numeric-like fragment clusters.

    Heuristic: if ≥NUM_CLUSTER_MIN_COLS groups of fragments align
    vertically (same X ± tolerance) across ≥NUM_CLUSTER_MIN_ROWS
    Y-bands, that region is a table candidate.

    Returns list of (x, y, w, h) in pixel coords.
    """
    height, width = page_img.shape[:2]
    gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold → binary
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10,
    )

    # Dilate to merge characters into word fragments
    wk = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    dilated = cv2.dilate(adapt, wk, iterations=1)

    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    frags = [cv2.boundingRect(c) for c in cnts
             if cv2.boundingRect(c)[2] > NUM_CLUSTER_MIN_FRAG_W]

    if len(frags) < 4:
        return []

    # Group fragments into Y-bands
    band_frags: dict = {}
    for (fx, fy, fw, fh) in frags:
        band_idx = fy // NUM_CLUSTER_BAND_PX
        band_frags.setdefault(band_idx, []).append((fx, fy, fw, fh))

    # For each Y-band, compute X-centers of fragments
    band_xcenters: dict = {}
    for band_idx, bfrags in band_frags.items():
        if len(bfrags) < 2:
            continue
        xcenters = sorted((fx + fw / 2) for (fx, fy, fw, fh) in bfrags)
        band_xcenters[band_idx] = xcenters

    if len(band_xcenters) < NUM_CLUSTER_MIN_ROWS:
        return []

    # Find X-positions that appear consistently across many bands.
    # Collect all X-centers, then cluster them.
    all_xc = []
    for band_idx, xcs in band_xcenters.items():
        for xc in xcs:
            all_xc.append((xc, band_idx))

    if not all_xc:
        return []

    # Cluster X-centers across bands
    all_xc.sort(key=lambda t: t[0])
    x_clusters: list = [[all_xc[0]]]
    for xc, bi in all_xc[1:]:
        if abs(xc - x_clusters[-1][-1][0]) <= NUM_CLUSTER_X_TOL_PX:
            x_clusters[-1].append((xc, bi))
        else:
            x_clusters.append([(xc, bi)])

    # A valid "column" has fragments in ≥ MIN_ROWS distinct bands
    valid_columns = []
    for cluster in x_clusters:
        distinct_bands = len(set(bi for _, bi in cluster))
        if distinct_bands >= NUM_CLUSTER_MIN_ROWS:
            valid_columns.append(cluster)

    if len(valid_columns) < NUM_CLUSTER_MIN_COLS:
        return []

    # Merge all valid column fragments into a single table bbox
    all_involved_frags = []
    involved_bands = set()
    for col in valid_columns:
        for xc, bi in col:
            involved_bands.add(bi)

    for bi in involved_bands:
        all_involved_frags.extend(band_frags.get(bi, []))

    if not all_involved_frags:
        return []

    x0 = min(fx for fx, fy, fw, fh in all_involved_frags)
    y0 = min(fy for fx, fy, fw, fh in all_involved_frags)
    x1 = max(fx + fw for fx, fy, fw, fh in all_involved_frags)
    y1 = max(fy + fh for fx, fy, fw, fh in all_involved_frags)

    # Confidence based on column count and row consistency
    max_band_count = max(len(set(bi for _, bi in col)) for col in valid_columns)
    total_bands = len(band_xcenters)
    row_consistency = max_band_count / max(total_bands, 1)
    confidence = min(0.5 + 0.3 * len(valid_columns) / 5 + 0.2 * row_consistency, 0.95)

    return [(x0, y0, x1 - x0, y1 - y0)]


# ────────────────────────────────────────────────────────────────────
#  Numeric density analysis
# ────────────────────────────────────────────────────────────────────
def compute_numeric_density(fitz_page: fitz.Page,
                            bbox_pt: List[float]) -> float:
    """Compute fraction of words in the bbox that are numeric/financial."""
    x0, y0, x1, y1 = bbox_pt
    clip = fitz.Rect(x0, y0, x1, y1)
    words = fitz_page.get_text('words', clip=clip)
    if not words:
        return 0.0
    num_count = sum(1 for w in words if FIN_NUM_RE.search(w[4]))
    return num_count / len(words)


# ────────────────────────────────────────────────────────────────────
#  Master segmentation function
# ────────────────────────────────────────────────────────────────────
def segment_page(
    doc: fitz.Document,
    page_idx: int,
    page_img_bgr: np.ndarray,
    use_surya_layout: bool = False,   # unused, kept for API compat
) -> List[PageRegion]:
    """
    Segment a single page into TABLE_BLOCK and TEXT_BLOCK regions.

    ARCHITECTURAL DECISION: Explicit routing by page type.

      IF page has native text (≥ NATIVE_WORD_THRESHOLD words):
          → Native-text band analysis only.
      ELSE (scanned / image-only):
          → CV-based detection: morphological lines + band analysis
            + numeric cluster heuristic.

    No fallbacks between paths. No mixing signals.
    Both paths emit identical PageRegion schema.
    Downstream modules do not care which path ran.

    Returns list of PageRegion sorted top→bottom.
    """
    fitz_page = doc[page_idx]
    page_num = page_idx + 1
    pw_pt = fitz_page.rect.width
    ph_pt = fitz_page.rect.height
    has_native = len(fitz_page.get_text('words')) > NATIVE_WORD_THRESHOLD

    table_regions: List[PageRegion] = []

    if has_native:
        # ════════════════════════════════════════════════════════════
        #  NATIVE PATH: PyMuPDF word-band analysis
        # ════════════════════════════════════════════════════════════
        band_tables = detect_tables_native_bands(fitz_page)
        for (x0, y0, x1, y1) in band_tables:
            nd = compute_numeric_density(fitz_page, [x0, y0, x1, y1])
            table_regions.append(PageRegion(
                region_type="TABLE_BLOCK",
                bbox=[x0, y0, x1, y1],
                confidence=0.85,
                detection_method="native_band",
                has_grid_lines=False,
                numeric_density=round(nd, 3),
                page_number=page_num,
            ))

    else:
        # ════════════════════════════════════════════════════════════
        #  SCANNED PATH: CV-based detection (3 sub-layers)
        # ════════════════════════════════════════════════════════════

        # Sub-layer A: Morphological grid lines (bordered tables)
        grid_boxes_px = cv_detect_lined_tables(page_img_bgr)
        for (x, y, w, h) in grid_boxes_px:
            bbox_pt = [
                round(x * CV_PX_TO_PT, 2),
                round(y * CV_PX_TO_PT, 2),
                round((x + w) * CV_PX_TO_PT, 2),
                round((y + h) * CV_PX_TO_PT, 2),
            ]
            table_regions.append(PageRegion(
                region_type="TABLE_BLOCK",
                bbox=bbox_pt,
                confidence=0.95,
                detection_method="cv_morph",
                has_grid_lines=True,
                numeric_density=0.5,
                page_number=page_num,
            ))

        # Sub-layer B: Adaptive threshold + band analysis (borderless)
        if not table_regions:
            cv_scan_boxes_px = cv_detect_tables_cv_scan(page_img_bgr)
            for (x, y, w, h) in cv_scan_boxes_px:
                bbox_pt = [
                    round(x * CV_PX_TO_PT, 2),
                    round(y * CV_PX_TO_PT, 2),
                    round((x + w) * CV_PX_TO_PT, 2),
                    round((y + h) * CV_PX_TO_PT, 2),
                ]
                if not _overlaps_existing(table_regions, bbox_pt):
                    table_regions.append(PageRegion(
                        region_type="TABLE_BLOCK",
                        bbox=bbox_pt,
                        confidence=0.75,
                        detection_method="cv_scan",
                        has_grid_lines=False,
                        numeric_density=0.5,
                        page_number=page_num,
                    ))

        # Sub-layer C: Numeric cluster heuristic (catch remaining tables)
        if not table_regions:
            nclust_boxes_px = detect_numeric_clusters_cv(page_img_bgr)
            for (x, y, w, h) in nclust_boxes_px:
                bbox_pt = [
                    round(x * CV_PX_TO_PT, 2),
                    round(y * CV_PX_TO_PT, 2),
                    round((x + w) * CV_PX_TO_PT, 2),
                    round((y + h) * CV_PX_TO_PT, 2),
                ]
                if not _overlaps_existing(table_regions, bbox_pt):
                    table_regions.append(PageRegion(
                        region_type="TABLE_BLOCK",
                        bbox=bbox_pt,
                        confidence=0.65,
                        detection_method="cv_numcluster",
                        has_grid_lines=False,
                        numeric_density=0.5,
                        page_number=page_num,
                    ))

    # ── Build TEXT_BLOCK regions for non-table areas ────────────────
    text_regions = _compute_text_regions(
        table_regions, pw_pt, ph_pt, page_num
    )

    all_regions = table_regions + text_regions
    all_regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
    return all_regions


def _overlaps_existing(regions: List[PageRegion], bbox: List[float],
                       threshold: float = 0.3) -> bool:
    """Check if bbox overlaps any existing region by >= threshold IoU."""
    for r in regions:
        iou = _iou(r.bbox, bbox)
        if iou > threshold:
            return True
    return False


def _iou(a: List[float], b: List[float]) -> float:
    """Intersection over Union of two [x0,y0,x1,y1] bboxes."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _compute_text_regions(
    table_regions: List[PageRegion],
    page_w: float,
    page_h: float,
    page_num: int,
) -> List[PageRegion]:
    """Compute TEXT_BLOCK regions as the complement of TABLE_BLOCK regions.

    Simple approach: vertical strips between table regions and page edges.
    """
    if not table_regions:
        # Whole page is text
        return [PageRegion(
            region_type="TEXT_BLOCK",
            bbox=[0, 0, page_w, page_h],
            confidence=1.0,
            detection_method="complement",
            page_number=page_num,
        )]

    text_regions = []
    # Sort tables by y0
    sorted_tables = sorted(table_regions, key=lambda r: r.bbox[1])

    # Region above first table
    if sorted_tables[0].bbox[1] > 10:
        text_regions.append(PageRegion(
            region_type="TEXT_BLOCK",
            bbox=[0, 0, page_w, sorted_tables[0].bbox[1]],
            confidence=1.0,
            detection_method="complement",
            page_number=page_num,
        ))

    # Regions between tables
    for i in range(len(sorted_tables) - 1):
        gap_top = sorted_tables[i].bbox[3]
        gap_bot = sorted_tables[i + 1].bbox[1]
        if gap_bot - gap_top > 10:
            text_regions.append(PageRegion(
                region_type="TEXT_BLOCK",
                bbox=[0, gap_top, page_w, gap_bot],
                confidence=1.0,
                detection_method="complement",
                page_number=page_num,
            ))

    # Region below last table
    if sorted_tables[-1].bbox[3] < page_h - 10:
        text_regions.append(PageRegion(
            region_type="TEXT_BLOCK",
            bbox=[0, sorted_tables[-1].bbox[3], page_w, page_h],
            confidence=1.0,
            detection_method="complement",
            page_number=page_num,
        ))

    return text_regions
