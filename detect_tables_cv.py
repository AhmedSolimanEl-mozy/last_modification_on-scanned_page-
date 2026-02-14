#!/usr/bin/env python3
"""
detect_tables_cv.py — Hybrid Table Detection Pipeline
======================================================
Three-layer detection strategy:

  Layer 1 — Morphological line isolation  (bordered tables)
      Otsu → horizontal kernel (w/40, 1) + vertical kernel (1, h/40)
      → cv2.add() → contour-based bounding boxes

  Layer 2 — Native-text band analysis  (borderless / digital-text pages)
      PyMuPDF word extraction → 6 pt row-bands → gap classification
      → tabular-band clustering → header/footer extension

  Layer 3 — OpenCV band analysis  (scanned pages with no native text)
      Adaptive threshold → word-fragment dilation → row-band gap
      → column-alignment filtering

Output:
  table_extraction/
  ├── crops/      table_001.png, table_002.png, …
  ├── masks/      page_04_mask.png, … (semi-transparent blue + green border)
  └── spatial_data.json

Coordinate math:
  • All detection in pixel space at 300 DPI
  • PDF Points = Pixels × (72 / 300)
  • Both units stored in JSON for downstream text injection

Usage:
    python detect_tables_cv.py [path/to/file.pdf]
"""

import os
import sys
import json
import re
import time

import cv2
import fitz  # PyMuPDF
import numpy as np
from pdf2image import convert_from_path

# ====================================================================
#  Configuration
# ====================================================================
DPI                = 300            # render resolution — DO NOT change
PX_TO_PT           = 72.0 / DPI    # pixel → PDF-point conversion factor
PT_TO_PX           = DPI / 72.0    # PDF-point → pixel conversion factor

# ── Layer 1: morphological line detection ────────────────────────────
HORIZ_KERNEL_DIV   = 50            # horizontal kernel = (width / DIV, 1)  (was 40, loosened for faded lines)
VERT_KERNEL_DIV    = 50            # vertical kernel   = (1, height / DIV)  (was 40, loosened for faded lines)
MIN_CONTOUR_W_PX   = 150           # ignore contours narrower  (px)  (was 200, loosened for partial borders)
MIN_CONTOUR_H_PX   = 80            # ignore contours shorter   (px)  (was 100, loosened for partial borders)

# ── Layer 2: native-text band analysis (PyMuPDF) ────────────────────
BAND_PT            = 6             # row-band height in PDF points
BIG_GAP_PT         = 20            # gap qualifying as "big" (pt)
SPAN_RATIO         = 0.30          # band must span > 30% page width
CLUSTER_TOL        = 10            # max band-index gap within cluster
MIN_TABULAR_BANDS  = 4             # minimum tabular bands per cluster
HEADER_EXTEND      = 5             # bands to extend for headers
FIN_NUM_RE         = re.compile(r'[\d,\.\(\)٠-٩]{2,}')

# ── Layer 3: OpenCV band analysis (scanned pages) ───────────────────
CV_BAND_PX         = 40            # Y-band height in pixels
CV_WORD_KERNEL     = (25, 3)       # dilation to merge chars → word frags
CV_MIN_FRAG_W      = 15            # ignore tiny noise fragments (px)  (was 20, loosened)
CV_BIG_GAP_PX      = 50            # gap qualifying as "big" (px)  (was 80, loosened for tighter scanned tables)
CV_MAX_GAP_PX      = 120           # min max-gap for table band (px)  (was 200, loosened for tighter columns)
CV_SPREAD_RATIO    = 0.15          # fragments must span > 15% page width  (was 0.20, loosened)
CV_RUN_GAP_TOL     = 4             # max band-index gap within a run  (was 3, allow more scan noise)
CV_MIN_RUN_LEN     = 3             # minimum bands in a valid run  (was 5, loosened for smaller tables)
CV_MULTI_COL_RATIO = 0.30          # ≥30% of bands must have ≥2 big gaps  (was 0.40, loosened)

# ── Rendering ────────────────────────────────────────────────────────
MASK_COLOR_BGR     = (255, 0, 0)   # semi-transparent blue overlay
MASK_ALPHA         = 0.35          # overlay opacity
BORDER_COLOR_BGR   = (0, 255, 0)   # bright green verification border
BORDER_THICKNESS   = 3             # border line width (px)
CROP_PAD           = 8             # padding around crop bbox (px)

OUTPUT_ROOT        = "table_extraction"


# ====================================================================
#  Layer 1: Morphological line isolation
# ====================================================================
def detect_lined_tables(page_img: np.ndarray):
    """
    Detect table bounding boxes via horizontal/vertical line detection.
    Works for tables with drawn grid lines / borders.

    Returns list of (x, y, w, h) in pixel coordinates.
    """
    height, width = page_img.shape[:2]

    gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    h_len = max(width  // HORIZ_KERNEL_DIV, 1)
    v_len = max(height // VERT_KERNEL_DIV,  1)

    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    h_lines = cv2.dilate(cv2.erode(binary, h_kern), h_kern)
    v_lines = cv2.dilate(cv2.erode(binary, v_kern), v_kern)

    mask = cv2.add(h_lines, v_lines)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, close_k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= MIN_CONTOUR_W_PX and h >= MIN_CONTOUR_H_PX:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


# ====================================================================
#  Layer 2: Native-text band analysis (PyMuPDF)
# ====================================================================
def detect_tables_native(fitz_page):
    """
    Detect tables using PyMuPDF word-level extraction.

    Groups words into 6 pt row-bands.  A band is "tabular" when it has
    ≥2 big gaps (>20 pt) between words, contains financial numbers,
    and spans >30% of the page width.

    Nearby tabular bands are clustered (tolerance 10 bands = 60 pt).
    Clusters with ≥4 tabular bands are accepted as tables, then
    extended by up to 5 bands above/below to capture headers.

    Returns list of (x, y, w, h) in PDF points, or empty list.
    """
    pw = fitz_page.rect.width
    ph = fitz_page.rect.height
    words = fitz_page.get_text('words')
    if not words:
        return []

    # ── Group words into Y-bands ─────────────────────────────────────
    row_bands: dict[int, list] = {}
    for w in words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        band = int(y0 // BAND_PT)
        row_bands.setdefault(band, []).append((x0, y0, x1, y1, text))

    # ── Classify each band ───────────────────────────────────────────
    band_info: dict[int, dict] = {}
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

    # ── Identify tabular bands (≥2 big gaps + numeric + wide) ────────
    tabular = sorted(
        b for b, info in band_info.items()
        if info['big_gaps'] >= 2 and info['has_num']
        and info['span'] > pw * SPAN_RATIO
    )
    if not tabular:
        return []

    # ── Cluster tabular bands ────────────────────────────────────────
    clusters: list[list[int]] = [[tabular[0]]]
    for b in tabular[1:]:
        if b - clusters[-1][-1] <= CLUSTER_TOL:
            clusters[-1].append(b)
        else:
            clusters.append([b])

    # ── Build table boxes ────────────────────────────────────────────
    tables = []
    for cluster in clusters:
        if len(cluster) < MIN_TABULAR_BANDS:
            continue

        first_tab, last_tab = cluster[0], cluster[-1]

        # Extend upward for header rows
        start = first_tab
        for b in range(first_tab - 1, max(first_tab - HEADER_EXTEND - 1, 0), -1):
            if b in band_info and band_info[b]['span'] > pw * 0.2:
                start = b
            else:
                break

        # Extend downward for footer / totals
        end = last_tab
        for b in range(last_tab + 1, last_tab + HEADER_EXTEND + 1):
            if b in band_info and band_info[b]['span'] > pw * 0.2:
                end = b
            else:
                break

        # Collect all words in the extended range
        all_words = []
        for b in range(start, end + 1):
            all_words.extend(row_bands.get(b, []))
        if not all_words:
            continue

        x0 = min(w[0] for w in all_words)
        x1 = max(w[2] for w in all_words)
        y0 = max(0, start * BAND_PT - BAND_PT)
        y1 = min(ph, (end + 1) * BAND_PT + BAND_PT)

        tables.append((round(x0, 1), round(y0, 1),
                        round(x1 - x0, 1), round(y1 - y0, 1)))

    tables.sort(key=lambda b: (b[1], b[0]))
    return tables


# ====================================================================
#  Layer 3: OpenCV band analysis (scanned / image-only pages)
# ====================================================================
def detect_tables_cv_scan(page_img: np.ndarray):
    """
    Detect tables on scanned (image-only) pages using OpenCV adaptive
    thresholding + word-fragment dilation + row-band gap analysis.

    Returns list of (x, y, w, h) in pixel coordinates.
    """
    height, width = page_img.shape[:2]

    gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10,
    )
    wk = cv2.getStructuringElement(cv2.MORPH_RECT, CV_WORD_KERNEL)
    dilated = cv2.dilate(adapt, wk, iterations=1)

    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    frags = [cv2.boundingRect(c) for c in cnts
             if cv2.boundingRect(c)[2] > CV_MIN_FRAG_W]

    # Group into Y-bands
    row_map: dict[int, list] = {}
    for (fx, fy, fw, fh) in frags:
        row_map.setdefault(fy // CV_BAND_PX, []).append((fx, fy, fw, fh))

    # Classify bands
    table_bands: list[tuple[int, int]] = []
    for band in sorted(row_map.keys()):
        items = row_map[band]
        if len(items) < 2:
            continue
        sorted_items = sorted(items, key=lambda it: it[0])
        spread = max(it[0] for it in sorted_items) - min(it[0] for it in sorted_items)
        if spread <= width * CV_SPREAD_RATIO:
            continue
        gaps = []
        for i in range(1, len(sorted_items)):
            g = sorted_items[i][0] - (sorted_items[i - 1][0] + sorted_items[i - 1][2])
            if g > 0:
                gaps.append(g)
        max_gap = max(gaps) if gaps else 0
        n_big = sum(1 for g in gaps if g > CV_BIG_GAP_PX)
        if max_gap > CV_MAX_GAP_PX and n_big >= 1:
            table_bands.append((band, n_big))

    if not table_bands:
        return []

    # Form contiguous runs
    runs: list[list[tuple[int, int]]] = []
    cur = [table_bands[0]]
    for band, nb in table_bands[1:]:
        if band - cur[-1][0] <= CV_RUN_GAP_TOL:
            cur.append((band, nb))
        else:
            runs.append(cur)
            cur = [(band, nb)]
    runs.append(cur)

    boxes = []
    for run in runs:
        if len(run) < CV_MIN_RUN_LEN:
            continue
        multi_col = sum(1 for (_, nb) in run if nb >= 2)
        if multi_col < len(run) * CV_MULTI_COL_RATIO:
            continue
        band_indices = [b for b, _ in run]
        y0 = min(band_indices) * CV_BAND_PX
        y1 = (max(band_indices) + 1) * CV_BAND_PX
        run_frags = []
        for b, _ in run:
            run_frags.extend(row_map.get(b, []))
        if not run_frags:
            continue
        x0 = min(f[0] for f in run_frags)
        x1 = max(f[0] + f[2] for f in run_frags)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


# ====================================================================
#  Pixel ↔ Point conversion
# ====================================================================
def px_to_pt(x_px, y_px, w_px, h_px):
    """Convert pixel bbox to PDF-point bbox."""
    return dict(
        x=round(x_px * PX_TO_PT, 2), y=round(y_px * PX_TO_PT, 2),
        w=round(w_px * PX_TO_PT, 2), h=round(h_px * PX_TO_PT, 2),
    )

def pt_to_px(x_pt, y_pt, w_pt, h_pt):
    """Convert PDF-point bbox to pixel bbox (integers)."""
    return (int(round(x_pt * PT_TO_PX)), int(round(y_pt * PT_TO_PX)),
            int(round(w_pt * PT_TO_PX)), int(round(h_pt * PT_TO_PX)))


# ====================================================================
#  Validation mask renderer
# ====================================================================
def render_mask(page_img: np.ndarray, boxes: list):
    """
    Draw semi-transparent blue rectangles and green borders over
    detected tables.  Returns annotated copy of the page image.
    """
    overlay = page_img.copy()
    output  = page_img.copy()

    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h),
                      MASK_COLOR_BGR, thickness=-1)
        cv2.rectangle(output, (x, y), (x + w, y + h),
                      BORDER_COLOR_BGR, thickness=BORDER_THICKNESS)

    cv2.addWeighted(overlay, MASK_ALPHA, output, 1 - MASK_ALPHA, 0, output)
    return output


# ====================================================================
#  Main pipeline
# ====================================================================
def run_pipeline(pdf_path: str, output_root: str = OUTPUT_ROOT):
    """
    Full pipeline: PDF → detection → crops + masks + JSON.
    """
    crops_dir = os.path.join(output_root, "crops")
    masks_dir = os.path.join(output_root, "masks")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # ── Open PDF with PyMuPDF for native text analysis ───────────────
    doc = fitz.open(pdf_path)

    # ── Convert PDF → images at 300 DPI ──────────────────────────────
    print(f"\n  Converting PDF → images at {DPI} DPI …")
    t0 = time.time()
    page_images = convert_from_path(pdf_path, dpi=DPI, fmt="png")
    print(f"  → {len(page_images)} pages in {time.time() - t0:.1f}s\n")

    global_table_id = 0
    all_metadata    = []
    method_stats    = {"morphological": 0, "native_text": 0, "cv_scan": 0}

    for page_idx, pil_img in enumerate(page_images):
        page_num = page_idx + 1
        page_np  = np.array(pil_img)
        page_bgr = cv2.cvtColor(page_np, cv2.COLOR_RGB2BGR)
        h_px, w_px = page_bgr.shape[:2]
        fitz_page = doc[page_idx]
        has_native = len(fitz_page.get_text('words')) > 0

        # ── Layer 1: Morphological line detection ────────────────────
        boxes_px = detect_lined_tables(page_bgr)
        method = "morphological"

        # ── Layer 2: Native-text band analysis ───────────────────────
        if not boxes_px and has_native:
            boxes_pt = detect_tables_native(fitz_page)
            if boxes_pt:
                # Convert point boxes → pixel boxes
                boxes_px = [pt_to_px(*b) for b in boxes_pt]
                method = "native_text"

        # ── Layer 3: OpenCV scan fallback (scanned pages ONLY) ───────
        # Only invoked when the page has zero native text (scanned image).
        # If Layer 2 ran but found nothing, the page has text but no
        # tables — we must NOT fall through to cv_scan which would
        # produce false positives on dense text paragraphs.
        if not boxes_px and not has_native:
            boxes_px = detect_tables_cv_scan(page_bgr)
            if boxes_px:
                method = "cv_scan"

        if not boxes_px:
            print(f"  Page {page_num:>2}/{len(page_images)} — no tables")
            continue

        method_stats[method] += len(boxes_px)

        # ── Render validation mask ───────────────────────────────────
        mask_img = render_mask(page_bgr, boxes_px)
        mask_path = os.path.join(masks_dir, f"page_{page_num:02d}_mask.png")
        cv2.imwrite(mask_path, mask_img)

        # ── Process each table ───────────────────────────────────────
        for (x, y, w, h) in boxes_px:
            global_table_id += 1
            table_name = f"table_{global_table_id:03d}.png"

            x0 = max(x - CROP_PAD, 0)
            y0 = max(y - CROP_PAD, 0)
            x1 = min(x + w + CROP_PAD, w_px)
            y1 = min(y + h + CROP_PAD, h_px)
            crop = page_bgr[y0:y1, x0:x1]
            cv2.imwrite(os.path.join(crops_dir, table_name), crop)

            bbox_pt = px_to_pt(x, y, w, h)
            aspect  = round(w / h, 4) if h > 0 else 0.0

            meta = dict(
                table_id      = global_table_id,
                page_number   = page_num,
                filename      = table_name,
                detection     = method,
                bbox_pixels   = dict(x=x, y=y, w=w, h=h),
                bbox_points   = bbox_pt,
                aspect_ratio  = aspect,
                page_size_px  = dict(width=w_px, height=h_px),
                page_size_pt  = dict(
                    width  = round(w_px * PX_TO_PT, 2),
                    height = round(h_px * PX_TO_PT, 2),
                ),
                dpi           = DPI,
            )
            all_metadata.append(meta)

            print(f"  Page {page_num:>2}/{len(page_images)} — "
                  f"Table {global_table_id:>3}  "
                  f"[{x},{y} {w}×{h} px]  "
                  f"[{bbox_pt['x']},{bbox_pt['y']} "
                  f"{bbox_pt['w']}×{bbox_pt['h']} pt]  "
                  f"AR={aspect}  ({method})  ✓")

    doc.close()

    # ── Write JSON manifest ──────────────────────────────────────────
    json_path = os.path.join(output_root, "spatial_data.json")
    payload = dict(
        source_pdf    = os.path.basename(pdf_path),
        total_pages   = len(page_images),
        total_tables  = global_table_id,
        dpi           = DPI,
        px_to_pt      = PX_TO_PT,
        detection     = dict(
            layer_1 = "morphological_line_detection (Otsu + w/40 h/40 kernels)",
            layer_2 = "native_text_band_analysis (PyMuPDF word gaps)",
            layer_3 = "cv_scan_band_analysis (adaptive thresh + dilation)",
        ),
        method_stats  = method_stats,
        tables        = all_metadata,
    )
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return all_metadata


# ====================================================================
#  Entry point
# ====================================================================
def main():
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else 'el-bankalahly .pdf'

    print('=' * 64)
    print('  TABLE DETECTION — THREE-LAYER HYBRID PIPELINE')
    print('=' * 64)
    print(f'  Input:      {pdf_file}')
    print(f'  Output:     {OUTPUT_ROOT}/')
    print(f'  DPI:        {DPI}')
    print(f'  px→pt:      ×{PX_TO_PT:.6f}')
    print(f'  Layer 1:    Otsu + morph kernels (w/{HORIZ_KERNEL_DIV}, 1)'
          f' + (1, h/{VERT_KERNEL_DIV})')
    print(f'  Layer 2:    PyMuPDF word-band analysis'
          f' (gap>{BIG_GAP_PT}pt, ≥{MIN_TABULAR_BANDS} bands)')
    print(f'  Layer 3:    OpenCV scan fallback'
          f' (gap>{CV_MAX_GAP_PX}px, run≥{CV_MIN_RUN_LEN})')

    if not os.path.exists(pdf_file):
        print(f'\n  [ERROR] PDF not found: {pdf_file}')
        sys.exit(1)

    metadata = run_pipeline(pdf_file)

    pages_with = sorted(set(m['page_number'] for m in metadata))
    total_area = sum(m['bbox_pixels']['w'] * m['bbox_pixels']['h']
                     for m in metadata)
    print(f"\n{'=' * 64}")
    print(f"  ✓ Tables detected:   {len(metadata)}")
    print(f"  ✓ Pages with tables: {pages_with}")
    print(f"  ✓ Crops:             {OUTPUT_ROOT}/crops/")
    print(f"  ✓ Masks:             {OUTPUT_ROOT}/masks/")
    print(f"  ✓ Metadata:          {OUTPUT_ROOT}/spatial_data.json")
    print(f"  ✓ Total table area:  {total_area:,} px²")
    print(f"{'=' * 64}")


if __name__ == '__main__':
    main()
