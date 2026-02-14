#!/usr/bin/env python3
"""
table_reconstruct.py — Geometric Table Reconstruction
======================================================

Takes character-level OCR output (TableOCRResult from table_ocr.py)
and reconstructs a structured table using pure geometry:

  1. Cluster glyphs by Y coordinate → rows
  2. Cluster glyphs by X coordinate → columns
  3. Build cell grid: cells[row][col]
  4. Enforce strict column alignment
  5. Preserve digit order, decimal/thousand separators
  6. Handle RTL text + LTR numeric islands

Key principle: GEOMETRY > SEMANTICS
  • Cell membership is determined by bbox position, not OCR text order.
  • Numerals are treated as LTR islands (never reversed).
  • Column alignment is enforced by bounding box clustering.

Output per table:
    {
      "type": "table",
      "bbox": [x0, y0, x1, y1],
      "rows": [...],
      "columns": [...],
      "cells": [
        {
          "row": int, "col": int,
          "text": str, "bbox": [x0,y0,x1,y1],
          "is_numeric": bool, "confidence": float,
          "glyphs": [...]
        }
      ]
    }

Usage:
    from table_reconstruct import reconstruct_table, TableCell, TableGrid
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

# ────────────────────────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────────────────────────
ROW_CLUSTER_TOLERANCE = 4.0     # pt: glyphs within ±N pt Y are same row
COL_CLUSTER_TOLERANCE = 5.0     # pt: glyph centers within ±N pt X are same col
MIN_COL_GAP           = 8.0     # pt: minimum gap between column boundaries
NUMERIC_RE = re.compile(r'^[\d٠-٩,،.\(\)\-−–\s%]+$')

# Arabic-Indic digit map for normalization
_ARABIC_INDIC = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')


# ────────────────────────────────────────────────────────────────────
#  Data model
# ────────────────────────────────────────────────────────────────────
@dataclass
class CellGlyph:
    char: str
    bbox: List[float]
    confidence: float
    is_numeric: bool = False


@dataclass
class TableCell:
    row: int
    col: int
    text: str
    bbox: List[float]           # [x0, y0, x1, y1] in PDF points
    is_numeric: bool = False
    is_header: bool = False
    confidence: float = 0.0
    direction: str = "ltr"      # "rtl" | "ltr"
    glyphs: List[CellGlyph] = field(default_factory=list)


@dataclass
class RowInfo:
    row_id: int
    y_center: float
    y0: float
    y1: float
    is_header: bool = False


@dataclass
class ColInfo:
    col_id: int
    x_center: float
    x0: float
    x1: float


@dataclass
class TableGrid:
    """Structured table reconstruction result."""
    bbox: List[float]
    rows: List[RowInfo]
    columns: List[ColInfo]
    cells: List[TableCell]
    num_rows: int = 0
    num_cols: int = 0
    confidence: float = 0.0
    page_number: int = 0


# ────────────────────────────────────────────────────────────────────
#  Clustering utilities
# ────────────────────────────────────────────────────────────────────
def cluster_1d(values: List[float], tolerance: float) -> List[List[int]]:
    """Cluster sorted values into groups within ±tolerance.
    Returns list of clusters, each a list of original indices."""
    if not values:
        return []

    indexed = sorted(enumerate(values), key=lambda x: x[1])
    clusters: List[List[int]] = [[indexed[0][0]]]
    cluster_center = indexed[0][1]

    for idx, val in indexed[1:]:
        if abs(val - cluster_center) <= tolerance:
            clusters[-1].append(idx)
            # Update center as running mean
            cluster_center = statistics.mean(
                values[i] for i in clusters[-1]
            )
        else:
            clusters.append([idx])
            cluster_center = val

    return clusters


def refine_row_clusters(
    glyphs: list,
    initial_tolerance: float = ROW_CLUSTER_TOLERANCE,
) -> List[List[int]]:
    """Cluster glyphs into rows by Y midpoint with adaptive tolerance."""
    if not glyphs:
        return []

    y_mids = [(g.bbox[1] + g.bbox[3]) / 2 for g in glyphs]
    return cluster_1d(y_mids, initial_tolerance)


def detect_column_boundaries(
    glyphs: list,
    row_clusters: List[List[int]],
) -> List[Tuple[float, float]]:
    """Detect column boundaries by finding consistent vertical gaps
    across multiple rows.

    Strategy:
      For each row, collect horizontal extents of glyph groups.
      A "glyph group" = consecutive glyphs with <MIN_COL_GAP gap.
      Column boundaries are where gaps appear consistently across rows.

    Returns list of (x0, x1) per column, sorted left→right.
    """
    # Build per-row glyph groups
    row_groups = []
    for cluster_indices in row_clusters:
        row_glyphs = sorted(
            [glyphs[i] for i in cluster_indices],
            key=lambda g: g.bbox[0]
        )
        groups = []
        if not row_glyphs:
            continue
        cur_group = [row_glyphs[0]]
        for g in row_glyphs[1:]:
            prev_right = cur_group[-1].bbox[2]
            cur_left   = g.bbox[0]
            if cur_left - prev_right > MIN_COL_GAP:
                groups.append(cur_group)
                cur_group = [g]
            else:
                cur_group.append(g)
        groups.append(cur_group)

        # Each group → (x0, x1)
        group_extents = [
            (min(g.bbox[0] for g in grp), max(g.bbox[2] for g in grp))
            for grp in groups
        ]
        row_groups.append(group_extents)

    if not row_groups:
        return []

    # Find the most common number of columns
    col_counts = [len(rg) for rg in row_groups]
    if not col_counts:
        return []
    most_common_ncols = max(set(col_counts), key=col_counts.count)

    # Use rows with the most common column count to establish boundaries
    reference_rows = [rg for rg in row_groups if len(rg) == most_common_ncols]
    if not reference_rows:
        reference_rows = row_groups

    # Average the column boundaries across reference rows
    n_cols = most_common_ncols
    col_bounds = []
    for ci in range(n_cols):
        x0s = [rg[ci][0] for rg in reference_rows if ci < len(rg)]
        x1s = [rg[ci][1] for rg in reference_rows if ci < len(rg)]
        if x0s and x1s:
            col_bounds.append((
                statistics.mean(x0s),
                statistics.mean(x1s),
            ))

    # Expand column boundaries to cover gaps (no dead zones)
    if len(col_bounds) >= 2:
        expanded = []
        for i, (cx0, cx1) in enumerate(col_bounds):
            if i == 0:
                # First column: extend left to the leftmost glyph
                new_x0 = min(g.bbox[0] for g in glyphs) - 1
            else:
                # Midpoint between this col's left and previous col's right
                new_x0 = (col_bounds[i - 1][1] + cx0) / 2
            if i == len(col_bounds) - 1:
                new_x1 = max(g.bbox[2] for g in glyphs) + 1
            else:
                new_x1 = (cx1 + col_bounds[i + 1][0]) / 2
            expanded.append((new_x0, new_x1))
        col_bounds = expanded

    return col_bounds


# ────────────────────────────────────────────────────────────────────
#  Glyph → Cell assignment
# ────────────────────────────────────────────────────────────────────
def assign_glyph_to_col(glyph_bbox: List[float],
                         col_bounds: List[Tuple[float, float]]) -> int:
    """Assign a glyph to a column by which column boundary contains
    the glyph's horizontal center."""
    x_center = (glyph_bbox[0] + glyph_bbox[2]) / 2
    for ci, (cx0, cx1) in enumerate(col_bounds):
        if cx0 <= x_center <= cx1:
            return ci
    # Fallback: find nearest column center
    best_col = 0
    best_dist = float('inf')
    for ci, (cx0, cx1) in enumerate(col_bounds):
        col_center = (cx0 + cx1) / 2
        dist = abs(x_center - col_center)
        if dist < best_dist:
            best_dist = dist
            best_col = ci
    return best_col


# ────────────────────────────────────────────────────────────────────
#  Numeric string handling
# ────────────────────────────────────────────────────────────────────
def assemble_cell_text(glyphs: List[CellGlyph], direction: str) -> str:
    """Assemble cell text from glyphs, respecting LTR for numbers
    and RTL for Arabic text.

    Key rule: numerals are ALWAYS assembled left-to-right by bbox X.
    Arabic text is assembled right-to-left by bbox X.
    """
    if not glyphs:
        return ""

    # Sort glyphs by X position (left to right)
    sorted_glyphs = sorted(glyphs, key=lambda g: g.bbox[0])
    raw_text = "".join(g.char for g in sorted_glyphs)

    # Check if the content is numeric
    stripped = raw_text.strip()
    if is_numeric_content(stripped):
        # Numeric: always LTR order (already sorted by X left→right)
        return stripped

    # Mixed or pure Arabic: sort glyphs by X, assemble
    # The visual order from left-to-right already gives correct Unicode order
    # for display (Arabic/Hebrew are stored logically RTL but displayed RTL)
    return raw_text


def is_numeric_content(text: str) -> bool:
    """Check if text is purely numeric/financial."""
    return bool(NUMERIC_RE.match(text.strip())) if text.strip() else False


def normalize_arabic_digits(text: str) -> str:
    """Convert Arabic-Indic digits (٠-٩) to Western digits (0-9)."""
    return text.translate(_ARABIC_INDIC)


# ────────────────────────────────────────────────────────────────────
#  Header detection
# ────────────────────────────────────────────────────────────────────
# Year range for header detection (e.g. fiscal year labels like 2023, 2024)
YEAR_MIN = 1990
YEAR_MAX = 2035
_ARABIC_INDIC_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')


def _is_year_value(text: str) -> bool:
    """Check if text is a 4-digit year within YEAR_MIN..YEAR_MAX.
    Handles both Western digits and Arabic-Indic digits."""
    t = text.strip().translate(_ARABIC_INDIC_DIGITS)
    if len(t) != 4:
        return False
    try:
        return YEAR_MIN <= int(t) <= YEAR_MAX
    except ValueError:
        return False


def detect_header_rows(cells: List[TableCell], num_rows: int) -> List[int]:
    """Detect header rows using three criteria:

    Criterion 1 (original): Rows where most cells are non-numeric → header.
    Criterion 2 (year-row):  ALL numeric cells in the row are 4-digit years
                             in the range [YEAR_MIN..YEAR_MAX].
    Criterion 3 (top-row):   Row 0 if it has a dramatically different
                             numeric magnitude vs data rows.

    A row is HEADER if it matches ANY criterion.
    Header rows are excluded from anomaly detection and numeric statistics.
    """
    if num_rows < 2:
        return []

    headers = set()

    # --- Criterion 1: Low numeric ratio (mostly text) ---
    row_numeric_ratio = {}
    for r in range(num_rows):
        row_cells = [c for c in cells if c.row == r and c.text.strip()]
        if not row_cells:
            continue
        num_count = sum(1 for c in row_cells if c.is_numeric)
        row_numeric_ratio[r] = num_count / len(row_cells)

    # First consecutive rows with low numeric ratio
    for r in sorted(row_numeric_ratio.keys()):
        if row_numeric_ratio[r] < 0.4:
            headers.add(r)
        else:
            break  # Stop at first data row

    # --- Criterion 2: Year-only rows ---
    # A row where ALL non-empty numeric cells are year values.
    for r in range(num_rows):
        row_cells = [c for c in cells if c.row == r and c.text.strip()]
        if not row_cells:
            continue
        numeric_cells = [c for c in row_cells if c.is_numeric]
        if not numeric_cells:
            continue
        # ALL numeric cells must be years
        if all(_is_year_value(c.text) for c in numeric_cells):
            headers.add(r)

    # --- Criterion 3: Top row with magnitude outlier ---
    # If row 0 has numeric cells whose median is an order of magnitude
    # different from the rest, mark it as header.
    if 0 not in headers:
        row0_nums = []
        other_nums = []
        for c in cells:
            if not c.text.strip() or not c.is_numeric:
                continue
            val = _try_parse_float(c.text)
            if val is None:
                continue
            if c.row == 0:
                row0_nums.append(abs(val))
            else:
                other_nums.append(abs(val))
        if row0_nums and other_nums:
            row0_med = statistics.median(row0_nums)
            other_med = statistics.median(other_nums) if other_nums else 1
            if other_med > 0 and (row0_med / other_med > 5 or other_med / max(row0_med, 0.01) > 5):
                headers.add(0)

    return sorted(headers)


def _try_parse_float(text: str) -> Optional[float]:
    """Best-effort parse numeric text to float."""
    t = text.strip().translate(_ARABIC_INDIC_DIGITS)
    neg = False
    if t.startswith('(') and t.endswith(')'):
        neg = True
        t = t[1:-1]
    elif t.startswith('-') or t.startswith('−') or t.startswith('–'):
        neg = True
        t = t[1:]
    t = t.replace(',', '').replace('،', '').replace('%', '').replace(' ', '').strip()
    try:
        val = float(t)
        return -val if neg else val
    except ValueError:
        return None


# ────────────────────────────────────────────────────────────────────
#  Main reconstruction
# ────────────────────────────────────────────────────────────────────
def reconstruct_table(ocr_result, page_number: int = 0) -> TableGrid:
    """
    Reconstruct a structured table from character-level OCR output.

    Args:
        ocr_result: TableOCRResult from table_ocr.py
        page_number: Page number for metadata.

    Returns:
        TableGrid with rows, columns, cells.
    """
    # Collect all glyphs from all lines
    all_glyphs = []
    for tl in ocr_result.table_lines:
        for g in tl.glyphs:
            all_glyphs.append(g)

    if not all_glyphs:
        return TableGrid(
            bbox=ocr_result.region_bbox,
            rows=[], columns=[], cells=[],
            num_rows=0, num_cols=0,
            page_number=page_number,
        )

    # ── Step 1: Cluster into rows by Y midpoint ────────────────────
    row_clusters = refine_row_clusters(all_glyphs)
    row_clusters.sort(key=lambda cl: statistics.mean(
        (all_glyphs[i].bbox[1] + all_glyphs[i].bbox[3]) / 2 for i in cl
    ))

    # Build RowInfo
    rows = []
    for ri, cluster_indices in enumerate(row_clusters):
        y_vals = [(all_glyphs[i].bbox[1] + all_glyphs[i].bbox[3]) / 2
                  for i in cluster_indices]
        y0_vals = [all_glyphs[i].bbox[1] for i in cluster_indices]
        y1_vals = [all_glyphs[i].bbox[3] for i in cluster_indices]
        rows.append(RowInfo(
            row_id=ri,
            y_center=round(statistics.mean(y_vals), 2),
            y0=round(min(y0_vals), 2),
            y1=round(max(y1_vals), 2),
        ))

    # ── Step 2: Detect column boundaries ───────────────────────────
    col_bounds = detect_column_boundaries(all_glyphs, row_clusters)

    columns = []
    for ci, (cx0, cx1) in enumerate(col_bounds):
        columns.append(ColInfo(
            col_id=ci,
            x_center=round((cx0 + cx1) / 2, 2),
            x0=round(cx0, 2),
            x1=round(cx1, 2),
        ))

    # ── Step 3: Assign glyphs to cells ─────────────────────────────
    # cell_map[row][col] → list of CellGlyphs
    cell_map: dict[tuple, list] = defaultdict(list)

    for ri, cluster_indices in enumerate(row_clusters):
        for gi in cluster_indices:
            g = all_glyphs[gi]
            ci = assign_glyph_to_col(g.bbox, col_bounds)
            cell_map[(ri, ci)].append(CellGlyph(
                char=g.char,
                bbox=g.bbox,
                confidence=g.confidence,
                is_numeric=g.is_numeric,
            ))

    # ── Step 4: Build cell objects ─────────────────────────────────
    num_rows = len(rows)
    num_cols = len(columns)
    cells = []

    for ri in range(num_rows):
        for ci in range(num_cols):
            glyphs = cell_map.get((ri, ci), [])
            if not glyphs:
                # Empty cell
                cells.append(TableCell(
                    row=ri, col=ci, text="",
                    bbox=[columns[ci].x0 if ci < len(columns) else 0,
                          rows[ri].y0,
                          columns[ci].x1 if ci < len(columns) else 0,
                          rows[ri].y1],
                    is_numeric=False,
                    confidence=0.0,
                    direction="ltr",
                    glyphs=[],
                ))
                continue

            # Sort glyphs by X for assembly
            sorted_g = sorted(glyphs, key=lambda g: g.bbox[0])

            # Determine direction: if mostly numeric, LTR; else detect
            text_concat = "".join(g.char for g in sorted_g)
            is_num = is_numeric_content(text_concat)

            if is_num:
                direction = "ltr"
                cell_text = text_concat.strip()
            else:
                arabic_count = sum(1 for c in text_concat if '\u0600' <= c <= '\u06FF')
                direction = "rtl" if arabic_count > len(text_concat) * 0.3 else "ltr"
                cell_text = assemble_cell_text(glyphs, direction)

            # Cell bbox = envelope of all glyphs
            cell_bbox = [
                round(min(g.bbox[0] for g in glyphs), 2),
                round(min(g.bbox[1] for g in glyphs), 2),
                round(max(g.bbox[2] for g in glyphs), 2),
                round(max(g.bbox[3] for g in glyphs), 2),
            ]

            avg_conf = statistics.mean(g.confidence for g in glyphs)

            cells.append(TableCell(
                row=ri, col=ci,
                text=cell_text,
                bbox=cell_bbox,
                is_numeric=is_num,
                confidence=round(avg_conf, 3),
                direction=direction,
                glyphs=sorted_g,
            ))

    # ── Step 5: Detect header rows ─────────────────────────────────
    header_rows = detect_header_rows(cells, num_rows)
    for cell in cells:
        if cell.row in header_rows:
            cell.is_header = True
    for row in rows:
        if row.row_id in header_rows:
            row.is_header = True

    # ── Step 6: Compute overall confidence ─────────────────────────
    confs = [c.confidence for c in cells if c.text.strip()]
    overall_conf = statistics.mean(confs) if confs else 0.0

    return TableGrid(
        bbox=ocr_result.region_bbox,
        rows=rows,
        columns=columns,
        cells=cells,
        num_rows=num_rows,
        num_cols=num_cols,
        confidence=round(overall_conf, 3),
        page_number=page_number,
    )


# ────────────────────────────────────────────────────────────────────
#  Serialization
# ────────────────────────────────────────────────────────────────────
def table_grid_to_dict(grid: TableGrid) -> dict:
    """Convert TableGrid to a JSON-serializable dict."""
    return {
        "type": "table",
        "bbox": grid.bbox,
        "num_rows": grid.num_rows,
        "num_cols": grid.num_cols,
        "confidence": grid.confidence,
        "page_number": grid.page_number,
        "rows": [
            {
                "row_id": r.row_id,
                "y_center": r.y_center,
                "y0": r.y0, "y1": r.y1,
                "is_header": r.is_header,
            }
            for r in grid.rows
        ],
        "columns": [
            {
                "col_id": c.col_id,
                "x_center": c.x_center,
                "x0": c.x0, "x1": c.x1,
            }
            for c in grid.columns
        ],
        "cells": [
            {
                "row": c.row, "col": c.col,
                "text": c.text,
                "bbox": c.bbox,
                "is_numeric": c.is_numeric,
                "is_header": c.is_header,
                "confidence": c.confidence,
                "direction": c.direction,
                "glyph_count": len(c.glyphs),
            }
            for c in grid.cells
        ],
    }
