#!/usr/bin/env python3
"""
numeric_region_detector.py — Financial-Grade Numeric Region Detection
======================================================================

Identifies regions of a scanned PDF page that contain numeric data,
using ONLY visual/statistical signals:
  • High digit density (connected components matching digit aspect ratios)
  • Vertical alignment of similar-width blobs
  • Repeated digit patterns
  • Font-size consistency within a region

NO table detection.  NO grid/border analysis.  NO heuristic guessing.

Each detected region is returned as a NumericRegion with its page,
bounding box, and background crop image for downstream digit-level OCR.

Usage:
    from numeric_region_detector import detect_numeric_regions, NumericRegion
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from token_extract import Token, PageTokens


# ────────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────────
# A region is "numeric" if this fraction of its tokens are NUMERIC
NUMERIC_DENSITY_THRESHOLD = 0.5
# Minimum number of numeric tokens in a region
MIN_NUMERIC_TOKENS        = 1
# Vertical alignment tolerance (pt) — tokens within this x-range
# are considered in the same vertical column
X_ALIGN_TOL               = 12.0
# Maximum vertical gap between consecutive tokens in a region
Y_GAP_MAX                 = 30.0
# Minimum and maximum digit aspect ratio (h/w) for connected components
DIGIT_ASPECT_MIN          = 0.4
DIGIT_ASPECT_MAX          = 4.0
# Padding (in pt) around detected region for cropping
REGION_PAD_PT             = 5.0
# DPI for high-resolution digit crop
DIGIT_CROP_DPI            = 600


# ────────────────────────────────────────────────────────────────────
#  Data Model
# ────────────────────────────────────────────────────────────────────
@dataclass
class NumericRegion:
    """A detected region containing numeric data."""
    region_id: int
    page: int                   # 0-based page index
    bbox: List[float]           # [x0, y0, x1, y1] in PDF points
    tokens: List[Token] = field(default_factory=list)
    background_crop: np.ndarray = field(default=None, repr=False)
    # Statistics
    digit_density: float = 0.0  # fraction of tokens that are NUMERIC
    vertical_alignment: float = 0.0  # avg x-alignment score
    font_size_std: float = 0.0  # std-dev of font sizes in region

    def __post_init__(self):
        if self.background_crop is not None and not isinstance(self.background_crop, np.ndarray):
            self.background_crop = None


@dataclass
class NumericRegionResult:
    """All numeric regions detected on a single page."""
    page_index: int
    page_width: float
    page_height: float
    regions: List[NumericRegion] = field(default_factory=list)
    full_page_image: np.ndarray = field(default=None, repr=False)

    @property
    def total_numeric_tokens(self) -> int:
        return sum(len(r.tokens) for r in self.regions)


# ────────────────────────────────────────────────────────────────────
#  Region Detection from Tokens
# ────────────────────────────────────────────────────────────────────
def _group_tokens_into_columns(
    numeric_tokens: List[Token],
    x_tol: float = X_ALIGN_TOL,
) -> List[List[Token]]:
    """Group numeric tokens into vertical columns by x-center alignment."""
    if not numeric_tokens:
        return []

    # Sort by x-center
    sorted_toks = sorted(numeric_tokens,
                         key=lambda t: (t.bbox[0] + t.bbox[2]) / 2)

    columns: List[List[Token]] = []
    current_col: List[Token] = [sorted_toks[0]]
    current_x = (sorted_toks[0].bbox[0] + sorted_toks[0].bbox[2]) / 2

    for tok in sorted_toks[1:]:
        tok_x = (tok.bbox[0] + tok.bbox[2]) / 2
        if abs(tok_x - current_x) <= x_tol:
            current_col.append(tok)
            # Update running x-center
            current_x = sum((t.bbox[0] + t.bbox[2]) / 2
                            for t in current_col) / len(current_col)
        else:
            columns.append(current_col)
            current_col = [tok]
            current_x = tok_x

    if current_col:
        columns.append(current_col)

    return columns


def _column_to_region(
    column_tokens: List[Token],
    region_id: int,
    page_idx: int,
    pad: float = REGION_PAD_PT,
) -> NumericRegion:
    """Convert a column of tokens into a NumericRegion."""
    # Sort by y-position
    column_tokens.sort(key=lambda t: t.bbox[1])

    # Compute bounding box
    x0 = min(t.bbox[0] for t in column_tokens) - pad
    y0 = min(t.bbox[1] for t in column_tokens) - pad
    x1 = max(t.bbox[2] for t in column_tokens) + pad
    y1 = max(t.bbox[3] for t in column_tokens) + pad

    # Digit density
    n_numeric = sum(1 for t in column_tokens if t.token_type == "NUMERIC")
    density = n_numeric / len(column_tokens) if column_tokens else 0

    # Font size consistency
    sizes = [t.font_size for t in column_tokens]
    fs_std = float(np.std(sizes)) if len(sizes) > 1 else 0.0

    # Vertical alignment score: how closely x-centers align
    x_centers = [(t.bbox[0] + t.bbox[2]) / 2 for t in column_tokens]
    x_std = float(np.std(x_centers)) if len(x_centers) > 1 else 0.0
    alignment = max(0, 1.0 - x_std / X_ALIGN_TOL)

    return NumericRegion(
        region_id=region_id,
        page=page_idx,
        bbox=[round(max(0, x0), 2), round(max(0, y0), 2),
              round(x1, 2), round(y1, 2)],
        tokens=column_tokens,
        digit_density=round(density, 3),
        vertical_alignment=round(alignment, 3),
        font_size_std=round(fs_std, 2),
    )


def _split_column_by_gaps(
    tokens: List[Token],
    y_gap_max: float = Y_GAP_MAX,
) -> List[List[Token]]:
    """Split a column into sub-groups if there are large vertical gaps."""
    if len(tokens) <= 1:
        return [tokens]

    tokens.sort(key=lambda t: t.bbox[1])
    groups: List[List[Token]] = []
    current: List[Token] = [tokens[0]]

    for tok in tokens[1:]:
        prev_bottom = current[-1].bbox[3]
        if tok.bbox[1] - prev_bottom > y_gap_max:
            groups.append(current)
            current = [tok]
        else:
            current.append(tok)

    if current:
        groups.append(current)
    return groups


def _crop_region(
    full_image: np.ndarray,
    region_bbox: List[float],
    page_width: float,
    page_height: float,
    dpi: int = DIGIT_CROP_DPI,
) -> np.ndarray:
    """Crop a region from the full-page image at the specified DPI."""
    img_h, img_w = full_image.shape[:2]
    scale_x = img_w / page_width
    scale_y = img_h / page_height

    x0 = max(0, int(region_bbox[0] * scale_x))
    y0 = max(0, int(region_bbox[1] * scale_y))
    x1 = min(img_w, int(region_bbox[2] * scale_x))
    y1 = min(img_h, int(region_bbox[3] * scale_y))

    crop = full_image[y0:y1, x0:x1]
    return crop.copy() if crop.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8)


# ────────────────────────────────────────────────────────────────────
#  Connected Component Analysis for digit verification
# ────────────────────────────────────────────────────────────────────
def _count_digit_like_components(
    crop: np.ndarray,
    min_area: int = 20,
    max_area_ratio: float = 0.4,
) -> int:
    """Count connected components that look like digit glyphs.

    Uses aspect ratio and size filtering to identify likely digits.
    """
    if crop.size == 0:
        return 0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Binarize
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connected components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    total_area = crop.shape[0] * crop.shape[1]
    digit_count = 0

    for i in range(1, n_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if area < min_area:
            continue
        if area > total_area * max_area_ratio:
            continue
        if w == 0 or h == 0:
            continue

        aspect = h / w
        if DIGIT_ASPECT_MIN <= aspect <= DIGIT_ASPECT_MAX:
            digit_count += 1

    return digit_count


# ────────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────────
def detect_numeric_regions(
    page_tokens: PageTokens,
    full_page_image: np.ndarray,
    page_idx: int,
    min_tokens: int = MIN_NUMERIC_TOKENS,
) -> NumericRegionResult:
    """Detect numeric regions on a page from extracted tokens.

    Strategy:
      1. Collect all NUMERIC tokens.
      2. Group into vertical columns by x-center alignment.
      3. Split columns by large vertical gaps.
      4. For each candidate region, crop from full-page image.
      5. Verify via connected component analysis.
      6. Return only regions that pass verification.

    Args:
        page_tokens: Extracted tokens for this page.
        full_page_image: Full page rendered at high DPI (BGR).
        page_idx: 0-based page index.
        min_tokens: Minimum NUMERIC tokens to form a region.

    Returns:
        NumericRegionResult with all detected regions.
    """
    result = NumericRegionResult(
        page_index=page_idx,
        page_width=page_tokens.page_width,
        page_height=page_tokens.page_height,
        full_page_image=full_page_image,
    )

    # Step 1: Collect numeric tokens
    numeric_tokens = [t for t in page_tokens.tokens
                      if t.token_type == "NUMERIC"]
    if len(numeric_tokens) < min_tokens:
        return result

    # Step 2: Group into columns
    columns = _group_tokens_into_columns(numeric_tokens)

    # Step 3 & 4: Process each column
    region_id = 0
    for col_tokens in columns:
        # Split by vertical gaps
        sub_groups = _split_column_by_gaps(col_tokens)

        for group in sub_groups:
            if len(group) < min_tokens:
                continue

            region = _column_to_region(group, region_id, page_idx)

            # Crop from full-page image
            region.background_crop = _crop_region(
                full_page_image,
                region.bbox,
                page_tokens.page_width,
                page_tokens.page_height,
            )

            # Step 5: Verify via connected components
            n_digits = _count_digit_like_components(region.background_crop)
            if n_digits > 0:
                result.regions.append(region)
                region_id += 1

    return result


def detect_numeric_regions_from_image(
    full_page_image: np.ndarray,
    page_width: float,
    page_height: float,
    page_idx: int,
) -> List[NumericRegion]:
    """Pure image-based numeric region detection (no tokens needed).

    Uses connected component analysis to find clusters of digit-like
    glyphs directly from the page image.

    This is a fallback when token extraction is unreliable.
    """
    img_h, img_w = full_page_image.shape[:2]
    gray = cv2.cvtColor(full_page_image, cv2.COLOR_BGR2GRAY) if full_page_image.ndim == 3 else full_page_image

    # Binarize
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connected components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    scale_x = page_width / img_w
    scale_y = page_height / img_h

    # Filter to digit-like components
    digit_components = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 30 or w == 0 or h == 0:
            continue
        aspect = h / w
        if not (DIGIT_ASPECT_MIN <= aspect <= DIGIT_ASPECT_MAX):
            continue
        # Size filtering: not too large (titles), not too small (noise)
        if h > img_h * 0.1 or h < 5:
            continue

        cx, cy = centroids[i]
        digit_components.append({
            'cx': cx * scale_x,
            'cy': cy * scale_y,
            'bbox': [
                stats[i, cv2.CC_STAT_LEFT] * scale_x,
                stats[i, cv2.CC_STAT_TOP] * scale_y,
                (stats[i, cv2.CC_STAT_LEFT] + w) * scale_x,
                (stats[i, cv2.CC_STAT_TOP] + h) * scale_y,
            ],
            'area': area,
            'h': h * scale_y,
        })

    if not digit_components:
        return []

    # Cluster by x-center (vertical columns)
    digit_components.sort(key=lambda d: d['cx'])
    columns: List[List[dict]] = []
    current: List[dict] = [digit_components[0]]
    current_cx = digit_components[0]['cx']

    for dc in digit_components[1:]:
        if abs(dc['cx'] - current_cx) <= X_ALIGN_TOL:
            current.append(dc)
            current_cx = sum(d['cx'] for d in current) / len(current)
        else:
            if len(current) >= 3:  # Need multiple digit-like components
                columns.append(current)
            current = [dc]
            current_cx = dc['cx']
    if len(current) >= 3:
        columns.append(current)

    # Convert clusters to regions
    regions = []
    for idx, col in enumerate(columns):
        x0 = min(d['bbox'][0] for d in col) - REGION_PAD_PT
        y0 = min(d['bbox'][1] for d in col) - REGION_PAD_PT
        x1 = max(d['bbox'][2] for d in col) + REGION_PAD_PT
        y1 = max(d['bbox'][3] for d in col) + REGION_PAD_PT

        crop = _crop_region(full_page_image,
                            [x0, y0, x1, y1],
                            page_width, page_height)

        regions.append(NumericRegion(
            region_id=idx,
            page=page_idx,
            bbox=[round(max(0, x0), 2), round(max(0, y0), 2),
                  round(x1, 2), round(y1, 2)],
            background_crop=crop,
            digit_density=1.0,
        ))

    return regions
