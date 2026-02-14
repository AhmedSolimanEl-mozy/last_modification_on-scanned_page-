#!/usr/bin/env python3
"""
numeric_validator.py — Line-Level Stability Validation
========================================================

Replaces token-level visual metrics (SSIM, pixel overlap, IoU)
with a deterministic line-level stability check.

For each rendered line, compare:
  - Original Surya line text vs final rendered line text
  - Same digit count
  - Same digit order
  - Bounding box width difference <= 5%

If any check fails -> mark the ENTIRE line UNTRUSTED.
Digits are NEVER modified during validation.

REMOVED:
  - SSIM computation
  - Pixel overlap / IoU
  - Visual similarity scoring
  - Trust-score arithmetic
  - Per-token visual diff images

Usage:
    from numeric_validator import validate_line_stability, LineStabilityResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from digit_ocr import (
    TrustStatus, FailureReason, TokenOCRResult,
    TRUST_SCORES, extract_digits,
)


# ────────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────────
BBOX_WIDTH_TOLERANCE = 0.05    # 5% max difference in bounding box width


# ────────────────────────────────────────────────────────────────────
#  Data Model
# ────────────────────────────────────────────────────────────────────
@dataclass
class LineStabilityResult:
    """Result of line-level stability check."""
    line_id: int
    original_text: str            # Surya line text
    final_text: str               # rendered/final line text
    original_digit_count: int = 0
    final_digit_count: int = 0
    original_digits: str = ""     # extracted digit sequence
    final_digits: str = ""        # extracted digit sequence
    original_bbox_width: float = 0.0
    final_bbox_width: float = 0.0
    bbox_width_diff_pct: float = 0.0
    passed: bool = True
    failure_reasons: List[str] = field(default_factory=list)

    @property
    def is_stable(self) -> bool:
        return self.passed


@dataclass
class PageStabilityResult:
    """Stability check results for an entire page."""
    page_number: int
    line_results: List[LineStabilityResult] = field(default_factory=list)
    total_lines: int = 0
    stable_lines: int = 0
    unstable_lines: int = 0

    def compute_summary(self):
        self.total_lines = len(self.line_results)
        self.stable_lines = sum(1 for r in self.line_results if r.passed)
        self.unstable_lines = self.total_lines - self.stable_lines


# ────────────────────────────────────────────────────────────────────
#  Line-Level Stability Check
# ────────────────────────────────────────────────────────────────────
def validate_line_stability(
    original_line_text: str,
    final_line_text: str,
    original_bbox: List[float],
    final_bbox: Optional[List[float]] = None,
    line_id: int = 0,
) -> LineStabilityResult:
    """Validate that a line is stable after processing.

    Checks:
      1. Same digit count (original vs final)
      2. Same digit order (sequence must match exactly)
      3. Bounding box width difference <= 5%

    On failure: marks entire line UNTRUSTED.
    NEVER modifies digits.

    Args:
        original_line_text: Text from Surya OCR for this line.
        final_line_text: Text after any token processing.
        original_bbox: Original line bbox [x0, y0, x1, y1].
        final_bbox: Final line bbox (defaults to original if None).
        line_id: Line identifier.

    Returns:
        LineStabilityResult with pass/fail and reasons.
    """
    result = LineStabilityResult(
        line_id=line_id,
        original_text=original_line_text,
        final_text=final_line_text,
    )

    # Extract digit sequences
    orig_digits = extract_digits(original_line_text)
    final_digits = extract_digits(final_line_text)

    result.original_digits = orig_digits
    result.final_digits = final_digits
    result.original_digit_count = len(orig_digits)
    result.final_digit_count = len(final_digits)

    # Check 1: Same digit count
    if len(orig_digits) != len(final_digits):
        result.passed = False
        result.failure_reasons.append(
            f"Digit count mismatch: {len(orig_digits)} -> {len(final_digits)}"
        )

    # Check 2: Same digit order
    if orig_digits != final_digits:
        result.passed = False
        result.failure_reasons.append(
            f"Digit sequence changed: '{orig_digits}' -> '{final_digits}'"
        )

    # Check 3: Bounding box width difference
    if original_bbox and len(original_bbox) >= 4:
        orig_width = original_bbox[2] - original_bbox[0]
        result.original_bbox_width = orig_width

        if final_bbox and len(final_bbox) >= 4:
            final_width = final_bbox[2] - final_bbox[0]
            result.final_bbox_width = final_width

            if orig_width > 0:
                width_diff = abs(final_width - orig_width) / orig_width
                result.bbox_width_diff_pct = round(width_diff, 4)

                if width_diff > BBOX_WIDTH_TOLERANCE:
                    result.passed = False
                    result.failure_reasons.append(
                        f"Bbox width diff {width_diff:.1%} > {BBOX_WIDTH_TOLERANCE:.0%}"
                    )
        else:
            result.final_bbox_width = orig_width

    return result


def validate_page_lines(
    original_lines: List[dict],
    final_lines: List[dict],
    page_number: int,
) -> PageStabilityResult:
    """Validate stability of all lines on a page.

    Compares original Surya line text against final rendered text.
    Only checks lines that contain numeric tokens.

    Args:
        original_lines: Line dicts from Surya OCR (with 'text', 'bbox').
        final_lines: Line dicts after token processing.
        page_number: 1-based page number.

    Returns:
        PageStabilityResult with per-line results.
    """
    page_result = PageStabilityResult(page_number=page_number)

    # Match lines by index (they should be parallel)
    n = min(len(original_lines), len(final_lines))

    for i in range(n):
        orig = original_lines[i]
        final = final_lines[i]

        orig_text = orig.get('text', '')
        final_text = final.get('text', '')

        # Only check lines that have digits
        if not extract_digits(orig_text) and not extract_digits(final_text):
            continue

        orig_bbox = orig.get('bbox', [0, 0, 0, 0])
        final_bbox = final.get('bbox', orig_bbox)

        line_result = validate_line_stability(
            original_line_text=orig_text,
            final_line_text=final_text,
            original_bbox=orig_bbox,
            final_bbox=final_bbox,
            line_id=orig.get('line_id', i),
        )
        page_result.line_results.append(line_result)

    page_result.compute_summary()
    return page_result


def apply_line_stability_to_tokens(
    page_stability: PageStabilityResult,
    ocr_results: List[TokenOCRResult],
    lines: List[dict],
) -> List[TokenOCRResult]:
    """Mark tokens in unstable lines as UNTRUSTED.

    For each unstable line, all numeric tokens in that line
    become UNTRUSTED (regardless of their individual status).
    LOCKED tokens in unstable lines are also marked UNTRUSTED.

    Args:
        page_stability: Line-level stability results.
        ocr_results: Token OCR results (parallel to numeric tokens).
        lines: Line dicts with 'tokens' lists.

    Returns:
        Updated ocr_results with line-stability failures applied.
    """
    # Build set of unstable line IDs
    unstable_line_ids = set()
    for lr in page_stability.line_results:
        if not lr.passed:
            unstable_line_ids.add(lr.line_id)

    if not unstable_line_ids:
        return ocr_results

    # Build mapping: token -> line_id
    # (tokens in ocr_results correspond to NUMERIC tokens only)
    numeric_idx = 0
    for line in lines:
        line_id = line.get('line_id', -1)
        for tok in line.get('tokens', []):
            if tok.token_type == "NUMERIC":
                if numeric_idx < len(ocr_results):
                    if line_id in unstable_line_ids:
                        result = ocr_results[numeric_idx]
                        result.trust_status = TrustStatus.UNTRUSTED
                        result.trust_score = TRUST_SCORES[TrustStatus.UNTRUSTED]
                        result.locked = False
                        if FailureReason.LINE_STABILITY_FAIL not in result.failure_reasons:
                            result.failure_reasons.append(
                                FailureReason.LINE_STABILITY_FAIL)
                    numeric_idx += 1

    return ocr_results
