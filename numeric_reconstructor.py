#!/usr/bin/env python3
"""
numeric_reconstructor.py — Financial-Grade Number Reconstruction
================================================================

Reconstructs complete numbers from validated digit tokens using
DISCRETE trust states (not continuous scores).

Trust States:
  LOCKED         -> 1.0  (frozen, cannot be modified)
  SURYA_VALID    -> 0.85 (Surya confident, valid digits)
  CNN_CONFIRMED  -> 0.80 (CNN confirmed low-confidence Surya)
  UNTRUSTED      -> 0.0  (cannot verify)

Rules:
  - No weighted averages
  - No trust-score arithmetic
  - No SSIM or pixel overlap
  - No visual similarity scoring
  - Never modify digits after initial OCR
  - If uncertain -> refuse, mark UNTRUSTED

REMOVED:
  - _compute_combined_trust() weighted formula
  - SSIM / pixel overlap weights
  - Dual-OCR agreement bonus
  - VISUAL_MISMATCH handling
  - Continuous trust score computation

Usage:
    from numeric_reconstructor import reconstruct_page_numbers, NumericValue
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from digit_ocr import (
    TrustStatus, FailureReason, TokenOCRResult,
    TRUST_SCORES, ARABIC_INDIC_DIGITS,
)


# ────────────────────────────────────────────────────────────────────
#  Data Model
# ────────────────────────────────────────────────────────────────────
@dataclass
class NumericValue:
    """A reconstructed numeric value with discrete trust status."""
    digits: str                 # Arabic-Indic representation
    trust_score: float          # Fixed: 1.0, 0.85, 0.80, or 0.0
    status: TrustStatus         # LOCKED | SURYA_VALID | CNN_CONFIRMED | UNTRUSTED
    failure_reasons: List[FailureReason] = field(default_factory=list)
    # Source information
    page: int = 0               # 1-based page number
    bbox: List[float] = field(default_factory=list)  # PDF points
    original_text: str = ""     # text from primary OCR
    # Validation details
    surya_confidence: float = 0.0
    locked: bool = False        # True -> was frozen at OCR stage
    cnn_confirmed: bool = False # True -> CNN validated
    revalidated: bool = False   # True -> was re-validated (column boost)
    # Font info
    font_size: float = 0.0
    # Rendering control
    render_badge: bool = False  # True -> show UNTRUSTED badge
    badge_tooltip: str = ""     # Explanation for UNTRUSTED

    def to_dict(self) -> dict:
        return {
            "digits": self.digits,
            "trust_score": self.trust_score,
            "status": self.status.value,
            "failure_reasons": [r.value for r in self.failure_reasons],
            "page": self.page,
            "bbox": self.bbox,
            "original_text": self.original_text,
            "surya_confidence": round(self.surya_confidence, 4),
            "locked": self.locked,
            "cnn_confirmed": self.cnn_confirmed,
            "revalidated": self.revalidated,
        }


# ────────────────────────────────────────────────────────────────────
#  Failure Reason Collection
# ────────────────────────────────────────────────────────────────────
def _collect_failure_reasons(
    ocr_result: TokenOCRResult,
) -> List[FailureReason]:
    """Collect all failure reasons from OCR result."""
    reasons = set()
    for r in ocr_result.failure_reasons:
        if r != FailureReason.NONE:
            reasons.add(r)
    return sorted(reasons, key=lambda r: r.value)


def _build_badge_tooltip(reasons: List[FailureReason]) -> str:
    """Build human-readable tooltip for UNTRUSTED badge."""
    descriptions = {
        FailureReason.CNN_DISAGREEMENT: "CNN classifier disagrees with Surya",
        FailureReason.LOW_CONFIDENCE: "OCR confidence below threshold",
        FailureReason.SEGMENTATION_FAILURE: "Could not segment digit region",
        FailureReason.EMPTY_RESULT: "No digit recognized",
        FailureReason.NO_VALID_DIGITS: "Token contains no valid Arabic-Indic digits",
        FailureReason.LINE_STABILITY_FAIL: "Line failed stability check",
    }
    parts = [descriptions.get(r, str(r)) for r in reasons
             if r != FailureReason.NONE]
    return "; ".join(parts) if parts else ""


# ────────────────────────────────────────────────────────────────────
#  Reconstruction (Discrete Trust — No Arithmetic)
# ────────────────────────────────────────────────────────────────────
def reconstruct_numeric_value(
    ocr_result: TokenOCRResult,
    page_number: int,
    font_size: float = 0.0,
) -> NumericValue:
    """Reconstruct a NumericValue from OCR result.

    Trust is DISCRETE — taken directly from the OCR result's
    trust_status. No weighted formula. No arithmetic.

    Rules:
      - LOCKED -> 1.0, digits frozen
      - SURYA_VALID -> 0.85
      - CNN_CONFIRMED -> 0.80
      - UNTRUSTED -> 0.0, badge shown
      - Never modify digits

    Args:
        ocr_result: Token OCR result with discrete trust.
        page_number: 1-based page number.
        font_size: Font size of the token.

    Returns:
        NumericValue with discrete trust status.
    """
    reasons = _collect_failure_reasons(ocr_result)
    status = ocr_result.trust_status
    trust_score = TRUST_SCORES.get(status, 0.0)

    render_badge = (status == TrustStatus.UNTRUSTED)
    tooltip = _build_badge_tooltip(reasons) if render_badge else ""

    return NumericValue(
        digits=ocr_result.validated_text,
        trust_score=trust_score,
        status=status,
        failure_reasons=reasons,
        page=page_number,
        bbox=ocr_result.bbox,
        original_text=ocr_result.original_text,
        surya_confidence=round(ocr_result.surya_confidence, 4),
        locked=ocr_result.locked,
        cnn_confirmed=ocr_result.cnn_confirmed,
        font_size=font_size,
        render_badge=render_badge,
        badge_tooltip=tooltip,
    )


def reconstruct_page_numbers(
    ocr_results: List[TokenOCRResult],
    page_number: int,
    token_font_sizes: Optional[List[float]] = None,
) -> List[NumericValue]:
    """Reconstruct all numeric values for a page.

    Args:
        ocr_results: OCR results for each numeric token.
        page_number: 1-based page number.
        token_font_sizes: Font sizes for each token (optional).

    Returns:
        List of NumericValue with discrete trust status.
    """
    values = []

    for i, ocr_result in enumerate(ocr_results):
        font_size = (token_font_sizes[i]
                     if token_font_sizes and i < len(token_font_sizes)
                     else 0.0)

        nv = reconstruct_numeric_value(
            ocr_result=ocr_result,
            page_number=page_number,
            font_size=font_size,
        )
        values.append(nv)

    return values


def page_trust_summary(values: List[NumericValue]) -> dict:
    """Generate trust summary statistics for a page.

    Returns dict with counts and status breakdown.
    """
    total = len(values)
    if total == 0:
        return {
            "total": 0,
            "locked": 0,
            "surya_valid": 0,
            "cnn_confirmed": 0,
            "trusted": 0,
            "untrusted": 0,
            "revalidated": 0,
            "pct_trusted": 100.0,
            "partial_failure": False,
        }

    locked = sum(1 for v in values if v.status == TrustStatus.LOCKED)
    surya_valid = sum(1 for v in values
                      if v.status == TrustStatus.SURYA_VALID)
    cnn_confirmed = sum(1 for v in values
                        if v.status == TrustStatus.CNN_CONFIRMED)
    untrusted = sum(1 for v in values
                    if v.status == TrustStatus.UNTRUSTED)
    revalidated = sum(1 for v in values if v.revalidated)

    trusted = locked + surya_valid + cnn_confirmed

    return {
        "total": total,
        "locked": locked,
        "surya_valid": surya_valid,
        "cnn_confirmed": cnn_confirmed,
        "trusted": trusted,
        "untrusted": untrusted,
        "revalidated": revalidated,
        "pct_trusted": round(trusted / total * 100, 1),
        "partial_failure": untrusted > 0,
    }
