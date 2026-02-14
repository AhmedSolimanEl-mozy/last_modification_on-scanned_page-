#!/usr/bin/env python3
"""
digit_ocr.py — Financial-Grade Numeric Token OCR (Surya-Only)
==============================================================

For every numeric token on a scanned page:
  1. Use ORIGINAL full-page Surya OCR result as primary signal.
  2. If confidence >= 0.92 and valid Arabic-Indic digits -> LOCK.
  3. If confidence < 0.92 -> attempt CNN validation (confirm or UNTRUSTED).
  4. Never override high-confidence Surya digits.
  5. Never guess or fabricate digits.

Tesseract has been REMOVED — it is fundamentally unreliable for
isolated Arabic digits and introduces nondeterministic noise.

Trust model (discrete, not continuous):
  LOCKED         -> 1.0  (high-confidence Surya, frozen)
  SURYA_VALID    -> 0.85 (Surya above threshold, valid digits)
  CNN_CONFIRMED  -> 0.80 (CNN confirmed low-confidence Surya)
  UNTRUSTED      -> 0.0  (cannot be verified)

Usage:
    from digit_ocr import ocr_page_numeric_tokens, TokenOCRResult, TrustStatus
"""

from __future__ import annotations

import hashlib
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

from numeric_region_detector import NumericRegion


# ────────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────────
DIGIT_CROP_DPI       = 600
CONFIDENCE_THRESHOLD = 0.45    # below this -> needs CNN validation
LOCK_THRESHOLD       = 0.92    # at or above this -> LOCKED (frozen)

# Arabic-Indic digit codepoints
ARABIC_INDIC_DIGITS = set('٠١٢٣٤٥٦٧٨٩')
EXTENDED_ARABIC_DIGITS = set('۰۱۲۳۴۵۶۷۸۹')  # Persian/Urdu (U+06F0-U+06F9)
WESTERN_DIGITS = set('0123456789')
ALL_DIGITS = ARABIC_INDIC_DIGITS | EXTENDED_ARABIC_DIGITS | WESTERN_DIGITS

# Mapping tables
WESTERN_TO_ARABIC = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')
ARABIC_TO_WESTERN = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
EXTENDED_TO_ARABIC = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '٠١٢٣٤٥٦٧٨٩')


# ────────────────────────────────────────────────────────────────────
#  Trust Model — Discrete States (NOT continuous)
# ────────────────────────────────────────────────────────────────────
class TrustStatus(str, Enum):
    LOCKED        = "LOCKED"          # 1.0 — frozen, skip all validation
    SURYA_VALID   = "SURYA_VALID"     # 0.85 — Surya confident, valid digits
    CNN_CONFIRMED = "CNN_CONFIRMED"   # 0.80 — CNN confirmed low-conf Surya
    UNTRUSTED     = "UNTRUSTED"       # 0.0 — cannot verify


# Map trust status to fixed score — NO weighted averages
TRUST_SCORES = {
    TrustStatus.LOCKED:        1.0,
    TrustStatus.SURYA_VALID:   0.85,
    TrustStatus.CNN_CONFIRMED: 0.80,
    TrustStatus.UNTRUSTED:     0.0,
}


class FailureReason(str, Enum):
    NONE                  = "NONE"
    LOW_CONFIDENCE        = "LOW_CONFIDENCE"
    SEGMENTATION_FAILURE  = "SEGMENTATION_FAILURE"
    EMPTY_RESULT          = "EMPTY_RESULT"
    CNN_DISAGREEMENT      = "CNN_DISAGREEMENT"
    NO_VALID_DIGITS       = "NO_VALID_DIGITS"
    LINE_STABILITY_FAIL   = "LINE_STABILITY_FAIL"


# ────────────────────────────────────────────────────────────────────
#  Data Model
# ────────────────────────────────────────────────────────────────────
@dataclass
class TokenOCRResult:
    """OCR result for a single numeric token with discrete trust."""
    original_text: str           # text from primary OCR (Surya full-page)
    validated_text: str          # text after validation (never fabricated)
    surya_confidence: float = 0.0
    trust_score: float = 0.0    # fixed: 1.0, 0.85, 0.80, or 0.0
    trust_status: TrustStatus = TrustStatus.UNTRUSTED
    locked: bool = False         # True -> skip all validation, cannot modify
    cnn_confirmed: bool = False  # True -> CNN agreed with Surya
    failure_reasons: List[FailureReason] = field(default_factory=list)
    bbox: List[float] = field(default_factory=list)  # PDF points


# ────────────────────────────────────────────────────────────────────
#  Digit Utilities
# ────────────────────────────────────────────────────────────────────
def _normalize_digit(ch: str) -> str:
    """Normalize a digit to Arabic-Indic for comparison."""
    if ch in WESTERN_DIGITS:
        return ch.translate(WESTERN_TO_ARABIC)
    if ch in EXTENDED_ARABIC_DIGITS:
        return ch.translate(EXTENDED_TO_ARABIC)
    return ch


def extract_digits(text: str) -> str:
    """Extract and normalize all digit characters from text."""
    return ''.join(
        _normalize_digit(c) for c in text
        if c in ARABIC_INDIC_DIGITS or c in WESTERN_DIGITS
        or c in EXTENDED_ARABIC_DIGITS
    )


def has_valid_arabic_digits(text: str) -> bool:
    """Check if text contains at least one Arabic-Indic digit
    (after normalization)."""
    digits = extract_digits(text)
    return len(digits) > 0 and all(c in ARABIC_INDIC_DIGITS for c in digits)


# ────────────────────────────────────────────────────────────────────
#  OCR Result Cache (determinism guarantee)
# ────────────────────────────────────────────────────────────────────
_ocr_cache: dict = {}


def _cache_key(image_hash: str, bbox: List[float]) -> str:
    """Generate deterministic cache key."""
    bbox_str = ','.join(f'{v:.2f}' for v in bbox)
    return f"{image_hash}:{bbox_str}"


def compute_image_hash(image: np.ndarray) -> str:
    """Compute deterministic hash of a normalized image."""
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


# ────────────────────────────────────────────────────────────────────
#  Token Classification and Locking
# ────────────────────────────────────────────────────────────────────
def classify_and_lock_token(
    original_text: str,
    original_confidence: float,
    token_bbox: List[float],
) -> TokenOCRResult:
    """Classify a numeric token and lock if high-confidence.

    Rules:
      - confidence >= 0.92 AND valid Arabic-Indic digits -> LOCKED (1.0)
      - confidence >= 0.60 AND valid Arabic-Indic digits -> SURYA_VALID (0.85)
      - confidence < 0.60 -> candidate for CNN validation (UNTRUSTED)
      - no valid digits -> UNTRUSTED

    LOCKED tokens:
      - Skip all validation phases
      - Cannot be modified
      - Cannot become UNTRUSTED
    """
    has_digits = has_valid_arabic_digits(original_text)

    # Case 1: High confidence + valid digits -> LOCK
    if original_confidence >= LOCK_THRESHOLD and has_digits:
        return TokenOCRResult(
            original_text=original_text,
            validated_text=original_text,
            surya_confidence=original_confidence,
            trust_score=TRUST_SCORES[TrustStatus.LOCKED],
            trust_status=TrustStatus.LOCKED,
            locked=True,
            failure_reasons=[],
            bbox=token_bbox,
        )

    # Case 2: Moderate confidence + valid digits -> SURYA_VALID
    if original_confidence >= CONFIDENCE_THRESHOLD and has_digits:
        return TokenOCRResult(
            original_text=original_text,
            validated_text=original_text,
            surya_confidence=original_confidence,
            trust_score=TRUST_SCORES[TrustStatus.SURYA_VALID],
            trust_status=TrustStatus.SURYA_VALID,
            locked=False,
            failure_reasons=[],
            bbox=token_bbox,
        )

    # Case 3: Low confidence or no valid digits -> UNTRUSTED
    reasons = []
    if not has_digits:
        reasons.append(FailureReason.NO_VALID_DIGITS)
    if original_confidence < CONFIDENCE_THRESHOLD:
        reasons.append(FailureReason.LOW_CONFIDENCE)
    if not original_text.strip():
        reasons.append(FailureReason.EMPTY_RESULT)

    return TokenOCRResult(
        original_text=original_text,
        validated_text=original_text,  # never fabricate
        surya_confidence=original_confidence,
        trust_score=TRUST_SCORES[TrustStatus.UNTRUSTED],
        trust_status=TrustStatus.UNTRUSTED,
        locked=False,
        failure_reasons=reasons if reasons else [FailureReason.LOW_CONFIDENCE],
        bbox=token_bbox,
    )


# ────────────────────────────────────────────────────────────────────
#  CNN Validation (validation-only, never primary)
# ────────────────────────────────────────────────────────────────────
def validate_token_with_cnn(
    ocr_result: TokenOCRResult,
    full_page_image: np.ndarray,
    page_width: float,
    page_height: float,
) -> TokenOCRResult:
    """Attempt CNN validation for a non-locked token.

    The CNN is NEVER a primary recognizer. It may only:
      - Confirm Surya output -> CNN_CONFIRMED (0.80)
      - Mark token UNTRUSTED -> CNN_DISAGREEMENT

    It may NOT:
      - Override high-confidence Surya digits
      - Generate multi-digit strings

    Only runs when:
      - Token is NOT locked
      - Surya confidence < 0.92
      - Token contains digits (NUMERIC or NUMERIC_CANDIDATE)
    """
    # Never touch locked tokens
    if ocr_result.locked:
        return ocr_result

    # Only validate tokens that have digits to check
    digits = extract_digits(ocr_result.original_text)
    if not digits:
        return ocr_result

    # Skip if already high confidence (shouldn't happen, but safety)
    if ocr_result.surya_confidence >= LOCK_THRESHOLD:
        return ocr_result

    # Crop token region from full page image
    img_h, img_w = full_page_image.shape[:2]
    scale_x = img_w / page_width
    scale_y = img_h / page_height

    bbox = ocr_result.bbox
    if not bbox or len(bbox) < 4:
        return ocr_result

    pad_px = 8
    x0 = max(0, int(bbox[0] * scale_x) - pad_px)
    y0 = max(0, int(bbox[1] * scale_y) - pad_px)
    x1 = min(img_w, int(bbox[2] * scale_x) + pad_px)
    y1 = min(img_h, int(bbox[3] * scale_y) + pad_px)

    crop = full_page_image[y0:y1, x0:x1]
    if crop.size == 0:
        return ocr_result

    # Run CNN classifier
    try:
        from cnn_digit_classifier import classify_digit_crop
        cnn_result = classify_digit_crop(crop, digits)

        if cnn_result is None:
            # CNN not available or failed — don't change status
            return ocr_result

        if cnn_result['confirmed']:
            # CNN confirms Surya's digits
            ocr_result.trust_status = TrustStatus.CNN_CONFIRMED
            ocr_result.trust_score = TRUST_SCORES[TrustStatus.CNN_CONFIRMED]
            ocr_result.cnn_confirmed = True
            return ocr_result
        else:
            # CNN disagrees — mark UNTRUSTED but DO NOT change digits
            ocr_result.trust_status = TrustStatus.UNTRUSTED
            ocr_result.trust_score = TRUST_SCORES[TrustStatus.UNTRUSTED]
            ocr_result.failure_reasons.append(FailureReason.CNN_DISAGREEMENT)
            return ocr_result

    except ImportError:
        # CNN module not available — keep existing status
        return ocr_result
    except Exception:
        # CNN failed — keep existing status, don't crash
        return ocr_result


# ────────────────────────────────────────────────────────────────────
#  Per-Token OCR (main entry point)
# ────────────────────────────────────────────────────────────────────
def ocr_numeric_token(
    full_page_image: np.ndarray,
    token_bbox_pt: List[float],
    page_width: float,
    page_height: float,
    original_text: str = "",
    original_confidence: float = 0.0,
) -> TokenOCRResult:
    """Process a single numeric token: classify, lock, or validate.

    Uses the ORIGINAL full-page Surya OCR result as primary signal.
    No secondary OCR engine. CNN for validation only.

    Args:
        full_page_image: Full page at high DPI (BGR).
        token_bbox_pt: Token bbox in PDF points [x0, y0, x1, y1].
        page_width: Page width in points.
        page_height: Page height in points.
        original_text: Text from primary OCR pass (Surya full-page).
        original_confidence: Confidence from primary OCR pass.

    Returns:
        TokenOCRResult with discrete trust status.
    """
    # Step 1: Classify and lock
    result = classify_and_lock_token(
        original_text=original_text,
        original_confidence=original_confidence,
        token_bbox=token_bbox_pt,
    )

    # Step 2: If not locked and low confidence, attempt CNN validation
    if not result.locked and result.surya_confidence < LOCK_THRESHOLD:
        result = validate_token_with_cnn(
            result, full_page_image, page_width, page_height,
        )

    return result


# ────────────────────────────────────────────────────────────────────
#  Batch OCR for a page's numeric tokens
# ────────────────────────────────────────────────────────────────────
def ocr_page_numeric_tokens(
    page_tokens,  # PageTokens
    full_page_image: np.ndarray,
) -> List[TokenOCRResult]:
    """Run classification + validation on all NUMERIC tokens from a page.

    Uses original Surya full-page OCR results as primary signal.
    Locks high-confidence tokens. CNN-validates low-confidence ones.

    Args:
        page_tokens: PageTokens with extracted tokens.
        full_page_image: Full page rendered at DIGIT_CROP_DPI (BGR).

    Returns:
        List of TokenOCRResult for each numeric token.
    """
    results = []
    numeric_tokens = [t for t in page_tokens.tokens
                      if t.token_type == "NUMERIC"]

    for tok in numeric_tokens:
        result = ocr_numeric_token(
            full_page_image=full_page_image,
            token_bbox_pt=tok.bbox,
            page_width=page_tokens.page_width,
            page_height=page_tokens.page_height,
            original_text=tok.text,
            original_confidence=tok.confidence,
        )
        results.append(result)

    return results
