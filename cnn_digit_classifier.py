#!/usr/bin/env python3
"""
cnn_digit_classifier.py — Lightweight CNN Arabic-Indic Digit Classifier
========================================================================

Validation-only classifier for Arabic-Indic digits {٠–٩}.

STRICT USAGE RULES:
  - The CNN is NEVER a primary recognizer.
  - It may only be used when Surya confidence < 0.92.
  - Output may CONFIRM Surya output or mark UNTRUSTED.
  - Output may NOT override high-confidence Surya digits.
  - Output may NOT generate multi-digit strings.

Architecture:
  - Input:  32×32 grayscale single-digit crop
  - Output: 10-class softmax (٠–٩)
  - Lightweight: ~15K parameters

The CNN confirms digit identity by classifying individual glyph
crops and comparing against the Surya prediction.

Usage:
    from cnn_digit_classifier import classify_digit_crop
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

# Arabic-Indic digit labels (index → digit)
DIGIT_LABELS = ['٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩']
DIGIT_TO_INDEX = {d: i for i, d in enumerate(DIGIT_LABELS)}

# Model path
MODEL_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = MODEL_DIR / "cnn_digit_model.npz"

# Input size for CNN
INPUT_SIZE = 32

# Confidence threshold for CNN confirmation
CNN_CONFIDENCE_THRESHOLD = 0.70


# ────────────────────────────────────────────────────────────────────
#  Lightweight CNN (NumPy-only inference)
# ────────────────────────────────────────────────────────────────────
# This CNN uses only NumPy for inference — no PyTorch/TensorFlow needed.
# Weights are stored in a .npz file.
# Architecture: Conv(3x3,16) → ReLU → MaxPool → Conv(3x3,32) → ReLU →
#               MaxPool → FC(512,64) → ReLU → FC(64,10)

class SimpleCNN:
    """Minimal CNN for Arabic-Indic digit classification.

    Pure NumPy inference — no deep learning framework required.
    """

    def __init__(self):
        self.loaded = False
        self.weights = {}

    def load(self, path: str) -> bool:
        """Load pre-trained weights from .npz file."""
        try:
            data = np.load(path, allow_pickle=True)
            self.weights = {k: data[k] for k in data.files}
            self.loaded = True
            return True
        except Exception:
            self.loaded = False
            return False

    def _conv2d(self, x: np.ndarray, w: np.ndarray,
                b: np.ndarray) -> np.ndarray:
        """2D convolution (valid padding)."""
        n_filters, _, kh, kw = w.shape
        h, wi = x.shape[1], x.shape[2]
        oh, ow = h - kh + 1, wi - kw + 1

        out = np.zeros((n_filters, oh, ow), dtype=np.float32)
        for f in range(n_filters):
            for c in range(x.shape[0]):
                for i in range(oh):
                    for j in range(ow):
                        out[f, i, j] += np.sum(
                            x[c, i:i+kh, j:j+kw] * w[f, c])
            out[f] += b[f]
        return out

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _maxpool2d(self, x: np.ndarray, size: int = 2) -> np.ndarray:
        """2×2 max pooling."""
        c, h, w = x.shape
        oh, ow = h // size, w // size
        out = np.zeros((c, oh, ow), dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                out[:, i, j] = x[:, i*size:(i+1)*size,
                                   j*size:(j+1)*size].reshape(c, -1).max(axis=1)
        return out

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def predict(self, img: np.ndarray) -> Optional[Dict]:
        """Run inference on a 32×32 grayscale image.

        Returns dict with 'digit' (str), 'confidence' (float),
        'probabilities' (list of 10 floats), or None on failure.
        """
        if not self.loaded:
            return None

        try:
            # Normalize to [0, 1] and add channel dimension
            x = img.astype(np.float32) / 255.0
            x = x.reshape(1, INPUT_SIZE, INPUT_SIZE)  # (1, 32, 32)

            # Conv1: (1,32,32) → (16,30,30) → ReLU → MaxPool → (16,15,15)
            x = self._conv2d(x, self.weights['conv1_w'],
                             self.weights['conv1_b'])
            x = self._relu(x)
            x = self._maxpool2d(x)

            # Conv2: (16,15,15) → (32,13,13) → ReLU → MaxPool → (32,6,6)
            x = self._conv2d(x, self.weights['conv2_w'],
                             self.weights['conv2_b'])
            x = self._relu(x)
            x = self._maxpool2d(x)

            # Flatten
            x = x.flatten()

            # FC1: 32*6*6=1152 → 64
            x = x @ self.weights['fc1_w'].T + self.weights['fc1_b']
            x = self._relu(x)

            # FC2: 64 → 10
            x = x @ self.weights['fc2_w'].T + self.weights['fc2_b']
            probs = self._softmax(x)

            idx = int(np.argmax(probs))
            return {
                'digit': DIGIT_LABELS[idx],
                'confidence': float(probs[idx]),
                'probabilities': probs.tolist(),
            }
        except Exception:
            return None


# Singleton model instance
_model: Optional[SimpleCNN] = None


def _get_model() -> Optional[SimpleCNN]:
    """Load CNN model (singleton)."""
    global _model
    if _model is not None:
        return _model if _model.loaded else None

    _model = SimpleCNN()
    if MODEL_PATH.exists():
        _model.load(str(MODEL_PATH))
        return _model if _model.loaded else None
    return None


# ────────────────────────────────────────────────────────────────────
#  Glyph Segmentation for Multi-Digit Tokens
# ────────────────────────────────────────────────────────────────────
def _segment_individual_digits(crop: np.ndarray) -> List[np.ndarray]:
    """Segment a crop into individual digit glyphs.

    Uses connected components to isolate individual characters.
    Returns list of 32×32 grayscale images, sorted left-to-right.
    """
    if crop.size == 0:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Ensure dark text on light background
    if np.mean(gray) < 128:
        gray = 255 - gray

    # Binarize
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connected components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    glyphs = []
    img_h, img_w = binary.shape[:2]

    for i in range(1, n_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # Filter noise (too small) and non-digit (too large)
        if area < 15 or h < 5:
            continue
        if h > img_h * 0.95 and w > img_w * 0.95:
            continue  # Covers entire crop — not a digit

        # Extract glyph with padding
        pad = 4
        gx0 = max(0, x - pad)
        gy0 = max(0, y - pad)
        gx1 = min(img_w, x + w + pad)
        gy1 = min(img_h, y + h + pad)

        glyph = gray[gy0:gy1, gx0:gx1]

        # Resize to INPUT_SIZE × INPUT_SIZE
        glyph_resized = cv2.resize(glyph, (INPUT_SIZE, INPUT_SIZE),
                                   interpolation=cv2.INTER_AREA)

        glyphs.append((centroids[i][0], glyph_resized))

    # Sort left-to-right by x-centroid
    glyphs.sort(key=lambda g: g[0])
    return [g[1] for g in glyphs]


# ────────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────────
def classify_digit_crop(
    crop: np.ndarray,
    surya_digits: str,
) -> Optional[Dict]:
    """Classify digit(s) in a crop and compare against Surya prediction.

    This function is VALIDATION-ONLY. It:
      - Segments the crop into individual glyphs
      - Classifies each glyph via CNN
      - Compares CNN sequence against surya_digits
      - Returns {'confirmed': True/False, 'cnn_digits': str,
                 'confidence': float}

    Returns None if CNN model is not available.

    Args:
        crop: BGR or grayscale image of the token region.
        surya_digits: Normalized Arabic-Indic digit string from Surya.

    Returns:
        Dict with 'confirmed', 'cnn_digits', 'confidence', or None.
    """
    model = _get_model()
    if model is None:
        return None  # CNN not available — caller keeps existing status

    # Segment into individual digits
    digit_images = _segment_individual_digits(crop)

    if not digit_images:
        return {
            'confirmed': False,
            'cnn_digits': '',
            'confidence': 0.0,
        }

    # Classify each glyph
    cnn_digits = []
    confidences = []

    for glyph_img in digit_images:
        result = model.predict(glyph_img)
        if result is None:
            return None  # Model inference failed

        cnn_digits.append(result['digit'])
        confidences.append(result['confidence'])

    cnn_sequence = ''.join(cnn_digits)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Compare against Surya prediction
    confirmed = (
        cnn_sequence == surya_digits
        and avg_confidence >= CNN_CONFIDENCE_THRESHOLD
    )

    return {
        'confirmed': confirmed,
        'cnn_digits': cnn_sequence,
        'confidence': avg_confidence,
    }
