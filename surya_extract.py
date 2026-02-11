#!/usr/bin/env python3
"""
surya_extract.py — Full PDF extraction using Surya OCR + PyMuPDF
================================================================
Pages 1-3  (scanned)  → preprocess + Surya OCR
Pages 4-18 (native)   → PyMuPDF direct text extraction (instant, lossless)

Output:  surya_results.txt
"""

import os
import sys
import time
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

import cv2
import fitz  # PyMuPDF
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PDF_PATH   = os.path.join(BASE_DIR, "el-bankalahly .pdf")
OUTPUT     = os.path.join(BASE_DIR, "surya_results.txt")
DPI        = 300
SCANNED    = {1, 2, 3}  # 1-indexed page numbers that are scanned images


# ── Preprocessing (for scanned pages only) ───────────────────────────
def render_page(doc, page_idx: int) -> np.ndarray:
    page = doc[page_idx]
    zoom = DPI / 72.0
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
               pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def clean_image(img: np.ndarray) -> np.ndarray:
    """Background removal + CLAHE + bilateral denoise + sharpen."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # Background estimation via large morphological closing
    bg_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, bg_kernel)

    # Background subtraction → normalise lighting
    diff = cv2.subtract(background, gray)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    clean = cv2.bitwise_not(diff)

    # CLAHE contrast boost
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    clean = clahe.apply(clean)

    # Bilateral filter (edge-preserving denoise)
    clean = cv2.bilateralFilter(clean, d=9, sigmaColor=75, sigmaSpace=75)

    # Unsharp mask
    blurred = cv2.GaussianBlur(clean, (0, 0), sigmaX=3)
    sharp   = cv2.addWeighted(clean, 1.5, blurred, -0.5, 0)

    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


# ── Surya OCR (models loaded once) ───────────────────────────────────
_foundation = None
_det = None
_rec = None


def load_surya():
    global _foundation, _det, _rec
    if _foundation is not None:
        return
    print('  [Surya] Loading models (one-time)...', flush=True)
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    _foundation = FoundationPredictor(device='cpu')
    _det        = DetectionPredictor(device='cpu')
    _rec        = RecognitionPredictor(_foundation)
    print('  [Surya] Models loaded.\n', flush=True)


def surya_ocr(img_bgr: np.ndarray) -> str:
    """Run Surya OCR on a BGR numpy image. Returns extracted text."""
    load_surya()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    predictions = _rec([pil_img],
                       task_names=["ocr_without_boxes"],
                       det_predictor=_det)

    lines = []
    if predictions and len(predictions) > 0:
        for tl in predictions[0].text_lines:
            lines.append(tl.text)
    return '\n'.join(lines)


# ── Native text extraction (for vector pages) ────────────────────────
def native_text(doc, page_idx: int) -> str:
    """Extract text directly from a native-vector PDF page via PyMuPDF."""
    page = doc[page_idx]
    text = page.get_text("text")
    return text.strip()


# ── Main ─────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    print('=' * 62)
    print('  SURYA FULL-PDF EXTRACTION')
    print(f'  PDF: {os.path.basename(PDF_PATH)}')
    print(f'  Scanned pages: {sorted(SCANNED)} → Surya OCR')
    print(f'  Native pages:  4-18 → PyMuPDF text extraction')
    print(f'  Output: {OUTPUT}')
    print('=' * 62)

    if not os.path.exists(PDF_PATH):
        print(f'\n  [ERROR] PDF not found: {PDF_PATH}')
        sys.exit(1)

    doc = fitz.open(PDF_PATH)
    total_pages = doc.page_count
    print(f'\n  PDF loaded: {total_pages} pages\n')

    out = open(OUTPUT, 'w', encoding='utf-8')
    out.write(f'{"=" * 72}\n')
    out.write(f'  SURYA EXTRACTION — {os.path.basename(PDF_PATH)}\n')
    out.write(f'  Scanned pages (Surya OCR): {sorted(SCANNED)}\n')
    out.write(f'  Native pages (PyMuPDF):    {list(range(4, total_pages + 1))}\n')
    out.write(f'{"=" * 72}\n\n')
    out.flush()

    total_chars = 0
    total_arabic = 0

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        t0 = time.time()

        if page_num in SCANNED:
            # ── Scanned page: render → clean → Surya OCR ──
            print(f'  Page {page_num:2d}/{total_pages}  [scanned] ', end='', flush=True)
            img = render_page(doc, page_idx)
            cleaned = clean_image(img)
            text = surya_ocr(cleaned)
            method = 'Surya OCR'
        else:
            # ── Native page: direct text extraction ──
            print(f'  Page {page_num:2d}/{total_pages}  [native]  ', end='', flush=True)
            text = native_text(doc, page_idx)
            method = 'PyMuPDF'

        elapsed = time.time() - t0
        char_count = len(text)
        arabic_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        total_chars += char_count
        total_arabic += arabic_count

        print(f'{elapsed:6.1f}s  {char_count:5d} chars  '
              f'{arabic_count:4d} Arabic  ({method})')

        # Write to output
        out.write(f'{"─" * 72}\n')
        out.write(f'  PAGE {page_num}  ({method})\n')
        out.write(f'{"─" * 72}\n')
        out.write(text)
        out.write('\n\n')
        out.flush()

    doc.close()

    elapsed_total = time.time() - t_start
    out.write(f'\n{"=" * 72}\n')
    out.write(f'  Total: {total_chars} chars, {total_arabic} Arabic chars\n')
    out.write(f'  Time:  {elapsed_total:.1f}s\n')
    out.write(f'{"=" * 72}\n')
    out.close()

    print(f'\n{"=" * 62}')
    print(f'  ✓ Done in {elapsed_total:.1f}s')
    print(f'  ✓ Total: {total_chars} chars ({total_arabic} Arabic)')
    print(f'  ✓ Output: {OUTPUT}')
    print(f'{"=" * 62}')


if __name__ == '__main__':
    main()
