#!/usr/bin/env python3
"""
preprocess_scanned.py — Background Removal & Multi-OCR for Scanned Pages 1-3
=============================================================================
Targets the first 3 pages of "el-bankalahly .pdf" which are scanned images.

Pipeline:
  1. Render each page at 300 DPI via PyMuPDF
  2. Apply multiple preprocessing variants:
     a) ORIGINAL     — raw render (baseline)
     b) CLEAN        — background removal + denoising + text enhancement
     c) BINARY_ADAPT — adaptive Gaussian binarisation
     d) BINARY_OTSU  — Otsu binarisation + morphological repair
     e) SAUVOLA      — Sauvola local binarisation (best for uneven lighting)
  3. Save all enhanced images to  scanned_pages/
  4. Run 3 OCR engines on the CLEAN variant:
     - PaddleOCR v3 (PP-OCRv5, Arabic)
     - Tesseract 5 (LSTM, Arabic)
     - Surya (transformer-based multilingual)
  5. Compare results side-by-side in a report

Output:
  scanned_pages/
    page1_original.png
    page1_clean.png
    page1_binary_adaptive.png
    page1_binary_otsu.png
    page1_sauvola.png
    page2_...
    page3_...
  scanned_pages/ocr_comparison.txt   — side-by-side OCR results
"""

import os
import sys
import time
import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    import fitz  # PyMuPDF
except ImportError:
    print("[ERROR] PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)


# ====================================================================
#  Configuration
# ====================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PDF_PATH   = os.path.join(BASE_DIR, "el-bankalahly .pdf")
OUTPUT_DIR = os.path.join(BASE_DIR, "scanned_pages")
PAGES      = [0, 1, 2]   # 0-indexed: pages 1, 2, 3
DPI        = 300


# ====================================================================
#  Preprocessing functions
# ====================================================================
def render_page(doc, page_idx: int) -> np.ndarray:
    """Render a PDF page to a numpy array at the target DPI."""
    page = doc[page_idx]
    zoom = DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    if len(img.shape) == 3 and img.shape[2] >= 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def remove_background_and_enhance(img: np.ndarray) -> np.ndarray:
    """
    Heavy-duty background removal + text enhancement pipeline:
      1. Convert to grayscale
      2. Estimate background via large morphological closing (60×60)
         → this captures the background colour/gradient
      3. Subtract background: text = gray - background + 255
         → normalises uneven lighting, removes coloured backgrounds
      4. CLAHE contrast enhancement (clipLimit=3.0, 16×16 grid)
         → boosts faint text
      5. Bilateral filter (denoise while preserving edges)
         → smooths paper texture noise without blurring characters
      6. Sharpen with unsharp mask
         → crisps up character edges for better OCR
    Returns a clean BGR image (3-channel for consistent saving).
    """
    gray = to_gray(img)

    # --- Step 1: Background estimation via morphological closing ---
    # Large kernel captures the "background" (paper colour, gradients, stains)
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, bg_kernel)

    # --- Step 2: Background subtraction ---
    # Normalize: text becomes dark on white regardless of original background
    normalised = cv2.subtract(background, gray)
    # Invert so text = dark, bg = white  → then stretch to full range
    normalised = cv2.normalize(normalised, None, 0, 255, cv2.NORM_MINMAX)
    # Invert: white bg, dark text
    clean = cv2.bitwise_not(normalised)

    # --- Step 3: CLAHE contrast enhancement ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    clean = clahe.apply(clean)

    # --- Step 4: Bilateral filter (edge-preserving denoise) ---
    clean = cv2.bilateralFilter(clean, d=9, sigmaColor=75, sigmaSpace=75)

    # --- Step 5: Unsharp mask for text edge sharpening ---
    blurred = cv2.GaussianBlur(clean, (0, 0), sigmaX=3)
    sharp = cv2.addWeighted(clean, 1.5, blurred, -0.5, 0)

    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def binarize_adaptive(img: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian binarisation — good for variable backgrounds."""
    gray = to_gray(img)
    # Denoise first
    denoised = cv2.fastNlMeansDenoising(gray, h=15)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=12)
    # Morphological close: repair broken Arabic ligatures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def binarize_otsu(img: np.ndarray) -> np.ndarray:
    """Otsu's automatic threshold — good for bimodal histograms."""
    gray = to_gray(img)
    # Gaussian blur to reduce noise before Otsu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological close for Arabic ligatures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def binarize_sauvola(img: np.ndarray, window_size: int = 51,
                     k: float = 0.2) -> np.ndarray:
    """
    Sauvola local binarisation — best for scanned documents with
    uneven lighting, shadows, or coloured backgrounds.

    T(x,y) = mean(x,y) * [1 + k * (std(x,y)/R - 1)]
    where R = max(std) = 128 for 8-bit images.
    """
    gray = to_gray(img).astype(np.float64)

    # Compute local mean and std using box filter
    mean = cv2.blur(gray, (window_size, window_size))
    mean_sq = cv2.blur(gray * gray, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))

    R = 128.0
    threshold = mean * (1.0 + k * (std / R - 1.0))

    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)

    # Morphological close for Arabic ligatures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ====================================================================
#  OCR engines
# ====================================================================
def ocr_paddleocr(img_bgr: np.ndarray) -> str:
    """
    Run PaddleOCR v3 (PP-OCRv5) with Arabic support.
    NOTE: PaddlePaddle 3.3.0 has a PIR/OneDNN CPU inference bug.
    We fall back to the CLI tool which may use a different code path.
    """
    import subprocess, tempfile
    # Save image to a temp file, run paddleocr CLI
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    cv2.imwrite(tmp.name, img_bgr)
    tmp.close()
    try:
        bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin')
        paddleocr_bin = os.path.join(bin_dir, 'paddleocr')
        env = os.environ.copy()
        env['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        result = subprocess.run(
            [paddleocr_bin, '--lang', 'ar', '--image_path', tmp.name],
            capture_output=True, text=True, timeout=120, env=env)
        return result.stdout.strip()
    except Exception as e:
        return f"[PaddleOCR CLI error: {e}]"
    finally:
        os.unlink(tmp.name)


def ocr_tesseract(img_bgr: np.ndarray) -> str:
    """Run Tesseract 5 LSTM with Arabic language model (PSM 6 — uniform block)."""
    import pytesseract
    gray = to_gray(img_bgr)
    config = '--oem 1 --psm 6 -l ara'
    text = pytesseract.image_to_string(gray, config=config)
    return text.strip()


def ocr_tesseract_auto(img_bgr: np.ndarray) -> str:
    """Run Tesseract 5 LSTM (PSM 3 — fully automatic page segmentation)."""
    import pytesseract
    gray = to_gray(img_bgr)
    config = '--oem 1 --psm 3 -l ara'
    text = pytesseract.image_to_string(gray, config=config)
    return text.strip()


def ocr_surya(img_bgr: np.ndarray) -> str:
    """Run Surya OCR (transformer-based multilingual)."""
    global _surya_foundation, _surya_det, _surya_rec
    from PIL import Image

    # Lazy-load models once (very expensive on CPU)
    if '_surya_foundation' not in globals() or _surya_foundation is None:
        print('\n      [Surya] Loading models (one-time)...', flush=True)
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        _surya_foundation = FoundationPredictor(device='cpu')
        _surya_det = DetectionPredictor(device='cpu')
        _surya_rec = RecognitionPredictor(_surya_foundation)
        print('      [Surya] Models loaded.', flush=True)

    # Convert BGR numpy to PIL RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Run detection + recognition
    predictions = _surya_rec([pil_img],
                             task_names=["ocr_without_boxes"],
                             det_predictor=_surya_det)

    lines = []
    if predictions and len(predictions) > 0:
        pred = predictions[0]
        for text_line in pred.text_lines:
            lines.append(text_line.text)
    return '\n'.join(lines)

_surya_foundation = None
_surya_det = None
_surya_rec = None


# ====================================================================
#  Main pipeline
# ====================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 62)
    print('  SCANNED PAGE PREPROCESSING & MULTI-OCR COMPARISON')
    print(f'  PDF: {os.path.basename(PDF_PATH)}')
    print(f'  Pages: 1, 2, 3  (scanned)')
    print(f'  DPI: {DPI}')
    print(f'  Output: {OUTPUT_DIR}/')
    print('=' * 62)

    if not os.path.exists(PDF_PATH):
        print(f'\n  [ERROR] PDF not found: {PDF_PATH}')
        sys.exit(1)

    doc = fitz.open(PDF_PATH)
    print(f'\n  PDF loaded: {doc.page_count} pages')

    # ── Step 1: Render and preprocess each page ──────────────────────
    preprocessing_methods = [
        ('original',        lambda img: img),
        ('clean',           remove_background_and_enhance),
        ('binary_adaptive', binarize_adaptive),
        ('binary_otsu',     binarize_otsu),
        ('sauvola',         binarize_sauvola),
    ]

    page_images = {}  # {page_num: {variant_name: img}}

    for page_idx in PAGES:
        page_num = page_idx + 1
        print(f'\n  ──── Page {page_num} ────')

        # Render at 300 DPI
        img = render_page(doc, page_idx)
        h, w = img.shape[:2]
        print(f'    Rendered: {w}×{h} px @ {DPI} DPI')

        page_images[page_num] = {}

        for variant_name, func in preprocessing_methods:
            t0 = time.time()
            processed = func(img)
            elapsed = time.time() - t0

            # Save
            fname = f'page{page_num}_{variant_name}.png'
            fpath = os.path.join(OUTPUT_DIR, fname)
            cv2.imwrite(fpath, processed)
            size_kb = os.path.getsize(fpath) / 1024
            print(f'    ✓ {variant_name:20s} → {fname} '
                  f'({size_kb:.0f} KB, {elapsed:.2f}s)')

            page_images[page_num][variant_name] = processed

    doc.close()

    # ── Step 2: Run OCR comparison on the CLEAN variant ──────────────
    print(f'\n{"=" * 62}')
    print('  RUNNING OCR COMPARISON (on "clean" variant)')
    print(f'{"=" * 62}')

    ocr_engines = [
        ('PaddleOCR_v3',    ocr_paddleocr),
        ('Tesseract_PSM6',  ocr_tesseract),
        ('Tesseract_PSM3',  ocr_tesseract_auto),
        ('Surya',           ocr_surya),
    ]

    report_path = os.path.join(OUTPUT_DIR, 'ocr_comparison.txt')
    report_f = open(report_path, 'w', encoding='utf-8')
    report_f.write('=' * 72 + '\n')
    report_f.write('  OCR COMPARISON REPORT — Scanned Pages 1-3\n')
    report_f.write(f'  Input: "clean" preprocessed variant @ {DPI} DPI\n')
    report_f.write(f'  Engines: {", ".join(e[0] for e in ocr_engines)}\n')
    report_f.write('=' * 72 + '\n\n')
    report_f.flush()

    for page_idx in PAGES:
        page_num = page_idx + 1
        clean_img = page_images[page_num]['clean']

        print(f'\n  ──── Page {page_num} OCR ────')
        report_f.write(f'{"─" * 72}\n')
        report_f.write(f'  PAGE {page_num}\n')
        report_f.write(f'{"─" * 72}\n\n')
        report_f.flush()

        for engine_name, engine_func in ocr_engines:
            print(f'    Running {engine_name:15s} ...', end=' ', flush=True)
            t0 = time.time()

            try:
                text = engine_func(clean_img)
                elapsed = time.time() - t0
                line_count = text.count('\n') + 1
                char_count = len(text)
                arabic_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF')

                print(f'{elapsed:.1f}s  '
                      f'({line_count} lines, {char_count} chars, '
                      f'{arabic_count} Arabic)')

                report_f.write(f'  ┌─ {engine_name} ({elapsed:.1f}s) ─\n')
                report_f.write(f'  │  Lines: {line_count}, '
                               f'Chars: {char_count}, '
                               f'Arabic: {arabic_count}\n')
                report_f.write(f'  └{"─" * 50}\n')
                for line in text.split('\n'):
                    report_f.write(f'    {line}\n')
                report_f.write('\n')
                report_f.flush()

            except Exception as e:
                elapsed = time.time() - t0
                print(f'FAILED ({elapsed:.1f}s): {e}')
                report_f.write(f'  ┌─ {engine_name} — FAILED ─\n')
                report_f.write(f'  │  Error: {e}\n')
                report_f.write(f'  └{"─" * 50}\n\n')
                report_f.flush()

    report_f.write('\n' + '=' * 72 + '\n')
    report_f.write('  END OF REPORT\n')
    report_f.write('=' * 72 + '\n')
    report_f.close()

    # ── Summary ──────────────────────────────────────────────────────
    img_count = len(PAGES) * len(preprocessing_methods)
    print(f'\n{"=" * 62}')
    print(f'  ✓ Images saved:  {img_count} files in {OUTPUT_DIR}/')
    print(f'  ✓ OCR report:    {report_path}')
    print(f'  ✓ Variants:      {", ".join(m[0] for m in preprocessing_methods)}')
    print(f'  ✓ OCR engines:   {", ".join(e[0] for e in ocr_engines)}')
    print(f'\n  💡 Compare images:  ls {OUTPUT_DIR}/page*_*.png')
    print(f'  💡 Read OCR report: cat {OUTPUT_DIR}/ocr_comparison.txt')
    print(f'{"=" * 62}')


if __name__ == '__main__':
    main()
