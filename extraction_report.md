# Extraction Report

**Date**: 2026-02-16 13:34:51

---

## OCR Tools Used

- **Gemini 2.5 Flash**: Multimodal extraction (text + tables)
- **PyMuPDF (fitz)**: PDF rendering at 300 DPI

## Gemini API Usage

- **API Calls**: 18 (one per page)
- **Model**: gemini-2.5-flash
- **Rate Limiting**: 2s delay between pages

## Extraction Statistics

- **Total Information Units**: 417
- **Sentence-Table Units**: 219
- **Text-Only Units**: 198
- **Pages Processed**: 18

## Known Uncertainties

- Gemini-based extraction may have alignment uncertainties
- OCR confidence scores are model-estimated
- Complex table structures may need manual verification

## Quality Notes

- ✅ No data invention - Gemini extracts only visible content
- ✅ Arabic text preserved exactly as written
- ✅ Numbers kept as strings (no conversion)
- ✅ Noise (stamps, logos) filtered by Gemini

---

**Status**: Extraction complete. Ready for hybrid retrieval.
