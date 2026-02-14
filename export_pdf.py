#!/usr/bin/env python3
"""
export_pdf.py — Re-export the rendered HTML back to PDF via headless Chromium.

Uses Playwright to open layout.html in a headless browser and print each
page to PDF with exact dimensions, zero margins, and no headers/footers.

The HTML contains one .page-wrapper per PDF page, each with known dimensions
stored in data-pw / data-ph attributes (in PDF points = CSS pixels at 1x).

Strategy
--------
1. Open the HTML file in Chromium.
2. For each .page element, read its data-pw and data-ph.
3. If all pages share the same dimensions, export as a single PDF.
4. If pages have different dimensions (e.g. landscape page 7), export
   per-page PDFs and merge them with PyMuPDF.

The @media print CSS in the HTML removes badges, shadows, and resets
transforms so the PDF captures native-resolution text positions.

Usage
-----
    python export_pdf.py
    python export_pdf.py --input layout.html --output layout_export.pdf
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path


DEFAULT_INPUT  = "layout.html"
DEFAULT_OUTPUT = "layout_export.pdf"


async def export_single_pdf(html_path: Path, output_path: Path):
    """Export the full HTML as a single multi-page PDF."""
    from playwright.async_api import async_playwright

    html_url = html_path.resolve().as_uri()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()

        print(f"Loading {html_url} ...")
        await page.goto(html_url, wait_until="networkidle")

        # Read page dimensions from the first .page element
        dims = await page.evaluate("""
            () => {
                const pages = document.querySelectorAll('.page');
                const result = [];
                pages.forEach(p => {
                    result.push({
                        w: parseFloat(p.dataset.pw),
                        h: parseFloat(p.dataset.ph)
                    });
                });
                return result;
            }
        """)

        if not dims:
            print("Error: no .page elements found in HTML.", file=sys.stderr)
            await browser.close()
            sys.exit(1)

        # Check if all pages have the same dimensions
        all_same = all(
            abs(d["w"] - dims[0]["w"]) < 1 and abs(d["h"] - dims[0]["h"]) < 1
            for d in dims
        )

        if all_same:
            # Simple case: single PDF with uniform page size
            w_pt = dims[0]["w"]
            h_pt = dims[0]["h"]
            # Playwright page.pdf() expects dimensions in CSS pixels or
            # named sizes.  PDF points = CSS pixels at 96 dpi... but
            # Playwright uses 1px = 1/96 inch by default.
            # We want 1pt = 1/72 inch.
            # So we must convert: width_in_inches = w_pt / 72
            #                     width_in_px_for_playwright = w_pt / 72 * 96
            # Or just pass as inches.
            w_in = w_pt / 72
            h_in = h_pt / 72

            print(f"All {len(dims)} pages are {w_pt:.0f}×{h_pt:.0f} pt "
                  f"({w_in:.2f}×{h_in:.2f} in)")

            await page.pdf(
                path=str(output_path),
                width=f"{w_in:.4f}in",
                height=f"{h_in:.4f}in",
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                print_background=True,
                scale=1.0,
            )
            print(f"Exported: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

        else:
            # Mixed dimensions (e.g. portrait + landscape).
            # Export each page separately, then merge with PyMuPDF.
            print(f"Mixed page sizes detected ({len(dims)} pages). Exporting per-page...")

            tmp_dir = output_path.parent / "_tmp_pdf_pages"
            tmp_dir.mkdir(exist_ok=True)
            tmp_files = []

            for i, d in enumerate(dims):
                w_in = d["w"] / 72
                h_in = d["h"] / 72

                # Hide all pages except the current one
                await page.evaluate(f"""
                    (idx) => {{
                        document.querySelectorAll('.page-wrapper').forEach((el, j) => {{
                            el.style.display = j === idx ? 'block' : 'none';
                        }});
                    }}
                """, i)

                tmp_path = tmp_dir / f"page_{i+1:03d}.pdf"
                await page.pdf(
                    path=str(tmp_path),
                    width=f"{w_in:.4f}in",
                    height=f"{h_in:.4f}in",
                    margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                    print_background=True,
                    scale=1.0,
                )
                tmp_files.append(tmp_path)
                print(f"  Page {i+1}: {d['w']:.0f}×{d['h']:.0f} pt → {tmp_path.name}")

            # Restore visibility
            await page.evaluate("""
                () => {
                    document.querySelectorAll('.page-wrapper').forEach(el => {
                        el.style.display = 'block';
                    });
                }
            """)

            await browser.close()

            # Merge with PyMuPDF
            import fitz
            merged = fitz.open()
            for tf in tmp_files:
                src = fitz.open(str(tf))
                merged.insert_pdf(src)
                src.close()
            merged.save(str(output_path))
            merged.close()

            # Cleanup
            for tf in tmp_files:
                tf.unlink()
            tmp_dir.rmdir()

            print(f"Merged: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
            return

        await browser.close()


def main():
    parser = argparse.ArgumentParser(description="Export HTML to PDF via Playwright.")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT)
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Error: {inp} not found.", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output)
    asyncio.run(export_single_pdf(inp, out))


if __name__ == "__main__":
    main()
