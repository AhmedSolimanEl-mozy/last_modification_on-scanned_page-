#!/usr/bin/env python3
"""
render_html.py — Convert normalized layout JSONL into a visually-faithful HTML.

Architecture
============
This renderer addresses all known layout failure modes:

1. DYNAMIC PAGE SIZING
   Page dimensions come from normalize_layout.py (content bbox + margin).
   The renderer never hard-codes page size — it reads page_width/page_height
   from each JSON object.

2. COORDINATE NORMALIZATION
   All coordinates in the JSONL are already in PDF points (1pt = 1/72 in).
   The renderer maps 1 PDF point → 1 CSS pixel.  Visual scaling is handled
   by a CSS `transform: scale(S)` on the page container, keeping inline
   coordinates untouched.

3. RTL POSITIONING STRATEGY  (the key fix)
   Problem:  `text-align: right` + `width: bbox_width` — if the browser
   font is wider than the PDF font (Arial vs TimesNewRoman, ~10-30% wider),
   text overflows LEFT past x0 and gets cropped.

   Solution: We position each line at `left: x0` and compute an EXPANDED
   width that accounts for font metric drift.  For RTL lines:
       render_width = max(bbox_width, bbox_width × FONT_DRIFT_FACTOR)
   The extra width extends LEFTWARD (toward x=0) by shifting left:
       adjusted_left = max(0, x0 - (render_width - bbox_width))
   This gives the browser enough room to render the full text right-aligned
   without clipping.

4. OVERFLOW PROTECTION
   - `.page` has `overflow: visible` (no clipping inside the page)
   - `.page-wrapper` has `overflow: hidden` (clips at page boundary)
   This lets text breathe within the page but prevents it from bleeding
   into adjacent pages.

5. LINE-HEIGHT STABILIZATION
   line-height = bbox_height × 1.0 (exact bbox match)
   font-size from metadata or bbox_height × 0.85
   line-height is never less than font-size

6. RESPONSIVE CSS SCALING
   JavaScript computes scale = min(viewport_width / page_width, 1.0)
   Applied via CSS custom property --scale on each .page-wrapper
   Updated on window resize.  Never scales above 1.0.

7. PRINT / PDF RE-EXPORT CSS
   @media print styles:
   - Removes background, shadows
   - page-break-after: always per page
   - Exact page dimensions via @page size
   - margin: 0 for pixel-perfect re-export

Usage
-----
    python render_html.py
    python render_html.py --input layout_normalized.jsonl --output layout.html --scale 1.5
    python render_html.py --pages 1,4 --scale 2.0
"""

import argparse
import json
import html
import sys
from pathlib import Path


# ─── Configuration ──────────────────────────────────────────────────────────

DEFAULT_INPUT  = "layout_normalized.jsonl"
DEFAULT_OUTPUT = "layout.html"
DEFAULT_SCALE  = 1.5
DEFAULT_FONT   = '"Arial", "Helvetica", "Noto Sans Arabic", sans-serif'

# Font drift compensation: browser Arial is ~15-25% wider than PDF TimesNewRoman
# for Arabic text.  We expand the RTL line width by this factor.
FONT_DRIFT_FACTOR  = 1.25
FONT_SIZE_FACTOR   = 0.85   # bbox_height → font-size (for OCR lines)


# ─── HTML Template ──────────────────────────────────────────────────────────

HTML_HEAD = """\
<!DOCTYPE html>
<html lang="ar" dir="ltr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Layout Extraction — Visual Render</title>
<style>
/*
 * Coordinate system:
 *   All positions in PDF points (1pt = 1/72in = 1 CSS px at scale 1.0).
 *   Visual scaling via CSS transform on .page containers.
 *
 * RTL strategy:
 *   <html dir="ltr"> — absolute coordinates, not flow layout.
 *   RTL lines get direction:rtl + text-align:right per-line.
 *   Line width is EXPANDED by {drift_pct}% to absorb font metric drift.
 *   The expansion shifts the left edge leftward so the right edge (x1)
 *   stays anchored — Arabic text starts at x1 and flows left.
 */

* {{
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}}

body {{
  background: #d0d0d0;
  font-family: {font_family};
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 30px 10px;
  direction: ltr;
}}

.page-label {{
  font-size: 14px;
  color: #555;
  margin-bottom: 6px;
  font-family: monospace;
  direction: ltr;
  text-align: center;
}}

/* Outer wrapper: holds the scaled page dimensions.
 * overflow: hidden clips content at page boundary.
 * --scale is set per-page (responsive JS updates it). */
.page-wrapper {{
  margin-bottom: 30px;
  overflow: hidden;
  position: relative;
}}

/* Page surface: exact PDF-point dimensions.
 * overflow: visible — let text breathe, wrapper clips.
 * direction: ltr — coordinates use left-edge origin. */
.page {{
  position: relative;
  background: #ffffff;
  box-shadow: 0 2px 12px rgba(0,0,0,.25);
  overflow: visible;
  transform-origin: top left;
  direction: ltr;
}}

/* Text lines — absolutely positioned, no wrapping, no clipping. */
.line {{
  position: absolute;
  white-space: pre;
  /* vertical centering: line-height set inline to match bbox height */
}}

/* RTL: text right-aligns within the (expanded) width.
 * Arabic text anchors at the right edge and flows left. */
.line.rtl {{
  direction: rtl;
  text-align: right;
  unicode-bidi: plaintext;
}}

.line.ltr {{
  direction: ltr;
  text-align: left;
  unicode-bidi: plaintext;
}}

.line span {{
  unicode-bidi: plaintext;
}}

/* method-badge removed — was a debug overlay, not PDF content */

/* ─── Print / PDF re-export styles ─── */
@media print {{
  body {{
    background: none;
    padding: 0;
  }}
  .page-label,
  .method-badge {{
    display: none;
  }}
  .page-wrapper {{
    margin: 0 !important;
    overflow: visible !important;
    width: auto !important;
    height: auto !important;
    page-break-after: always;
    page-break-inside: avoid;
  }}
  .page {{
    box-shadow: none;
    transform: none !important;  /* print at native resolution */
  }}
}}
</style>
</head>
<body>
"""

HTML_TAIL = """\
<!-- ─── Responsive scaling ─── -->
<script>
(function() {{
  // Compute scale = min(viewport / page_width, BASE_SCALE) for each page.
  // BASE_SCALE is the authored scale (from --scale flag).
  var BASE = {base_scale};
  var wrappers = document.querySelectorAll('.page-wrapper');

  function rescale() {{
    var vw = window.innerWidth - 40;  // 20px padding each side
    wrappers.forEach(function(wr) {{
      var page = wr.querySelector('.page');
      if (!page) return;
      var pw = parseFloat(page.dataset.pw);
      // Never exceed BASE scale; shrink to fit viewport
      var s = Math.min(BASE, vw / pw);
      page.style.transform = 'scale(' + s + ')';
      wr.style.width  = (pw * s) + 'px';
      wr.style.height = (parseFloat(page.dataset.ph) * s) + 'px';
    }});
  }}

  window.addEventListener('resize', rescale);
  rescale();  // initial
}})();
</script>
</body>
</html>
"""


# ─── Helpers ────────────────────────────────────────────────────────────────

def get_font_size(line: dict) -> float:
    """Get font-size: from span metadata or estimate from bbox height."""
    for sp in line.get("spans", []):
        fs = sp.get("font_size")
        if fs is not None:
            return fs
    _, y0, _, y1 = line["bbox"]
    return max((y1 - y0) * FONT_SIZE_FACTOR, 4.0)


def sort_blocks(blocks: list) -> list:
    """Sort blocks top→bottom, RTL within horizontal bands."""
    if not blocks:
        return blocks
    srt = sorted(blocks, key=lambda b: (b["bbox"][1], -b["bbox"][0]))
    bands, cur = [], [srt[0]]
    for blk in srt[1:]:
        if abs(blk["bbox"][1] - cur[0]["bbox"][1]) <= 5:
            cur.append(blk)
        else:
            bands.append(cur)
            cur = [blk]
    bands.append(cur)
    out = []
    for band in bands:
        band.sort(key=lambda b: -b["bbox"][0])
        out.extend(band)
    return out


def render_line(line: dict, page_width: float) -> str:
    """Render one text line as an absolutely-positioned div.

    RTL font-drift compensation:
        The bbox says text runs from x0 to x1 (width = x1-x0).
        Browser fonts are wider → text overflows LEFT past x0.
        We expand the div width by FONT_DRIFT_FACTOR and shift its
        left edge leftward so x1 stays anchored:

            expanded_w = bbox_w × FONT_DRIFT_FACTOR
            extra      = expanded_w − bbox_w
            left       = max(0, x0 − extra)

        text-align:right keeps text pinned to x1.  The extra space
        absorbs the wider browser font on the left side.
    """
    x0, y0, x1, y1 = line["bbox"]
    bbox_w = x1 - x0
    height = y1 - y0
    direction = line.get("direction", "rtl")
    dir_class = "rtl" if direction == "rtl" else "ltr"
    font_size = get_font_size(line)

    # ── Width computation with font-drift compensation ──
    if direction == "rtl":
        expanded_w = bbox_w * FONT_DRIFT_FACTOR
        extra = expanded_w - bbox_w
        left = max(0, x0 - extra)
        # Ensure the div's right edge = x1
        # actual right = left + expanded_w
        # We want actual right >= x1, so expanded_w = x1 - left
        render_w = x1 - left
    else:
        left = x0
        render_w = bbox_w

    # line-height = bbox height for vertical centering; never < font-size
    line_height = max(height, font_size)

    # ── Build inner HTML ──
    spans = line.get("spans", [])
    has_multi_fonts = (
        len(spans) > 1
        and any(s.get("font_name") for s in spans)
        and len({s.get("font_name") for s in spans}) > 1
    )

    if has_multi_fonts:
        parts = []
        for sp in spans:
            sp_text = html.escape(sp.get("text", ""))
            sp_font = sp.get("font_name")
            sp_size = sp.get("font_size")
            bits = []
            if sp_font:
                bits.append(f'font-family:"{html.escape(sp_font)}", Arial, sans-serif')
            if sp_size and abs(sp_size - font_size) > 0.5:
                bits.append(f"font-size:{sp_size:.2f}px")
            if "Bold" in (sp_font or ""):
                bits.append("font-weight:bold")
            if "Italic" in (sp_font or ""):
                bits.append("font-style:italic")
            attr = f' style="{";".join(bits)}"' if bits else ""
            parts.append(f"<span{attr}>{sp_text}</span>")
        inner = "".join(parts)
    else:
        raw = line.get("text", "")
        inner = html.escape(raw).replace("&lt;br&gt;", "<br>")
        if spans and spans[0].get("font_name"):
            fname = spans[0]["font_name"]
            if "Bold" in fname:
                inner = f"<b>{inner}</b>"
            if "Italic" in fname:
                inner = f"<i>{inner}</i>"

    return (
        f'<div class="line {dir_class}" '
        f'style="left:{left:.2f}px; top:{y0:.2f}px; '
        f'width:{render_w:.2f}px; '
        f'height:{line_height:.2f}px; line-height:{line_height:.2f}px; '
        f'font-size:{font_size:.2f}px;">'
        f'{inner}'
        f'</div>'
    )


def render_page(page: dict, scale: float) -> str:
    """Render one page as HTML section."""
    pn     = page["page_number"]
    pw     = page["page_width"]
    ph     = page["page_height"]
    method = page["extraction_method"]
    dpi    = page["dpi"]

    ordered = sort_blocks(page["blocks"])
    line_divs = []
    for block in ordered:
        for line in block.get("lines", []):
            line_divs.append(render_line(line, pw))

    lines_html = "\n    ".join(line_divs)
    scaled_w = pw * scale
    scaled_h = ph * scale

    # data-pw / data-ph used by responsive JS
    return f"""
  <!-- ═══ Page {pn} ({method}, dpi={dpi}) ═══ -->
  <div class="page-label">Page {pn} &mdash; {method} (dpi {dpi})</div>
  <div class="page-wrapper" style="width:{scaled_w:.2f}px; height:{scaled_h:.2f}px;">
    <div class="page" data-pw="{pw:.2f}" data-ph="{ph:.2f}"
         style="width:{pw:.2f}px; height:{ph:.2f}px; transform:scale({scale});">
      {lines_html}
    </div>
  </div>
"""


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Render normalized layout JSONL as HTML.")
    ap.add_argument("--input", "-i", default=DEFAULT_INPUT)
    ap.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    ap.add_argument("--scale", "-s", type=float, default=DEFAULT_SCALE)
    ap.add_argument("--pages", "-p", default=None,
                    help="Comma-separated page numbers (default: all)")
    args = ap.parse_args()

    page_filter = None
    if args.pages:
        page_filter = {int(x.strip()) for x in args.pages.split(",")}

    inp = Path(args.input)
    if not inp.exists():
        print(f"Error: {inp} not found.", file=sys.stderr)
        sys.exit(1)

    pages = []
    with open(inp, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            p = json.loads(raw)
            if page_filter and p["page_number"] not in page_filter:
                continue
            pages.append(p)

    if not pages:
        print("No pages to render.", file=sys.stderr)
        sys.exit(1)

    pages.sort(key=lambda p: p["page_number"])

    drift_pct = int((FONT_DRIFT_FACTOR - 1) * 100)
    head = HTML_HEAD.format(font_family=DEFAULT_FONT, drift_pct=drift_pct)
    tail = HTML_TAIL.format(base_scale=args.scale)

    parts = [head]
    tb, tl = 0, 0
    for page in pages:
        nb = len(page["blocks"])
        nl = sum(len(b["lines"]) for b in page["blocks"])
        tb += nb
        tl += nl
        parts.append(render_page(page, args.scale))
    parts.append(tail)

    out = Path(args.output)
    out.write_text("".join(parts), encoding="utf-8")

    kb = out.stat().st_size / 1024
    print(f"Rendered {len(pages)} pages ({tb} blocks, {tl} lines)")
    print(f"Scale: {args.scale}x  |  Font drift compensation: +{drift_pct}%")
    print(f"Output: {out}  ({kb:.1f} KB)")


if __name__ == "__main__":
    main()
