#!/usr/bin/env python3
"""
table_render.py — HTML Rendering for Reconstructed Tables
==========================================================

Renders TableGrid objects from table_reconstruct.py as HTML with:

  • CSS Grid layout with fixed column widths
  • Absolute positioning within the page context
  • RTL text + LTR numeric island handling
  • QA highlighting: low-confidence cells get red borders
  • Column sum verification shown in footer
  • Anomaly flagging for suspicious numeric values

Two rendering modes:
  1. Embedded mode: renders table HTML fragments for insertion into
     the full-page layout HTML (from render_html.py).
  2. Standalone mode: full HTML page with all tables for review.

Usage:
    from table_render import render_table_html, render_tables_standalone
"""

from __future__ import annotations

import html as html_mod
import re
from typing import List, Optional

from table_reconstruct import TableGrid, TableCell, ColInfo, RowInfo

# ────────────────────────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────────────────────────
LOW_CONFIDENCE_THRESHOLD = 0.7      # cells below this get QA highlight
ANOMALY_THRESHOLD_RATIO  = 3.0      # cell value > N× column median → flag
QA_BORDER_COLOR          = "#ff4444"
QA_BG_COLOR              = "#fff0f0"
HEADER_BG_COLOR          = "#e8eef3"
NUMERIC_RE = re.compile(r'^[\d٠-٩,،.\(\)\-−–\s%]+$')
_ARABIC_INDIC = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')


# ────────────────────────────────────────────────────────────────────
#  QA: Column sum + anomaly detection
# ────────────────────────────────────────────────────────────────────
def parse_numeric_value(text: str) -> Optional[float]:
    """Try to parse a numeric cell value (handles parens for negatives,
    commas, Arabic-Indic digits)."""
    if not text.strip():
        return None
    t = text.strip()
    t = t.translate(_ARABIC_INDIC)
    t = t.replace('،', ',')

    # Handle parentheses as negative
    negative = False
    if t.startswith('(') and t.endswith(')'):
        negative = True
        t = t[1:-1]
    elif t.startswith('-') or t.startswith('−') or t.startswith('–'):
        negative = True
        t = t[1:]

    t = t.replace(',', '').replace('%', '').strip()

    try:
        val = float(t)
        return -val if negative else val
    except ValueError:
        return None


def compute_column_sums(grid: TableGrid) -> dict:
    """Compute column sums for numeric columns.
    Returns {col_id: {'sum': float, 'count': int, 'values': list}}"""
    col_data = {}
    for cell in grid.cells:
        if cell.is_header:
            continue
        val = parse_numeric_value(cell.text)
        if val is not None:
            if cell.col not in col_data:
                col_data[cell.col] = {'sum': 0.0, 'count': 0,
                                       'values': [], 'cells': []}
            col_data[cell.col]['sum'] += val
            col_data[cell.col]['count'] += 1
            col_data[cell.col]['values'].append(val)
            col_data[cell.col]['cells'].append(cell)
    return col_data


def detect_anomalies(grid: TableGrid) -> List[dict]:
    """Detect numeric anomalies: values that are outliers in their column.

    IMPORTANT: Header rows (including year-label rows) are EXCLUDED from
    both column statistics and anomaly flagging. This prevents 4-digit
    year labels (e.g. 2023, 2024) from being flagged as outliers against
    financial data values.
    """
    import statistics
    anomalies = []
    col_data = compute_column_sums(grid)  # already skips headers

    for col_id, data in col_data.items():
        if data['count'] < 3:
            continue
        abs_values = [abs(v) for v in data['values'] if v != 0]
        if not abs_values:
            continue
        median = statistics.median(abs_values)
        if median == 0:
            continue
        for i, (val, cell) in enumerate(zip(data['values'], data['cells'])):
            # Header rows are already excluded by compute_column_sums,
            # but double-check for safety.
            if cell.is_header:
                continue
            if abs(val) > median * ANOMALY_THRESHOLD_RATIO and abs(val) != 0:
                anomalies.append({
                    'row': cell.row, 'col': cell.col,
                    'text': cell.text,
                    'value': val,
                    'median': median,
                    'ratio': abs(val) / median,
                    'reason': f'Value {val:.2f} is {abs(val)/median:.1f}× column median {median:.2f}',
                })
    return anomalies


def identify_low_confidence_cells(grid: TableGrid) -> List[dict]:
    """Find cells with confidence below threshold."""
    flagged = []
    for cell in grid.cells:
        if cell.text.strip() and cell.confidence < LOW_CONFIDENCE_THRESHOLD:
            flagged.append({
                'row': cell.row, 'col': cell.col,
                'text': cell.text,
                'confidence': cell.confidence,
            })
    return flagged


# ────────────────────────────────────────────────────────────────────
#  Embedded table renderer (for page layout integration)
# ────────────────────────────────────────────────────────────────────
def render_table_embedded(
    grid: TableGrid,
    table_id: int = 0,
    show_qa: bool = True,
) -> str:
    """Render a table as an absolutely-positioned HTML fragment
    for insertion into the page layout (from render_html.py).

    Uses CSS Grid with explicit column widths derived from ColInfo.
    Positioned at the table's bbox within the page coordinate system.
    """
    if not grid.cells:
        return ""

    x0, y0, x1, y1 = grid.bbox
    table_w = x1 - x0
    table_h = y1 - y0

    # Compute QA info
    low_conf = {(c['row'], c['col']) for c in identify_low_confidence_cells(grid)} if show_qa else set()
    anomalies_set = set()
    if show_qa:
        for a in detect_anomalies(grid):
            anomalies_set.add((a['row'], a['col']))

    # Column widths from ColInfo boundaries
    col_widths = []
    for col in grid.columns:
        w = col.x1 - col.x0
        col_widths.append(f"{w:.1f}px")
    grid_cols = " ".join(col_widths) if col_widths else "1fr"

    # Build cell HTML
    cell_htmls = []
    for cell in grid.cells:
        if cell.col >= len(grid.columns) or cell.row >= len(grid.rows):
            continue

        text = html_mod.escape(cell.text) if cell.text else "&nbsp;"
        classes = []
        styles = []

        # Direction
        if cell.is_numeric:
            classes.append("num")
            styles.append("direction:ltr; text-align:right;")
        elif cell.direction == "rtl":
            classes.append("rtl")
            styles.append("direction:rtl; text-align:right;")
        else:
            styles.append("direction:ltr; text-align:left;")

        # Header styling
        if cell.is_header:
            classes.append("hdr")
            styles.append(f"background:{HEADER_BG_COLOR}; font-weight:bold;")

        # QA: low confidence
        is_qa = (cell.row, cell.col) in low_conf
        is_anomaly = (cell.row, cell.col) in anomalies_set
        if is_qa:
            classes.append("low-conf")
            styles.append(f"border:2px solid {QA_BORDER_COLOR};")
        if is_anomaly:
            classes.append("anomaly")
            styles.append(f"background:{QA_BG_COLOR};")

        # Row height from RowInfo
        row_h = grid.rows[cell.row].y1 - grid.rows[cell.row].y0 if cell.row < len(grid.rows) else 14
        styles.append(f"height:{row_h:.1f}px; line-height:{row_h:.1f}px;")

        # Font size estimation
        font_size = max(row_h * 0.75, 6)
        styles.append(f"font-size:{font_size:.1f}px;")

        cls_str = f' class="{" ".join(classes)}"' if classes else ""
        sty_str = f' style="{" ".join(styles)}"'

        # Grid position (CSS grid is 1-indexed)
        grid_row = cell.row + 1
        grid_col = cell.col + 1

        cell_htmls.append(
            f'<div{cls_str}{sty_str} '
            f'style="grid-row:{grid_row}; grid-col:{grid_col}; '
            f'{" ".join(styles)}" '
            f'title="r{cell.row}c{cell.col} conf={cell.confidence:.2f}">'
            f'{text}</div>'
        )

    cells_html = "\n      ".join(cell_htmls)

    # Build the table container
    return f"""
    <div class="table-block" id="table-{table_id}"
         style="position:absolute; left:{x0:.1f}px; top:{y0:.1f}px;
                width:{table_w:.1f}px; min-height:{table_h:.1f}px;
                display:grid;
                grid-template-columns:{grid_cols};
                grid-template-rows:repeat({grid.num_rows}, auto);
                gap:0; border:1px solid #ccc;
                font-family:Arial, 'Noto Sans Arabic', sans-serif;
                box-sizing:border-box;
                overflow:visible;">
      {cells_html}
    </div>"""


# ────────────────────────────────────────────────────────────────────
#  Standalone HTML renderer (for review/QA)
# ────────────────────────────────────────────────────────────────────
def render_tables_standalone(
    grids: List[TableGrid],
    output_path: str = "tables_review.html",
    show_qa: bool = True,
) -> str:
    """Render all tables as a standalone HTML page for QA review.

    Includes:
      • CSS Grid tables with fixed column widths
      • Column sums displayed below each table
      • Low-confidence cells highlighted in red
      • Anomaly cells highlighted in pink
      • Confidence scores on hover
    """
    parts = [_STANDALONE_HEAD]

    for ti, grid in enumerate(grids):
        parts.append(f'<h2>Table {ti+1} — Page {grid.page_number} '
                     f'({grid.num_rows}×{grid.num_cols}, '
                     f'conf={grid.confidence:.2f})</h2>')

        # QA info
        low_conf_cells = identify_low_confidence_cells(grid)
        anomalies = detect_anomalies(grid)
        col_sums = compute_column_sums(grid)
        low_conf_set = {(c['row'], c['col']) for c in low_conf_cells}
        anomaly_set = {(a['row'], a['col']) for a in anomalies}

        # Column widths
        col_widths = []
        for col in grid.columns:
            w = max(col.x1 - col.x0, 40)
            col_widths.append(f"{w:.0f}px")
        grid_cols = " ".join(col_widths) if col_widths else "repeat(auto-fill, 1fr)"

        parts.append(f'<div class="table-grid" style="'
                     f'grid-template-columns:{grid_cols}; '
                     f'grid-template-rows:repeat({grid.num_rows}, auto);">')

        for cell in grid.cells:
            text = html_mod.escape(cell.text) if cell.text else "&nbsp;"
            classes = ["cell"]
            extra_style = ""

            if cell.is_header:
                classes.append("hdr")
            if cell.is_numeric:
                classes.append("num")
            if cell.direction == "rtl":
                classes.append("rtl")
            if (cell.row, cell.col) in low_conf_set:
                classes.append("low-conf")
            if (cell.row, cell.col) in anomaly_set:
                classes.append("anomaly")

            cls = " ".join(classes)
            parts.append(
                f'  <div class="{cls}" '
                f'style="grid-row:{cell.row+1}; grid-column:{cell.col+1};{extra_style}" '
                f'title="[{cell.row},{cell.col}] conf={cell.confidence:.2f} '
                f'{"NUM" if cell.is_numeric else "TXT"}">'
                f'{text}</div>'
            )

        parts.append('</div>')

        # Column sums footer
        if col_sums:
            parts.append('<div class="qa-section">')
            parts.append('<h3>📊 Column Sums</h3><table class="qa-table">')
            parts.append('<tr><th>Column</th><th>Sum</th><th>Count</th></tr>')
            for ci in sorted(col_sums.keys()):
                d = col_sums[ci]
                parts.append(f'<tr><td>Col {ci}</td>'
                             f'<td>{d["sum"]:,.2f}</td>'
                             f'<td>{d["count"]}</td></tr>')
            parts.append('</table>')

            # Low confidence warnings
            if low_conf_cells:
                parts.append(f'<h3>⚠️ Low Confidence ({len(low_conf_cells)} cells)</h3>')
                parts.append('<ul>')
                for lc in low_conf_cells:
                    parts.append(f'<li>Row {lc["row"]}, Col {lc["col"]}: '
                                 f'"{lc["text"]}" (conf={lc["confidence"]:.2f})</li>')
                parts.append('</ul>')

            # Anomalies
            if anomalies:
                parts.append(f'<h3>🔴 Numeric Anomalies ({len(anomalies)})</h3>')
                parts.append('<ul>')
                for a in anomalies:
                    parts.append(f'<li>Row {a["row"]}, Col {a["col"]}: '
                                 f'"{a["text"]}" — {a["reason"]}</li>')
                parts.append('</ul>')

            parts.append('</div>')

        parts.append('<hr>')

    parts.append(_STANDALONE_TAIL)

    html_str = "\n".join(parts)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_str)
    return html_str


# ────────────────────────────────────────────────────────────────────
#  Standalone HTML template
# ────────────────────────────────────────────────────────────────────
_STANDALONE_HEAD = """<!DOCTYPE html>
<html lang="ar" dir="ltr">
<head>
<meta charset="utf-8">
<title>Table Extraction — QA Review</title>
<style>
body {
  font-family: Arial, 'Noto Sans Arabic', sans-serif;
  background: #f5f5f5;
  padding: 20px;
  direction: ltr;
}
h2 { color: #333; margin-top: 30px; }
.table-grid {
  display: grid;
  gap: 0;
  border: 2px solid #333;
  background: #fff;
  margin: 10px 0;
  max-width: 100%;
  overflow-x: auto;
}
.cell {
  border: 1px solid #ddd;
  padding: 4px 6px;
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-height: 20px;
}
.cell.hdr {
  background: #e8eef3;
  font-weight: bold;
  border-bottom: 2px solid #999;
}
.cell.num {
  direction: ltr;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.cell.rtl {
  direction: rtl;
  text-align: right;
  unicode-bidi: plaintext;
}
.cell.low-conf {
  border: 2px solid #ff4444 !important;
  background: #fff8f8;
}
.cell.anomaly {
  background: #fff0f0 !important;
}
.qa-section {
  background: #fafafa;
  border: 1px solid #ddd;
  padding: 10px 15px;
  margin: 10px 0;
  border-radius: 4px;
}
.qa-table {
  border-collapse: collapse;
  margin: 5px 0;
}
.qa-table th, .qa-table td {
  border: 1px solid #ccc;
  padding: 4px 8px;
  text-align: right;
}
.qa-table th { background: #eee; }
hr { margin: 30px 0; border: 1px solid #ddd; }
</style>
</head>
<body>
<h1>📋 Table Extraction — QA Review</h1>
<p>Cells with <span style="border:2px solid #ff4444; padding:2px;">red borders</span>
have low OCR confidence.
Cells with <span style="background:#fff0f0; padding:2px;">pink background</span>
are numeric anomalies.</p>
"""

_STANDALONE_TAIL = """
</body>
</html>
"""
