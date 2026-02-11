#!/usr/bin/env python3
"""
Analyze raw PDF text structure for specific financial table pages.
Extracts span-level detail including text, bbox, font, size, 
and groups spans into visual rows by y-coordinate.
"""

import fitz  # PyMuPDF
from collections import defaultdict
import sys

PDF_PATH = "/home/ahmedsoliman/AI_projects/venv_arabic_rag/el-bankalahly .pdf"

# Pages of interest (0-indexed internally, user specifies 1-indexed)
PAGES_OF_INTEREST = {
    3: "Balance Sheet (scanned)",
    4: "Income Statement",
    7: "Equity Table",
    8: "Distributions",
    14: "Investments",
    16: "Deposits",
    18: "EPS",
}

Y_ROUND = 2  # round y0 to nearest 2pt for row grouping


def round_to(val, step):
    return round(val / step) * step


def analyze_page(page, page_num, label):
    """Analyze a single page and print detailed span information."""
    separator = "=" * 100
    print(f"\n{separator}")
    print(f"PAGE {page_num}: {label}")
    print(f"Page size: {page.rect.width:.1f} x {page.rect.height:.1f}")
    print(separator)

    data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES)

    blocks = data.get("blocks", [])
    total_blocks = len(blocks)
    total_lines = 0
    total_spans = 0
    all_spans = []  # collect all spans for row grouping

    # Count and collect
    for block in blocks:
        if block["type"] != 0:  # skip image blocks
            continue
        lines = block.get("lines", [])
        total_lines += len(lines)
        for line in lines:
            spans = line.get("spans", [])
            total_spans += len(spans)
            for span in spans:
                all_spans.append(span)

    print(f"\nSUMMARY: {total_blocks} blocks, {total_lines} lines, {total_spans} text spans")
    print(f"(Image/non-text blocks are excluded from line/span counts)\n")

    # ---- Print every span in document order ----
    print("-" * 100)
    print("ALL SPANS (document order):")
    print("-" * 100)
    print(f"{'#':>4}  {'x0':>7} {'y0':>7} {'x1':>7} {'y1':>7}  {'size':>5}  {'font':<30}  text")
    print(f"{'':>4}  {'':>7} {'':>7} {'':>7} {'':>7}  {'':>5}  {'':.<30}  ....")
    
    span_idx = 0
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                span_idx += 1
                bbox = span["bbox"]
                text = span["text"].replace("\n", "\\n")
                font = span["font"]
                size = span["size"]
                print(f"{span_idx:4d}  {bbox[0]:7.1f} {bbox[1]:7.1f} {bbox[2]:7.1f} {bbox[3]:7.1f}  {size:5.1f}  {font:<30}  |{text}|")

    # ---- Group spans by visual row (y0 rounded) ----
    print(f"\n{'-' * 100}")
    print(f"VISUAL ROWS (spans grouped by y0 rounded to {Y_ROUND}pt):")
    print(f"{'-' * 100}")

    row_map = defaultdict(list)
    for span in all_spans:
        y_key = round_to(span["bbox"][1], Y_ROUND)
        row_map[y_key].append(span)

    # Sort rows by y coordinate
    for y_key in sorted(row_map.keys()):
        spans_in_row = row_map[y_key]
        # Sort spans in row by x0
        spans_in_row.sort(key=lambda s: s["bbox"][0])
        
        print(f"\n  ROW y≈{y_key:.0f}  ({len(spans_in_row)} spans)")
        for s in spans_in_row:
            b = s["bbox"]
            text = s["text"].replace("\n", "\\n")
            print(f"    x0={b[0]:7.1f}  x1={b[2]:7.1f}  w={b[2]-b[0]:6.1f}  size={s['size']:5.1f}  |{text}|")

    # ---- Unique x0 positions (column boundary detection) ----
    print(f"\n{'-' * 100}")
    print("UNIQUE x0 POSITIONS (potential column boundaries):")
    print(f"{'-' * 100}")
    
    x0_values = sorted(set(round(s["bbox"][0], 1) for s in all_spans))
    print(f"  Found {len(x0_values)} unique x0 positions:")
    
    # Group nearby x0 values (within 3pt)
    x0_clusters = []
    if x0_values:
        current_cluster = [x0_values[0]]
        for x in x0_values[1:]:
            if x - current_cluster[-1] <= 3.0:
                current_cluster.append(x)
            else:
                x0_clusters.append(current_cluster)
                current_cluster = [x]
        x0_clusters.append(current_cluster)
    
    print(f"  Clustered into {len(x0_clusters)} column groups (within 3pt):")
    for i, cluster in enumerate(x0_clusters):
        avg_x = sum(cluster) / len(cluster)
        # Count how many spans fall in this cluster
        count = sum(1 for s in all_spans if round(s["bbox"][0], 1) in cluster)
        print(f"    Column {i+1}: x0 ≈ {avg_x:7.1f}  (range {min(cluster):.1f}-{max(cluster):.1f}, {count} spans)")

    # ---- Unique x1 positions (right edges) ----
    print(f"\n  UNIQUE x1 POSITIONS (right edges):")
    x1_values = sorted(set(round(s["bbox"][2], 1) for s in all_spans))
    x1_clusters = []
    if x1_values:
        current_cluster = [x1_values[0]]
        for x in x1_values[1:]:
            if x - current_cluster[-1] <= 3.0:
                current_cluster.append(x)
            else:
                x1_clusters.append(current_cluster)
                current_cluster = [x]
        x1_clusters.append(current_cluster)
    
    print(f"  Clustered into {len(x1_clusters)} right-edge groups:")
    for i, cluster in enumerate(x1_clusters):
        avg_x = sum(cluster) / len(cluster)
        count = sum(1 for s in all_spans if round(s["bbox"][2], 1) in cluster)
        print(f"    RightEdge {i+1}: x1 ≈ {avg_x:7.1f}  (range {min(cluster):.1f}-{max(cluster):.1f}, {count} spans)")

    return total_spans


def main():
    print(f"Opening PDF: {PDF_PATH}")
    doc = fitz.open(PDF_PATH)
    print(f"Total pages in document: {len(doc)}")
    
    grand_total_spans = 0
    
    for page_num in sorted(PAGES_OF_INTEREST.keys()):
        label = PAGES_OF_INTEREST[page_num]
        if page_num - 1 >= len(doc):
            print(f"\n*** Page {page_num} does not exist (doc has {len(doc)} pages) ***")
            continue
        page = doc[page_num - 1]  # 0-indexed
        count = analyze_page(page, page_num, label)
        grand_total_spans += count
    
    print(f"\n{'=' * 100}")
    print(f"GRAND TOTAL: {grand_total_spans} spans across {len(PAGES_OF_INTEREST)} pages analyzed")
    print(f"{'=' * 100}")
    
    doc.close()


if __name__ == "__main__":
    main()
