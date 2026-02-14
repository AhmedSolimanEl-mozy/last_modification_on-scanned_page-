#!/usr/bin/env python3
"""
normalize_layout.py — Data-driven layout normalization for layout JSONL.

Reads layout_output.jsonl and produces layout_normalized.jsonl with:

1. Dynamic page sizing   — page dims = max(content bbox) + margins
2. Coordinate clamping   — no negative positions, no overflow past page edge
3. Collision resolution  — overlapping lines nudged apart vertically
4. Font-size stabilization — consistent font-size estimation for OCR lines
5. Metadata enrichment   — adds _norm section with audit trail

Every correction is logged in the _norm.corrections list so changes are
fully transparent and reversible.

Usage:
    python normalize_layout.py
    python normalize_layout.py --input layout_output.jsonl --output layout_normalized.jsonl
    python normalize_layout.py --margin 15 --font-factor 0.85
"""

import argparse
import copy
import json
import sys
from pathlib import Path

# ─── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_INPUT   = "layout_output.jsonl"
DEFAULT_OUTPUT  = "layout_normalized.jsonl"
DEFAULT_MARGIN  = 10       # pt padding added beyond content bbox
FONT_SIZE_FACTOR = 0.85    # bbox_height × factor = estimated font-size
MIN_LINE_GAP    = 1.0      # minimum vertical gap (pt) between lines
OVERLAP_THRESH  = 2.0      # px overlap in both axes to count as collision


def collect_all_lines(page: dict) -> list:
    """Flatten all lines from all blocks, preserving block_id reference."""
    lines = []
    for b in page["blocks"]:
        for ln in b["lines"]:
            lines.append(ln)
    return lines


def compute_content_bbox(lines: list) -> tuple:
    """Return (min_x0, min_y0, max_x1, max_y1) across all lines."""
    if not lines:
        return (0, 0, 100, 100)
    xs0 = [ln["bbox"][0] for ln in lines]
    ys0 = [ln["bbox"][1] for ln in lines]
    xs1 = [ln["bbox"][2] for ln in lines]
    ys1 = [ln["bbox"][3] for ln in lines]
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def normalize_page(page: dict, margin: float, font_factor: float) -> dict:
    """Apply all normalization passes to one page.  Returns new page dict."""
    page = copy.deepcopy(page)
    corrections = []

    all_lines = collect_all_lines(page)
    if not all_lines:
        page["_norm"] = {"corrections": [], "version": 1}
        return page

    # ── Pass 1: Dynamic page sizing ─────────────────────────────────────
    cx0, cy0, cx1, cy1 = compute_content_bbox(all_lines)
    declared_w = page["page_width"]
    declared_h = page["page_height"]

    # Page must be at least as wide/tall as content + margin
    needed_w = cx1 + margin
    needed_h = cy1 + margin

    # Use the larger of declared vs needed — never shrink below declared
    new_w = max(declared_w, needed_w)
    new_h = max(declared_h, needed_h)

    if abs(new_w - declared_w) > 0.5 or abs(new_h - declared_h) > 0.5:
        corrections.append({
            "type": "page_resize",
            "old": [declared_w, declared_h],
            "new": [round(new_w, 2), round(new_h, 2)],
            "reason": f"content extends to ({cx1:.1f}, {cy1:.1f})"
        })
        page["page_width"] = round(new_w, 2)
        page["page_height"] = round(new_h, 2)

    pw = page["page_width"]
    ph = page["page_height"]

    # ── Pass 2: Coordinate clamping ─────────────────────────────────────
    for b in page["blocks"]:
        for ln in b["lines"]:
            x0, y0, x1, y1 = ln["bbox"]
            new_bbox = [x0, y0, x1, y1]
            changed = False

            # Clamp left edge
            if x0 < 0:
                new_bbox[0] = 0
                changed = True
            # Clamp top edge
            if y0 < 0:
                new_bbox[1] = 0
                changed = True
            # Clamp right edge — shift left if needed
            if x1 > pw:
                shift = x1 - pw + 1
                new_bbox[0] = max(0, new_bbox[0] - shift)
                new_bbox[2] = pw - 1
                changed = True
            # Clamp bottom edge
            if y1 > ph:
                shift = y1 - ph + 1
                new_bbox[1] = max(0, new_bbox[1] - shift)
                new_bbox[3] = ph - 1
                changed = True

            if changed:
                corrections.append({
                    "type": "clamp",
                    "line_id": ln.get("line_id"),
                    "old_bbox": [round(v, 2) for v in [x0, y0, x1, y1]],
                    "new_bbox": [round(v, 2) for v in new_bbox],
                })
                ln["bbox"] = [round(v, 2) for v in new_bbox]

            # Also clamp span bboxes
            for sp in ln.get("spans", []):
                sb = sp.get("bbox")
                if sb:
                    sp["bbox"] = [
                        max(0, sb[0]),
                        max(0, sb[1]),
                        min(pw, sb[2]),
                        min(ph, sb[3]),
                    ]

    # ── Pass 3: Font-size stabilization ─────────────────────────────────
    for b in page["blocks"]:
        for ln in b["lines"]:
            has_font = any(
                sp.get("font_size") is not None
                for sp in ln.get("spans", [])
            )
            if not has_font:
                # Estimate from bbox height
                _, y0, _, y1 = ln["bbox"]
                est_fs = max((y1 - y0) * font_factor, 4.0)
                for sp in ln.get("spans", []):
                    sp["font_size"] = round(est_fs, 2)
                    sp["_font_estimated"] = True

    # ── Pass 4: Collision detection & resolution ────────────────────────
    # Collect all lines sorted by y0, then resolve overlaps by nudging down
    flat_lines = []
    for b in page["blocks"]:
        for ln in b["lines"]:
            flat_lines.append(ln)

    # Sort by y0, then x0
    flat_lines.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))

    # Greedy overlap resolution: for each line, check against all previous
    for i in range(1, len(flat_lines)):
        ln = flat_lines[i]
        x0, y0, x1, y1 = ln["bbox"]
        h = y1 - y0

        for j in range(i - 1, max(i - 30, -1), -1):  # check last 30 lines
            prev = flat_lines[j]
            px0, py0, px1, py1 = prev["bbox"]

            # Check horizontal overlap
            if x0 >= px1 or x1 <= px0:
                continue
            # Check vertical overlap
            if y0 >= py1:
                continue

            # There's an overlap
            ov_x = min(x1, px1) - max(x0, px0)
            ov_y = py1 - y0

            if ov_x > OVERLAP_THRESH and ov_y > OVERLAP_THRESH:
                # Nudge current line down by the overlap + minimum gap
                nudge = ov_y + MIN_LINE_GAP
                corrections.append({
                    "type": "collision_nudge",
                    "line_id": ln.get("line_id"),
                    "nudge_y": round(nudge, 2),
                    "overlapped_with": prev.get("line_id"),
                })
                ln["bbox"][1] = round(py1 + MIN_LINE_GAP, 2)
                ln["bbox"][3] = round(ln["bbox"][1] + h, 2)
                # Update local vars for cascading checks
                y0 = ln["bbox"][1]
                y1 = ln["bbox"][3]

    # Also update block-level bboxes to envelope their lines
    for b in page["blocks"]:
        if not b["lines"]:
            continue
        bx0 = min(ln["bbox"][0] for ln in b["lines"])
        by0 = min(ln["bbox"][1] for ln in b["lines"])
        bx1 = max(ln["bbox"][2] for ln in b["lines"])
        by1 = max(ln["bbox"][3] for ln in b["lines"])
        b["bbox"] = [round(bx0, 2), round(by0, 2), round(bx1, 2), round(by1, 2)]

    # ── Store metadata ──────────────────────────────────────────────────
    page["_norm"] = {
        "version": 1,
        "corrections_count": len(corrections),
        "corrections": corrections,
    }

    return page


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Normalize layout JSONL.")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT)
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--font-factor", type=float, default=FONT_SIZE_FACTOR)
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Error: {inp} not found.", file=sys.stderr)
        sys.exit(1)

    pages = []
    with open(inp, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                pages.append(json.loads(raw))

    print(f"Loaded {len(pages)} pages from {inp}")

    total_corrections = 0
    normalized = []
    for page in pages:
        np = normalize_page(page, args.margin, args.font_factor)
        nc = np["_norm"]["corrections_count"]
        total_corrections += nc
        pn = np["page_number"]
        if nc > 0:
            print(f"  Page {pn:2d}: {nc} corrections applied")
            for c in np["_norm"]["corrections"]:
                print(f"    - {c['type']}: {c.get('reason', c.get('nudge_y', ''))}")
        normalized.append(np)

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for np in normalized:
            f.write(json.dumps(np, ensure_ascii=False) + "\n")

    size_kb = out.stat().st_size / 1024
    print(f"\nOutput: {out}  ({size_kb:.1f} KB)")
    print(f"Total corrections: {total_corrections}")


if __name__ == "__main__":
    main()
