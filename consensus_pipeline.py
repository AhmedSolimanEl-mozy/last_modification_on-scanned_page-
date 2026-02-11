#!/usr/bin/env python3
"""
consensus_pipeline.py — Multi-Version OCR Consensus via Local LLM
=================================================================
Reads multiple OCR text files of the same Arabic financial document,
aligns them page-by-page, then uses a local LLM (via Ollama) to
reconcile differences block-by-block and produce a clean output.

Architecture
────────────
  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
  │ extracted_   │ │ extracted_  │ │ extracted_  │  ...  N files
  │ lines.txt   │ │ v9.txt      │ │ layout.txt  │
  └─────┬───────┘ └─────┬───────┘ └─────┬───────┘
        │               │               │
        └───────┬───────┘───────┬───────┘
                │               │
        ┌───────▼───────────────▼───────┐
        │   Page Segmentation           │  ← regex per file format
        │   → dict[page_num] = text     │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │   Block Chunking              │  ← split pages into ~30-line
        │   (within token budget)       │     blocks with overlap
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │   Cross-Version Alignment     │  ← SequenceMatcher fuzzy
        │   (sliding window similarity) │     match for block pairing
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │   LLM Consensus Prompt        │  ← Ollama qwen2.5:7b
        │   (per aligned block)         │     structured Arabic prompt
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │   Validation + Assembly       │  ← extract markdown tables,
        │                               │     verify structure
        └───────────────┬───────────────┘
                        │
                ┌───────▼───────┐
                │  consensus_   │
                │  output.txt   │
                └───────────────┘

Model Selection Notes
─────────────────────
  qwen2.5:7b (7.6B params) is chosen over the alternatives because:
  - Best Arabic language coverage among available models (trained on
    multilingual data with explicit CJK + Arabic representation)
  - 128K native context window (vs 32K for mistral, 128K for llama3.2)
  - Strong instruction following for structured output tasks
  - 4.7 GB fits in 16 GB RAM with room for context

  mistral:latest (7.2B) is a viable alternative but weaker on Arabic.
  llama3.2:1b/3b are too small for nuanced Arabic financial reconciliation.

Usage
─────
    python consensus_pipeline.py                     # defaults
    python consensus_pipeline.py --model qwen2.5:7b  # explicit model
    python consensus_pipeline.py --dry-run            # show blocks, no LLM
"""

import os
import re
import sys
import json
import time
import hashlib
import argparse
import textwrap
from pathlib import Path
from difflib import SequenceMatcher
from typing import Optional

# Force stdout unbuffered so nohup captures progress in real-time
sys.stdout.reconfigure(line_buffering=True)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request
    import urllib.error

# ====================================================================
#  Configuration
# ====================================================================
OLLAMA_URL     = "http://localhost:11434"
DEFAULT_MODEL  = "qwen2.5:7b"

# Input OCR files — ordered by quality (best first → LLM sees best version first)
OCR_FILES = [
    ("V1_v9",         "extracted_v9.txt"),
    ("V2_structured", "extracted_structured.txt"),
    ("V3_lines",      "extracted_lines.txt"),
    ("V4_layout",     "extracted_layout.txt"),
    ("V5_position",   "txt-with-position.txt"),
]

OUTPUT_FILE    = "consensus_output.txt"
CACHE_DIR      = ".consensus_cache"

# Block chunking
BLOCK_LINES    = 30     # target lines per block
BLOCK_OVERLAP  = 5      # overlap lines between consecutive blocks
MAX_VERSIONS   = 3      # max OCR versions sent per prompt (token budget)

# LLM parameters
LLM_TEMPERATURE = 0.1   # low temperature → deterministic reconciliation
LLM_TOP_P       = 0.9
LLM_NUM_CTX     = 8192  # context window to request from Ollama


# ====================================================================
#  Page delimiter patterns per file format
# ====================================================================
PAGE_PATTERNS = {
    "extracted_v9.txt":         re.compile(r'^\s*Page\s+(\d+)\s*$'),
    "extracted_structured.txt": re.compile(r'صفح(?:ه|ة)\s+(\d+)\s*\|\s*PAGE\s+(\d+)'),
    "extracted_lines.txt":      re.compile(r'صفح(?:ه|ة)\s+(\d+)\s*\|\s*PAGE\s+(\d+)'),
    "extracted_layout.txt":     re.compile(r'صفح(?:ه|ة)\s+(\d+)\s*\|\s*PAGE\s+(\d+)'),
    "extracted_units.txt":      re.compile(r'^=====\s*Page\s+(\d+)\s*====='),
    "txt-with-position.txt":    re.compile(r'^---\s*Page\s+(\d+)\s*---'),
    "moreenhance.txt":          re.compile(r'^\s*PAGE\s+(\d+)\s*$'),
}


def _detect_page_pattern(filename: str, lines: list[str]) -> re.Pattern:
    """Auto-detect the page delimiter pattern for a file."""
    basename = os.path.basename(filename)
    if basename in PAGE_PATTERNS:
        return PAGE_PATTERNS[basename]

    # Fallback: try all patterns and pick the one with most matches
    best_pat, best_count = None, 0
    for pat in PAGE_PATTERNS.values():
        count = sum(1 for line in lines if pat.search(line))
        if count > best_count:
            best_count = count
            best_pat = pat

    if best_pat and best_count >= 2:
        return best_pat

    # Last resort: generic PAGE N pattern
    return re.compile(r'PAGE\s+(\d+)|Page\s+(\d+)|صفح\w?\s+(\d+)')


def _extract_page_num(match: re.Match) -> int:
    """Extract the page number from a regex match (handles multiple groups)."""
    for g in match.groups():
        if g and g.isdigit():
            return int(g)
    return -1


# ====================================================================
#  File segmentation → dict[page_num] = text
# ====================================================================
def segment_file_by_page(filepath: str) -> dict[int, str]:
    """Read an OCR output file and split it into page-keyed sections."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pattern = _detect_page_pattern(filepath, lines)
    pages: dict[int, list[str]] = {}
    current_page = 0  # header / preamble

    for line in lines:
        m = pattern.search(line)
        if m:
            pn = _extract_page_num(m)
            if pn > 0:
                current_page = pn
                pages.setdefault(current_page, [])
                continue
        if current_page > 0:
            pages.setdefault(current_page, []).append(line)

    # Join and clean
    result = {}
    for pn, page_lines in pages.items():
        text = ''.join(page_lines).strip()
        # Remove decoration lines (═══, ───, ---)
        text = re.sub(r'^[═─\-─━]{3,}\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        if text:
            result[pn] = text

    return result


# ====================================================================
#  Block chunking  (split page text into prompt-sized blocks)
# ====================================================================
def chunk_page_into_blocks(text: str,
                           block_lines: int = BLOCK_LINES,
                           overlap: int = BLOCK_OVERLAP) -> list[str]:
    """Split page text into overlapping blocks of ~block_lines lines.

    Returns list of text blocks.  Short pages return a single block.
    """
    lines = text.split('\n')
    if len(lines) <= block_lines:
        return [text]

    blocks = []
    start = 0
    while start < len(lines):
        end = min(start + block_lines, len(lines))
        block = '\n'.join(lines[start:end])
        blocks.append(block)
        if end >= len(lines):
            break
        start = end - overlap   # overlap for continuity

    return blocks


# ====================================================================
#  Cross-version block alignment  (fuzzy matching)
# ====================================================================
def _text_similarity(a: str, b: str) -> float:
    """Quick similarity ratio between two text blocks."""
    if not a or not b:
        return 0.0
    # Use a sample for speed on large blocks
    sa = a[:500]
    sb = b[:500]
    return SequenceMatcher(None, sa, sb).ratio()


def align_blocks_across_versions(
    version_blocks: dict[str, list[str]]
) -> list[dict[str, str]]:
    """Align blocks across OCR versions using sequential + fuzzy matching.

    For files with the same number of blocks, align 1:1.
    For mismatched counts, use sliding-window similarity to find best match.

    Returns list of dicts: {version_name: block_text, ...}
    """
    version_names = list(version_blocks.keys())
    if not version_names:
        return []

    # Use the version with the most blocks as the reference
    ref_name = max(version_names, key=lambda v: len(version_blocks[v]))
    ref_blocks = version_blocks[ref_name]

    aligned = []
    for i, ref_block in enumerate(ref_blocks):
        group = {ref_name: ref_block}

        for vname in version_names:
            if vname == ref_name:
                continue
            other_blocks = version_blocks[vname]
            if not other_blocks:
                continue

            # If same count, align by index
            if len(other_blocks) == len(ref_blocks):
                group[vname] = other_blocks[i]
            else:
                # Sliding window: search nearby indices for best match
                search_start = max(0, i - 2)
                search_end = min(len(other_blocks), i + 3)
                best_idx, best_sim = i, 0.0

                for j in range(search_start, search_end):
                    sim = _text_similarity(ref_block, other_blocks[j])
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = j

                if best_idx < len(other_blocks):
                    group[vname] = other_blocks[best_idx]

        aligned.append(group)

    return aligned


# ====================================================================
#  LLM interaction  (Ollama REST API)
# ====================================================================
def _ollama_generate(prompt: str, model: str = DEFAULT_MODEL,
                     system: str = "") -> Optional[str]:
    """Call Ollama's /api/generate endpoint with streaming for progress.

    Uses streaming to show a token counter during CPU inference, then
    assembles the full response text.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "num_ctx": LLM_NUM_CTX,
        }
    }

    url = f"{OLLAMA_URL}/api/generate"

    if HAS_REQUESTS:
        try:
            resp = requests.post(url, json=payload, timeout=600, stream=True)
            resp.raise_for_status()
            full_response = []
            token_count = 0
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        full_response.append(token)
                        token_count += 1
                        if token_count % 20 == 0:
                            print(f'{token_count}', end=' ', flush=True)
                    if chunk.get("done", False):
                        break
            return ''.join(full_response)
        except requests.exceptions.ConnectionError:
            print("\n  [ERROR] Cannot connect to Ollama. Is it running?")
            print(f"         Start it with: ollama serve")
            sys.exit(1)
        except Exception as e:
            print(f"\n  [ERROR] Ollama request failed: {e}")
            return None
    else:
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"})
            full_response = []
            token_count = 0
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw_line in resp:
                    line = raw_line.decode('utf-8').strip()
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            full_response.append(token)
                            token_count += 1
                            if token_count % 20 == 0:
                                print(f'{token_count}', end=' ', flush=True)
                        if chunk.get("done", False):
                            break
            return ''.join(full_response)
        except urllib.error.URLError:
            print("\n  [ERROR] Cannot connect to Ollama. Is it running?")
            sys.exit(1)
        except Exception as e:
            print(f"\n  [ERROR] Ollama request failed: {e}")
            return None


# ====================================================================
#  Prompt Engineering
# ====================================================================
SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Arabic financial document specialist. Your task is to
    reconcile multiple OCR outputs of the same Arabic financial statement.

    CRITICAL RULES:
    1. Arabic text is RIGHT-TO-LEFT. In tables, labels are on the RIGHT
       and numeric values are on the LEFT.
    2. Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) and Western digits (0123456789)
       must BOTH be preserved exactly as they appear in the best version.
    3. Parenthesised numbers like (٧٣ ٥١٩) indicate NEGATIVE values.
       Preserve the parentheses exactly.
    4. Financial numbers often have spaces between digit groups:
       ٨٤٩ ٥٦٩  means  849,569.  Preserve the original spacing.
    5. Common OCR confusions to fix:
       - ٣ (3) vs ٩ (9) — use context (does this row's total make sense?)
       - ه vs ة (ta marbuta) — use correct Arabic grammar
       - ا vs أ vs إ (alef variants) — use standard form
       - ل vs لل (lam) — check if word exists in Arabic
    6. If versions disagree on a NUMBER, prefer the version where the
       number is consistent with row totals and column sums.
    7. If versions disagree on TEXT, prefer the version with correct
       Arabic spelling and complete words (no truncation).
    8. Output ONLY the reconciled text. No explanations, no commentary.
    9. For table data, output as a Markdown table with | delimiters.
    10. Preserve the original document structure (headers, sections, notes).
""")


def build_consensus_prompt(aligned_block: dict[str, str],
                           page_num: int,
                           block_idx: int,
                           total_blocks: int) -> str:
    """Build the LLM prompt for one aligned block."""

    version_texts = []
    for i, (vname, text) in enumerate(aligned_block.items(), 1):
        # Truncate very long blocks to stay within token budget
        truncated = text[:3000] if len(text) > 3000 else text
        version_texts.append(f"=== VERSION {i} ({vname}) ===\n{truncated}")

    versions_joined = "\n\n".join(version_texts)

    prompt = textwrap.dedent(f"""\
        CONTEXT: Page {page_num} of an Arabic financial statement
        (El Ahly Bank / البنك الاهلي المصري), block {block_idx}/{total_blocks}.

        Below are {len(aligned_block)} OCR versions of the SAME section.
        Each version may have different errors. Your job is to produce
        the single CORRECT version by cross-referencing all inputs.

        {versions_joined}

        TASK:
        1. Compare all versions line by line.
        2. Fix OCR typos by cross-referencing (if V1 says ٨٤٩ and V2 says ٨٤۹,
           pick the one consistent with column totals).
        3. For table rows: output as a Markdown table. Ensure the Arabic label
           is in the rightmost column (RTL reading order) and numbers in
           subsequent columns to the left.
        4. For paragraph text: output the cleanest, most complete version.
           Merge partial sentences from different versions if needed.
        5. Preserve Arabic-Indic numerals (٠-٩) exactly — do NOT convert
           to Western digits.
        6. Ensure BiDi correctness: Arabic text reads right-to-left,
           numbers within Arabic text read left-to-right.
        7. Output ONLY the reconciled text. No explanations.

        RECONCILED OUTPUT:
    """)

    return prompt


# ====================================================================
#  Response caching  (avoid re-running expensive LLM calls)
# ====================================================================
def _cache_key(prompt: str, model: str) -> str:
    h = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()[:16]
    return h


def _cache_get(key: str) -> Optional[str]:
    path = os.path.join(CACHE_DIR, f"{key}.txt")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return None


def _cache_set(key: str, value: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(value)


# ====================================================================
#  Output validation
# ====================================================================
def validate_block_output(text: str) -> dict:
    """Basic validation of LLM output for an Arabic financial block."""
    issues = []

    # Check for Arabic content
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    if arabic_chars < 5 and len(text) > 50:
        issues.append("LOW_ARABIC: very few Arabic characters in output")

    # Check for table formatting if pipe chars present in input
    if '|' in text:
        table_lines = [l for l in text.split('\n') if '|' in l]
        if table_lines:
            col_counts = [l.count('|') for l in table_lines]
            if len(set(col_counts)) > 2:
                issues.append("TABLE_COLS: inconsistent column count across rows")

    # Check for LLM commentary leakage
    commentary_markers = ['note:', 'explanation:', 'i have', 'here is',
                          'the reconciled', 'analysis:', 'comparing']
    lower = text.lower()
    for marker in commentary_markers:
        if marker in lower[:200]:
            issues.append(f"COMMENTARY: LLM commentary detected ('{marker}')")
            break

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "arabic_chars": arabic_chars,
        "line_count": text.count('\n') + 1,
    }


def _strip_commentary(text: str) -> str:
    """Remove common LLM commentary prefixes from the output."""
    # Remove lines that start with common commentary patterns
    lines = text.split('\n')
    cleaned = []
    skip = True
    for line in lines:
        lower = line.strip().lower()
        if skip and any(lower.startswith(p) for p in
                        ['here', 'note', 'the reconciled', 'below',
                         'i have', 'based on', 'after comparing',
                         'reconciled output', '---']):
            continue
        if line.strip():
            skip = False
        cleaned.append(line)
    return '\n'.join(cleaned)


# ====================================================================
#  Main pipeline
# ====================================================================
def run_pipeline(model: str = DEFAULT_MODEL,
                 dry_run: bool = False,
                 max_pages: int = 0,
                 pages: Optional[list[int]] = None):
    """Run the full consensus pipeline."""

    base_dir = os.path.dirname(os.path.abspath(__file__))

    print('=' * 62)
    print('  MULTI-VERSION OCR CONSENSUS PIPELINE')
    print('  Model: ' + model)
    print('  Mode:  ' + ('DRY RUN (no LLM calls)' if dry_run else 'LIVE'))
    print('=' * 62)

    # ── 1. Load and segment all OCR files ────────────────────────────
    print('\n  Loading OCR versions ...')
    all_versions: dict[str, dict[int, str]] = {}

    for label, fname in OCR_FILES:
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            print(f'    ✗ {label:20s} — file not found: {fname}')
            continue
        page_data = segment_file_by_page(fpath)
        if page_data:
            all_versions[label] = page_data
            page_nums = sorted(page_data.keys())
            print(f'    ✓ {label:20s} — {len(page_data)} pages '
                  f'({page_nums[0]}–{page_nums[-1]})')
        else:
            print(f'    ✗ {label:20s} — no pages found in {fname}')

    if len(all_versions) < 2:
        print('\n  [ERROR] Need at least 2 OCR versions. Aborting.')
        sys.exit(1)

    # ── 2. Determine page range ──────────────────────────────────────
    all_page_nums = set()
    for v in all_versions.values():
        all_page_nums.update(v.keys())
    all_page_nums = sorted(all_page_nums)

    if pages:
        all_page_nums = [p for p in all_page_nums if p in pages]
    elif max_pages > 0:
        all_page_nums = all_page_nums[:max_pages]

    print(f'\n  Pages to process: {all_page_nums}')
    print(f'  Versions loaded:  {len(all_versions)}')

    # ── 3. Pre-warm the model ────────────────────────────────────────
    if not dry_run:
        print(f'\n  Pre-warming {model} ...')
        warmup = _ollama_generate("Say OK.", model=model)
        if warmup is None:
            print('  [ERROR] Model did not respond. Check ollama serve.')
            sys.exit(1)
        print(f'  → Model ready.')

    # ── 4. Process page by page, block by block ──────────────────────
    output_parts: dict[int, str] = {}
    total_blocks_processed = 0
    total_cached = 0
    start_time = time.time()
    # Open output file NOW so blocks are written incrementally
    output_path = os.path.join(base_dir, OUTPUT_FILE)
    live_f = open(output_path, 'w', encoding='utf-8')
    live_f.write('=' * 62 + '\n')
    live_f.write('  CONSENSUS OUTPUT — Multi-Version OCR Reconciliation\n')
    live_f.write(f'  Model: {model}\n')
    live_f.write(f'  Versions: {len(all_versions)}\n')
    live_f.write(f'  Pages queued: {len(all_page_nums)}\n')
    live_f.write(f'  Started: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    live_f.write('=' * 62 + '\n\n')
    live_f.flush()
    for page_num in all_page_nums:
        print(f'\n  ──── Page {page_num} ────')

        # Write page header to live output immediately
        live_f.write(f'{"─" * 62}\n')
        live_f.write(f'  Page {page_num}\n')
        live_f.write(f'{"─" * 62}\n\n')
        live_f.flush()

        # Collect this page from each version
        page_versions: dict[str, str] = {}
        for vname, vpages in all_versions.items():
            if page_num in vpages:
                page_versions[vname] = vpages[page_num]

        if not page_versions:
            print(f'    (no content in any version)')
            continue

        # Select top N versions by content length (longer = more content)
        if len(page_versions) > MAX_VERSIONS:
            sorted_versions = sorted(page_versions.items(),
                                     key=lambda kv: len(kv[1]), reverse=True)
            page_versions = dict(sorted_versions[:MAX_VERSIONS])
            used = [v[0] for v in sorted_versions[:MAX_VERSIONS]]
            print(f'    Using top {MAX_VERSIONS} versions: {used}')

        # Chunk each version into blocks
        version_blocks: dict[str, list[str]] = {}
        for vname, text in page_versions.items():
            blocks = chunk_page_into_blocks(text)
            version_blocks[vname] = blocks

        block_counts = {v: len(b) for v, b in version_blocks.items()}
        print(f'    Block counts: {block_counts}')

        # Align blocks across versions
        aligned = align_blocks_across_versions(version_blocks)
        print(f'    Aligned blocks: {len(aligned)}')

        page_output_parts = []

        for bi, aligned_block in enumerate(aligned, 1):
            prompt = build_consensus_prompt(
                aligned_block, page_num, bi, len(aligned))

            if dry_run:
                # In dry run, just show the block info
                for vname, text in aligned_block.items():
                    preview = text[:80].replace('\n', ' ')
                    print(f'      Block {bi} [{vname}]: {preview}...')
                block_text = (
                    f"[DRY RUN — Block {bi}: "
                    f"{len(aligned_block)} versions, "
                    f"~{sum(len(t) for t in aligned_block.values())} chars]")
                page_output_parts.append(block_text)
                live_f.write(block_text + '\n\n')
                live_f.flush()
                total_blocks_processed += 1
                continue

            # Check cache
            ckey = _cache_key(prompt, model)
            cached = _cache_get(ckey)
            if cached:
                page_output_parts.append(cached)
                live_f.write(cached + '\n\n')
                live_f.flush()
                total_blocks_processed += 1
                total_cached += 1
                print(f'      Block {bi}/{len(aligned)}: cached ✓')
                continue

            # Call LLM
            print(f'      Block {bi}/{len(aligned)}: calling {model} ...',
                  end=' ', flush=True)
            t0 = time.time()

            response = _ollama_generate(prompt, model=model,
                                        system=SYSTEM_PROMPT)

            elapsed = time.time() - t0

            if response:
                # Clean up
                response = _strip_commentary(response)
                validation = validate_block_output(response)

                if validation['valid']:
                    print(f'{elapsed:.1f}s ✓ '
                          f'({validation["line_count"]} lines, '
                          f'{validation["arabic_chars"]} ar chars)')
                else:
                    issues = ', '.join(validation['issues'])
                    print(f'{elapsed:.1f}s ⚠ ({issues})')

                page_output_parts.append(response)
                _cache_set(ckey, response)
                # ── Write to live output file immediately ──
                live_f.write(response + '\n\n')
                live_f.flush()
            else:
                print(f'FAILED')
                # Fallback: use the longest version as-is
                fallback = max(aligned_block.values(), key=len)
                page_output_parts.append(fallback)
                live_f.write(fallback + '\n\n')
                live_f.flush()

            total_blocks_processed += 1

        # Assemble page output
        if page_output_parts:
            output_parts[page_num] = '\n\n'.join(page_output_parts)

    # ── 5. Finalize output file ──────────────────────────────────────
    elapsed_total = time.time() - start_time

    live_f.write('=' * 62 + '\n')
    live_f.write(f'  END OF DOCUMENT\n')
    live_f.write(f'  Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    live_f.write(f'  Total time: {elapsed_total:.1f}s\n')
    live_f.write('=' * 62 + '\n')
    live_f.flush()
    live_f.close()

    # ── 6. Summary ───────────────────────────────────────────────────
    pages_done = len([p for p in all_page_nums
                      if any(p in v for v in all_versions.values())])
    print(f'\n{"=" * 62}')
    print(f'  ✓ Output:       {OUTPUT_FILE}  (live-written)')
    print(f'  ✓ Pages:        {pages_done}')
    print(f'  ✓ Blocks:       {total_blocks_processed}')
    print(f'  ✓ Cached:       {total_cached}')
    print(f'  ✓ Time:         {elapsed_total:.1f}s')
    print(f'  ✓ Model:        {model}')
    if not dry_run:
        size = os.path.getsize(output_path)
        print(f'  ✓ File size:    {size:,} bytes')
    print(f'  💡 Monitor:     tail -f {OUTPUT_FILE}')
    print(f'{"=" * 62}')


# ====================================================================
#  CLI
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Multi-Version OCR Consensus Pipeline')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'Ollama model name (default: {DEFAULT_MODEL})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show block alignment without calling LLM')
    parser.add_argument('--max-pages', type=int, default=0,
                        help='Limit processing to first N pages (0=all)')
    parser.add_argument('--pages', type=str, default='',
                        help='Comma-separated page numbers to process (e.g. 4,5,6)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear the response cache before running')
    parser.add_argument('--output', type=str, default='',
                        help='Override output filename')

    args = parser.parse_args()

    if args.output:
        global OUTPUT_FILE
        OUTPUT_FILE = args.output

    if args.clear_cache and os.path.exists(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f'  Cache cleared: {CACHE_DIR}')

    pages_list = None
    if args.pages:
        pages_list = [int(p.strip()) for p in args.pages.split(',') if p.strip()]

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    run_pipeline(
        model=args.model,
        dry_run=args.dry_run,
        max_pages=args.max_pages,
        pages=pages_list,
    )


if __name__ == '__main__':
    main()
