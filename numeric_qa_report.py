#!/usr/bin/env python3
"""
numeric_qa_report.py — Financial-Grade Numeric QA Report
=========================================================

Generates a comprehensive QA report (HTML) showing:
  1. Which digits are LOCKED (frozen, high-confidence)
  2. Which tokens were re-validated (CNN or column boost)
  3. Which tokens remain UNTRUSTED (with reasons)
  4. Line-level stability results
  5. Summary statistics

NO "fixed" or "corrected" claims.
NO SSIM / pixel overlap columns.
NO dual-OCR agreement columns.
NO continuous trust scores.

Discrete trust only:
  LOCKED        -> 1.0
  SURYA_VALID   -> 0.85
  CNN_CONFIRMED -> 0.80
  UNTRUSTED     -> 0.0

Usage:
    from numeric_qa_report import generate_qa_report, PageNumericAudit
"""

from __future__ import annotations

import html as html_mod
from dataclasses import dataclass, field
from typing import List, Optional

from digit_ocr import TrustStatus, FailureReason, TokenOCRResult
from numeric_validator import PageStabilityResult
from numeric_reconstructor import NumericValue, page_trust_summary


# ────────────────────────────────────────────────────────────────────
#  Data Model
# ────────────────────────────────────────────────────────────────────
@dataclass
class PageNumericAudit:
    """Audit data for all numeric tokens on a single page."""
    page_number: int             # 1-based
    numeric_values: List[NumericValue] = field(default_factory=list)
    ocr_results: List[TokenOCRResult] = field(default_factory=list)
    line_stability: Optional[PageStabilityResult] = None
    trust_summary: dict = field(default_factory=dict)

    def compute_summary(self):
        self.trust_summary = page_trust_summary(self.numeric_values)


@dataclass
class PipelineNumericAudit:
    """Full pipeline audit for all pages."""
    pages: List[PageNumericAudit] = field(default_factory=list)
    total_numeric_tokens: int = 0
    total_trusted: int = 0
    total_untrusted: int = 0
    overall_trust_pct: float = 0.0
    has_partial_failure: bool = False

    def compute_overall(self):
        self.total_numeric_tokens = sum(
            len(p.numeric_values) for p in self.pages)
        self.total_trusted = sum(
            sum(1 for v in p.numeric_values
                if v.status != TrustStatus.UNTRUSTED)
            for p in self.pages)
        self.total_untrusted = self.total_numeric_tokens - self.total_trusted
        self.overall_trust_pct = (
            round(self.total_trusted / self.total_numeric_tokens * 100, 1)
            if self.total_numeric_tokens > 0 else 100.0)
        self.has_partial_failure = self.total_untrusted > 0


# ────────────────────────────────────────────────────────────────────
#  Trust Status Display
# ────────────────────────────────────────────────────────────────────
def _status_color(status: TrustStatus) -> str:
    """Map trust status to CSS color."""
    return {
        TrustStatus.LOCKED:        '#3b82f6',  # blue
        TrustStatus.SURYA_VALID:   '#22c55e',  # green
        TrustStatus.CNN_CONFIRMED: '#a78bfa',  # purple
        TrustStatus.UNTRUSTED:     '#ef4444',  # red
    }.get(status, '#94a3b8')


def _status_icon(status: TrustStatus) -> str:
    """Map trust status to icon."""
    return {
        TrustStatus.LOCKED:        '🔒',
        TrustStatus.SURYA_VALID:   '✓',
        TrustStatus.CNN_CONFIRMED: '🧠',
        TrustStatus.UNTRUSTED:     '⚠',
    }.get(status, '?')


# ────────────────────────────────────────────────────────────────────
#  Trust Heatmap (simple bar-based)
# ────────────────────────────────────────────────────────────────────
def _trust_heatmap_html(values: List[NumericValue]) -> str:
    """Generate a visual trust heatmap as HTML bars."""
    if not values:
        return '<p>No numeric tokens</p>'

    bars = []
    for i, v in enumerate(values):
        color = _status_color(v.status)
        icon = _status_icon(v.status)
        tooltip = (f'"{v.digits}" {v.status.value} '
                   f'(Surya conf={v.surya_confidence:.2f})')
        bars.append(
            f'<div class="hm-bar" style="background:{color};" '
            f'title="{html_mod.escape(tooltip)}">{icon}</div>'
        )

    return '<div class="heatmap">' + ''.join(bars) + '</div>'


# ────────────────────────────────────────────────────────────────────
#  HTML Report Generation
# ────────────────────────────────────────────────────────────────────
def generate_qa_report(audit: PipelineNumericAudit) -> str:
    """Generate comprehensive numeric QA report as HTML.

    Shows:
      - Which digits are LOCKED
      - Which tokens were re-validated
      - Which tokens remain UNTRUSTED (with reasons)
      - Line-level stability results
      - NO "fixed" or "corrected" claims
    """
    audit.compute_overall()

    # Status badge
    if audit.has_partial_failure:
        status_badge = ('<span class="badge badge-warn">'
                        '⚠ PARTIAL FAILURE — UNTRUSTED numbers detected</span>')
    else:
        status_badge = ('<span class="badge badge-ok">'
                        '✓ ALL NUMBERS VERIFIED</span>')

    # Build per-page sections
    page_sections = []
    for pa in audit.pages:
        pa.compute_summary()
        section = _render_page_section(pa)
        page_sections.append(section)

    pages_html = '\n'.join(page_sections)

    # Failure summary
    failure_summary = _render_failure_summary(audit)

    # Count totals by status
    total_locked = sum(p.trust_summary.get('locked', 0)
                       for p in audit.pages if p.trust_summary)
    total_surya = sum(p.trust_summary.get('surya_valid', 0)
                      for p in audit.pages if p.trust_summary)
    total_cnn = sum(p.trust_summary.get('cnn_confirmed', 0)
                    for p in audit.pages if p.trust_summary)
    total_revalidated = sum(p.trust_summary.get('revalidated', 0)
                            for p in audit.pages if p.trust_summary)

    return f"""<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
<meta charset="utf-8">
<title>Numeric QA Report — Financial-Grade Audit</title>
<style>
{_css()}
</style>
</head>
<body>
<div class="container">

<h1>🔒 Numeric QA Report — Financial-Grade Audit</h1>
<p class="architecture-note">Architecture: Surya-only OCR | Discrete Trust |
No Tesseract | No SSIM | No Dual-OCR</p>

<div class="summary-box">
  <h2>Overall Summary</h2>
  {status_badge}
  <div class="stats">
    <div class="stat">
      <span class="stat-val">{audit.total_numeric_tokens}</span>
      <span class="stat-label">Total Numeric Tokens</span>
    </div>
    <div class="stat locked">
      <span class="stat-val">{total_locked}</span>
      <span class="stat-label">🔒 LOCKED</span>
    </div>
    <div class="stat surya">
      <span class="stat-val">{total_surya}</span>
      <span class="stat-label">✓ SURYA_VALID</span>
    </div>
    <div class="stat cnn">
      <span class="stat-val">{total_cnn}</span>
      <span class="stat-label">🧠 CNN_CONFIRMED</span>
    </div>
    <div class="stat untrusted">
      <span class="stat-val">{audit.total_untrusted}</span>
      <span class="stat-label">⚠ UNTRUSTED</span>
    </div>
    <div class="stat revalidated">
      <span class="stat-val">{total_revalidated}</span>
      <span class="stat-label">🔄 Re-validated</span>
    </div>
  </div>
</div>

{failure_summary}

{pages_html}

<footer>
  <p>Generated by financial-grade numeric OCR pipeline (Surya-only).<br>
  Discrete trust model — no weighted averages, no silent corrections.<br>
  Every UNTRUSTED digit is listed. Uncertainty is exposed, never hidden.</p>
</footer>

</div>
</body>
</html>"""


def _render_page_section(pa: PageNumericAudit) -> str:
    """Render a single page's audit section."""
    s = pa.trust_summary
    status_cls = 'page-ok' if not s.get('partial_failure') else 'page-warn'

    # Trust heatmap
    heatmap = _trust_heatmap_html(pa.numeric_values)

    # Audit table rows — NO SSIM, NO pixel overlap, NO dual-agree
    rows = []
    for i, nv in enumerate(pa.numeric_values):
        status_cls_row = {
            TrustStatus.LOCKED: 'locked',
            TrustStatus.SURYA_VALID: 'trusted',
            TrustStatus.CNN_CONFIRMED: 'cnn',
            TrustStatus.UNTRUSTED: 'untrusted',
        }.get(nv.status, 'untrusted')

        icon = _status_icon(nv.status)
        reasons = ', '.join(r.value for r in nv.failure_reasons) if nv.failure_reasons else '—'

        revalidated_badge = '🔄' if nv.revalidated else ''

        rows.append(f'''<tr class="{status_cls_row}">
  <td class="digit-cell">{html_mod.escape(nv.digits)}</td>
  <td>{html_mod.escape(nv.original_text)}</td>
  <td class="score">{nv.surya_confidence:.3f}</td>
  <td class="status-cell" style="color:{_status_color(nv.status)}">{icon} {nv.status.value}</td>
  <td class="score">{nv.trust_score:.2f}</td>
  <td>{'🔒' if nv.locked else '—'}</td>
  <td>{'🧠' if nv.cnn_confirmed else '—'}</td>
  <td>{revalidated_badge}</td>
  <td class="reason">{reasons}</td>
</tr>''')

    rows_html = '\n'.join(rows)

    # Line stability section
    stability_html = ''
    if pa.line_stability and pa.line_stability.unstable_lines > 0:
        stability_rows = []
        for lr in pa.line_stability.line_results:
            if not lr.passed:
                stability_rows.append(
                    f'<tr class="unstable">'
                    f'<td>Line {lr.line_id}</td>'
                    f'<td>{html_mod.escape(lr.original_digits)}</td>'
                    f'<td>{html_mod.escape(lr.final_digits)}</td>'
                    f'<td>{", ".join(lr.failure_reasons)}</td>'
                    f'</tr>'
                )
        if stability_rows:
            stability_html = f'''
  <h3>⚠ Line Stability Failures</h3>
  <table class="stability-table">
    <thead><tr>
      <th>Line</th><th>Original Digits</th><th>Final Digits</th><th>Reason</th>
    </tr></thead>
    <tbody>{''.join(stability_rows)}</tbody>
  </table>'''

    return f'''
<div class="page-section {status_cls}">
  <h2>Page {pa.page_number}
    <span class="page-stats">
      {s.get("total", 0)} tokens |
      🔒 {s.get("locked", 0)} locked |
      ✓ {s.get("surya_valid", 0)} valid |
      🧠 {s.get("cnn_confirmed", 0)} cnn |
      ⚠ {s.get("untrusted", 0)} untrusted
    </span>
  </h2>

  <h3>Trust Heatmap</h3>
  {heatmap}

  <h3>Token Audit (Discrete Trust)</h3>
  <table class="audit-table">
    <thead>
      <tr>
        <th>Digits</th>
        <th>Original OCR</th>
        <th>Surya Conf</th>
        <th>Status</th>
        <th>Trust</th>
        <th>Locked</th>
        <th>CNN</th>
        <th>Re-val</th>
        <th>Failure Reason</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
  {stability_html}
</div>'''


def _render_failure_summary(audit: PipelineNumericAudit) -> str:
    """Render summary of all UNTRUSTED tokens across pages."""
    if not audit.has_partial_failure:
        return ('<div class="no-failures">'
                '<p>✓ No UNTRUSTED numbers detected. '
                'All numeric tokens are LOCKED or VERIFIED.</p></div>')

    failures = []
    for pa in audit.pages:
        for nv in pa.numeric_values:
            if nv.status == TrustStatus.UNTRUSTED:
                reasons = ', '.join(r.value for r in nv.failure_reasons)
                failures.append(
                    f'<li>P{pa.page_number}: '
                    f'<span class="digit-cell">{html_mod.escape(nv.digits)}</span> '
                    f'(Surya conf={nv.surya_confidence:.3f}) — {reasons}</li>')

    return f'''
<div class="failure-summary">
  <h2>⚠ UNTRUSTED Tokens ({len(failures)} numbers)</h2>
  <p class="failure-note">These tokens could not be verified.
  Digits are preserved as-is from Surya OCR — no corrections applied.</p>
  <ul>{''.join(failures)}</ul>
</div>'''


def _css() -> str:
    return """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    padding: 2rem;
}
.container { max-width: 1200px; margin: 0 auto; }
h1 {
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
    color: #f8fafc;
    border-bottom: 2px solid #334155;
    padding-bottom: 0.5rem;
}
.architecture-note {
    font-size: 0.8rem;
    color: #64748b;
    margin-bottom: 1.5rem;
    font-style: italic;
}
h2 { font-size: 1.3rem; margin: 1rem 0 0.5rem; color: #f1f5f9; }
h3 { font-size: 1rem; margin: 0.8rem 0 0.4rem; color: #94a3b8; }

.summary-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.stats { display: flex; gap: 1.2rem; margin-top: 1rem; flex-wrap: wrap; }
.stat {
    text-align: center;
    padding: 0.6rem 1rem;
    background: #0f172a;
    border-radius: 6px;
    min-width: 100px;
}
.stat-val { display: block; font-size: 1.8rem; font-weight: bold; }
.stat-label { font-size: 0.75rem; color: #94a3b8; }
.stat.locked .stat-val { color: #3b82f6; }
.stat.surya .stat-val { color: #22c55e; }
.stat.cnn .stat-val { color: #a78bfa; }
.stat.untrusted .stat-val { color: #ef4444; }
.stat.revalidated .stat-val { color: #f59e0b; }

.badge {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 6px;
    font-weight: bold;
    font-size: 0.9rem;
}
.badge-ok { background: #14532d; color: #22c55e; border: 1px solid #22c55e; }
.badge-warn { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }

.page-section {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.page-warn { border-color: #f59e0b; }
.page-ok { border-color: #22c55e; }
.page-stats { font-size: 0.8rem; color: #94a3b8; font-weight: normal; }

.heatmap { display: flex; gap: 3px; flex-wrap: wrap; align-items: center; }
.hm-bar {
    width: 28px;
    height: 24px;
    border-radius: 3px;
    cursor: pointer;
    text-align: center;
    line-height: 24px;
    font-size: 12px;
}

.audit-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}
.audit-table th {
    background: #0f172a;
    color: #94a3b8;
    padding: 0.5rem;
    text-align: left;
    border-bottom: 2px solid #334155;
    font-size: 0.75rem;
    text-transform: uppercase;
}
.audit-table td {
    padding: 0.4rem 0.5rem;
    border-bottom: 1px solid #1e293b;
}
.audit-table tr.locked td { background: rgba(59, 130, 246, 0.08); }
.audit-table tr.trusted td { background: rgba(34, 197, 94, 0.05); }
.audit-table tr.cnn td { background: rgba(167, 139, 250, 0.08); }
.audit-table tr.untrusted td { background: rgba(239, 68, 68, 0.08); }
.audit-table tr.untrusted td.status-cell { font-weight: bold; }

.stability-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    margin-top: 0.5rem;
}
.stability-table th {
    background: #0f172a;
    color: #94a3b8;
    padding: 0.4rem;
    text-align: left;
    border-bottom: 1px solid #334155;
}
.stability-table td { padding: 0.3rem 0.4rem; }
.stability-table tr.unstable td { background: rgba(239, 68, 68, 0.1); }

.digit-cell {
    font-family: 'Noto Naskh Arabic', 'Arial', serif;
    font-size: 1.1rem;
    direction: rtl;
}
.score { font-family: monospace; text-align: right; }
.reason { font-size: 0.75rem; color: #f59e0b; max-width: 200px; }

.failure-summary {
    background: #1e0000;
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
}
.failure-summary ul {
    list-style: none;
    padding: 0;
    margin-top: 0.5rem;
}
.failure-summary li {
    padding: 0.3rem 0;
    border-bottom: 1px solid #2d0000;
    font-size: 0.85rem;
}
.failure-note {
    font-size: 0.8rem;
    color: #fca5a5;
    font-style: italic;
    margin-top: 0.3rem;
}
.no-failures {
    background: #002200;
    border: 1px solid #22c55e;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    color: #22c55e;
}
footer {
    text-align: center;
    color: #475569;
    font-size: 0.75rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #1e293b;
}
"""
