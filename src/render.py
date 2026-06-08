"""Markdown → PDF renderer.

Primary backend: WeasyPrint (needs GTK3 on Windows).
Fallback: skip PDF generation gracefully if WeasyPrint unavailable.
"""
from __future__ import annotations

from pathlib import Path

import markdown as md


def _try_weasyprint():
    try:
        from weasyprint import CSS, HTML  # type: ignore

        return HTML, CSS
    except (ImportError, OSError):
        return None, None


_CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; color: #1a202c; line-height: 1.4; }
h1 { font-size: 18pt; color: #0b3d91; border-bottom: 2px solid #0b3d91; padding-bottom: 6px; }
h2 { font-size: 13pt; color: #0b3d91; margin-top: 16px; }
h3 { font-size: 11pt; color: #1a365d; margin-top: 12px; }
table { width: 100%; border-collapse: collapse; font-size: 9pt; margin: 8px 0; }
th, td { border: 1px solid #cbd5e0; padding: 5px 7px; vertical-align: top; text-align: left; }
th { background: #edf2f7; }
code { background: #f1f5f9; padding: 0 3px; border-radius: 2px; font-size: 9pt; }
blockquote { border-left: 3px solid #c2410c; padding: 4px 10px; color: #4a5568; }
.badge-covered { color: #166534; font-weight: 600; }
.badge-partial { color: #92400e; font-weight: 600; }
.badge-missing { color: #991b1b; font-weight: 600; }
"""


def markdown_to_pdf(md_text: str, out_pdf: Path) -> bool:
    """Render markdown to PDF. Returns True on success, False if WeasyPrint missing."""
    HTML, CSS = _try_weasyprint()
    if HTML is None:
        return False
    html_body = md.markdown(md_text, extensions=["tables", "fenced_code"])
    html_doc = f"<html><head><meta charset='utf-8'></head><body>{html_body}</body></html>"
    HTML(string=html_doc).write_pdf(str(out_pdf), stylesheets=[CSS(string=_CSS)])
    return True


def save_markdown(md_text: str, out_md: Path) -> None:
    out_md.write_text(md_text, encoding="utf-8")
