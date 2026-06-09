"""Markdown → PDF. Tries WeasyPrint first (needs GTK3 on Windows), then
PyMuPDF Story API (pure-Python). Same CSS feeds both; PyMuPDF understands
a subset, so styling degrades gracefully."""
from __future__ import annotations

from pathlib import Path

import markdown as md


def _try_weasyprint():
    try:
        from weasyprint import CSS, HTML  # type: ignore

        return HTML, CSS
    except (ImportError, OSError):
        return None, None


def _try_pymupdf():
    try:
        import pymupdf  # type: ignore

        if not hasattr(pymupdf, "Story"):
            return None
        return pymupdf
    except ImportError:
        return None


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


def _markdown_to_html(md_text: str) -> str:
    body = md.markdown(md_text, extensions=["tables", "fenced_code"])
    return (
        "<html><head><meta charset='utf-8'><style>"
        f"{_CSS}"
        "</style></head><body>"
        f"{body}"
        "</body></html>"
    )


def _render_pymupdf(md_text: str, out_pdf: Path) -> bool:
    pymupdf = _try_pymupdf()
    if pymupdf is None:
        return False
    html_doc = _markdown_to_html(md_text)
    try:
        story = pymupdf.Story(html=html_doc, user_css=_CSS)
        writer = pymupdf.DocumentWriter(str(out_pdf))
        page_w, page_h = pymupdf.paper_size("a4")
        media_box = pymupdf.Rect(0, 0, page_w, page_h)
        margin = 36
        where = pymupdf.Rect(
            margin, margin, page_w - margin, page_h - margin
        )
        more = 1
        while more:
            dev = writer.begin_page(media_box)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()
        writer.close()
        return True
    except Exception:
        return False


def markdown_to_pdf(md_text: str, out_pdf: Path) -> bool:
    """Render markdown to PDF. Returns True on success."""
    HTML, CSS = _try_weasyprint()
    if HTML is not None:
        try:
            html_doc = _markdown_to_html(md_text)
            HTML(string=html_doc).write_pdf(
                str(out_pdf), stylesheets=[CSS(string=_CSS)]
            )
            return True
        except Exception:
            pass
    return _render_pymupdf(md_text, out_pdf)


def save_markdown(md_text: str, out_md: Path) -> None:
    out_md.write_text(md_text, encoding="utf-8")
