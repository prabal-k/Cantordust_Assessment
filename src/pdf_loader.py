"""PyMuPDF wrapper. Returns page-indexed text dict so every downstream extraction
can cite a source page."""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def load_pdf_pages(path: str | Path) -> dict[int, str]:
    """Return {page_number_1_indexed: page_text}."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")
    doc = fitz.open(p)
    return {i + 1: page.get_text("text") for i, page in enumerate(doc)}


def slice_pages(pages: dict[int, str], wanted: list[int]) -> str:
    """Concatenate the requested pages into one labeled string.

    Each page is prefixed with `=== page N ===` so the LLM can cite page numbers
    accurately when filling FieldClaim.source_page.
    """
    chunks: list[str] = []
    for n in wanted:
        if n not in pages:
            continue
        chunks.append(f"=== page {n} ===\n{pages[n]}")
    return "\n\n".join(chunks)
