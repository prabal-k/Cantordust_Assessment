"""Node 1: load all 3 PDFs into state as {page: text} dicts."""
from __future__ import annotations

from src.pdf_loader import load_pdf_pages
from src.state import AgentState


def load_pdfs_node(state: AgentState) -> AgentState:
    return {
        "pdf1_pages": load_pdf_pages(state["pdf1_path"]),
        "pdf2_pages": load_pdf_pages(state["pdf2_path"]),
        "nepqa_pages": load_pdf_pages(state["nepqa_path"]),
    }
