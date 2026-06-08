"""Node 2: structured extraction → ProductRecord for PDF1 (Chisage test report)."""
from __future__ import annotations

from src.config import PDF1_USEFUL_PAGES
from src.llm import invoke_structured
from src.pdf_loader import slice_pages
from src.prompts import PRODUCT_EXTRACTION_SYSTEM, product_extraction_user
from src.schemas import ProductRecord
from src.state import AgentState


def extract_pdf1_node(state: AgentState) -> AgentState:
    sliced = slice_pages(state["pdf1_pages"], PDF1_USEFUL_PAGES)
    record = invoke_structured(
        ProductRecord,
        PRODUCT_EXTRACTION_SYSTEM,
        product_extraction_user("pdf1", sliced),
    )
    # Guarantee the source_doc is correct even if the LLM strays.
    if record.source_doc != "pdf1":
        record = record.model_copy(update={"source_doc": "pdf1"})
    return {"pdf1_record": record}
