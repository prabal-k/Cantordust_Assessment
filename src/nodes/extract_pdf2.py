"""Node 3: structured extraction → ProductRecord for PDF2 (Deye certificate)."""
from __future__ import annotations

from src.config import PDF2_USEFUL_PAGES
from src.llm import invoke_structured
from src.pdf_loader import slice_pages
from src.prompts import PRODUCT_EXTRACTION_SYSTEM, product_extraction_user
from src.schemas import ProductRecord
from src.state import AgentState


def extract_pdf2_node(state: AgentState) -> AgentState:
    sliced = slice_pages(state["pdf2_pages"], PDF2_USEFUL_PAGES)
    record = invoke_structured(
        ProductRecord,
        PRODUCT_EXTRACTION_SYSTEM,
        product_extraction_user("pdf2", sliced),
    )
    if record.source_doc != "pdf2":
        record = record.model_copy(update={"source_doc": "pdf2"})
    return {"pdf2_record": record}
