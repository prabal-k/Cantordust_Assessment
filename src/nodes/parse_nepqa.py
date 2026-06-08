"""Node 4: extract NEPQA Section 1.4 PV Inverter checklist."""
from __future__ import annotations

from src.config import NEPQA_USEFUL_PAGES
from src.llm import invoke_structured
from src.pdf_loader import slice_pages
from src.prompts import NEPQA_EXTRACTION_SYSTEM, nepqa_extraction_user
from src.schemas import NEPQAChecklist
from src.state import AgentState


def parse_nepqa_node(state: AgentState) -> AgentState:
    sliced = slice_pages(state["nepqa_pages"], NEPQA_USEFUL_PAGES)
    checklist = invoke_structured(
        NEPQAChecklist,
        NEPQA_EXTRACTION_SYSTEM,
        nepqa_extraction_user(sliced),
    )
    return {"nepqa_checklist": checklist.items}
