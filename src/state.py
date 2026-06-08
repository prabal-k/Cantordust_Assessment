"""LangGraph state container. TypedDict, total=False so nodes can write partial
updates without re-declaring every key."""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

from src.schemas import (
    CoverageResult,
    CriticFlag,
    HumanChoice,
    MismatchEntry,
    NEPQAItem,
    ProductRecord,
    VariantDecision,
)


class AgentState(TypedDict, total=False):
    # Inputs
    pdf1_path: str
    pdf2_path: str
    nepqa_path: str
    interface: Literal["cli", "streamlit"]

    # Human-friendly display names (filename stems) for citations + UI labels
    pdf1_name: str
    pdf2_name: str
    nepqa_name: str

    # Loaded raw text (page-indexed)
    pdf1_pages: dict[int, str]
    pdf2_pages: dict[int, str]
    nepqa_pages: dict[int, str]

    # Extraction outputs
    pdf1_record: ProductRecord
    pdf2_record: ProductRecord
    nepqa_checklist: list[NEPQAItem]

    # Decision + human input
    variant_decision: VariantDecision
    human_choice: HumanChoice
    chosen_record: ProductRecord

    # Analysis
    mismatches: list[MismatchEntry]
    coverage: list[CoverageResult]

    # Output
    draft_markdown: str
    draft_md_path: str
    draft_pdf_path: str
    critic_flags: list[CriticFlag]
    ask_factory_list: list[str]

    # Self-correction loop
    max_retries: int
    retry_count: int
    patch_history: list[dict]
    best_attempt: dict

    # Meta
    errors: list[str]
    run_timestamp: str
