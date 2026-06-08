"""Self-correction node.

Reads chosen ProductRecord + critic_flags + source PDF pages. Re-extracts ONLY
the flagged fields and returns the patched ProductRecord. Increments
retry_count + appends to patch_history. Clears critic_flags, ask_factory_list,
and draft_markdown so the downstream drafter + critic run fresh.

Uses the SAME `ProductRecord` schema (already Gemini/Groq-compatible) — a narrow
diff schema would add Pydantic surface area + a merge step for marginal token
savings. The prompt instructs the LLM to copy non-flagged fields verbatim.
"""
from __future__ import annotations

import json

from src.config import PDF1_USEFUL_PAGES, PDF2_USEFUL_PAGES
from src.llm import invoke_structured
from src.pdf_loader import slice_pages
from src.prompts import PATCH_EXTRACTOR_SYSTEM, patch_extractor_user
from src.schemas import ProductRecord
from src.state import AgentState


def _source_pages_for(record: ProductRecord, state: AgentState) -> str:
    """Pull the useful pages of the PDF the chosen record came from."""
    if record.source_doc == "pdf1":
        return slice_pages(state["pdf1_pages"], PDF1_USEFUL_PAGES)
    return slice_pages(state["pdf2_pages"], PDF2_USEFUL_PAGES)


def patch_extractor_node(state: AgentState) -> AgentState:
    chosen = state["chosen_record"]
    flags = state.get("critic_flags") or []
    if not flags:
        # Nothing to patch; pass through unchanged but advance retry counter.
        return {
            "retry_count": state.get("retry_count", 0) + 1,
        }

    flags_json = json.dumps(
        [f.model_dump() for f in flags], indent=2, ensure_ascii=False
    )
    record_json = chosen.model_dump_json(indent=2)
    source_text = _source_pages_for(chosen, state)

    patched = invoke_structured(
        ProductRecord,
        PATCH_EXTRACTOR_SYSTEM,
        patch_extractor_user(record_json, flags_json, source_text),
    )

    # Force source_doc + family_label + document_type to match the original.
    # The prompt says so, but belt + suspenders.
    updates = {}
    if patched.source_doc != chosen.source_doc:
        updates["source_doc"] = chosen.source_doc
    if patched.family_label != chosen.family_label:
        updates["family_label"] = chosen.family_label
    if updates:
        patched = patched.model_copy(update=updates)

    retry_count = state.get("retry_count", 0) + 1
    patch_history = list(state.get("patch_history", []))
    patch_history.append(
        {
            "attempt": retry_count,
            "flag_count_before": len(flags),
            "flagged_sections": [f.section for f in flags],
        }
    )

    return {
        "chosen_record": patched,
        "retry_count": retry_count,
        "patch_history": patch_history,
        # Clear downstream artifacts so drafter + critic run clean.
        "critic_flags": [],
        "ask_factory_list": [],
        "draft_markdown": "",
    }
