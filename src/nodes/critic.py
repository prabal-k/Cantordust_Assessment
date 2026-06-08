"""Node 10: self-review.

Re-reads the draft against NEPQA source text. Flags:
  - numeric claims not supported by NEPQA pages,
  - low-confidence FieldClaims stated as fact,
  - COVERED items with weak evidence.

Also produces an "Ask the factory" punch list — concrete items SunBridge should
chase before submitting to the import agent.

Uses a separate LLM invocation at temperature=0 for tighter behavior.
"""
from __future__ import annotations

import json

from src.config import CRITIC_TEMPERATURE, NEPQA_USEFUL_PAGES
from src.llm import invoke_structured
from src.pdf_loader import slice_pages
from src.prompts import CRITIC_SYSTEM, critic_user
from src.schemas import CriticReport
from src.state import AgentState


def critic_node(state: AgentState) -> AgentState:
    draft = state.get("draft_markdown", "")
    if not draft:
        return {"critic_flags": [], "ask_factory_list": []}

    nepqa_text = slice_pages(state["nepqa_pages"], NEPQA_USEFUL_PAGES)
    chosen = state["chosen_record"]
    evidence = {
        "chosen_record": chosen.model_dump(),
        "coverage_count": len(state.get("coverage", [])),
        "mismatch_count": len(state.get("mismatches", [])),
    }
    report = invoke_structured(
        CriticReport,
        CRITIC_SYSTEM,
        critic_user(draft, nepqa_text, json.dumps(evidence, indent=2, default=str)),
        temperature=CRITIC_TEMPERATURE,
    )
    return {"critic_flags": report.flags, "ask_factory_list": report.ask_factory}
