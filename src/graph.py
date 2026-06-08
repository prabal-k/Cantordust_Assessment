"""LangGraph StateGraph wiring.

Pipeline:
    START → load_pdfs → [extract_pdf1, extract_pdf2, parse_nepqa] (parallel)
          → variant_detector
          → (conditional) human_in_loop OR auto_choose
          → reconciler → nepqa_mapper → drafter → critic
          → (conditional) patch_extractor → drafter → critic (loop) OR END
"""
from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes.critic import critic_node
from src.nodes.drafter import drafter_node
from src.nodes.extract_pdf1 import extract_pdf1_node
from src.nodes.extract_pdf2 import extract_pdf2_node
from src.nodes.human_in_loop import (
    auto_choose_node,
    human_in_loop_node,
)
from src.nodes.load_pdfs import load_pdfs_node
from src.nodes.nepqa_mapper import nepqa_mapper_node
from src.nodes.parse_nepqa import parse_nepqa_node
from src.nodes.patch_extractor import patch_extractor_node
from src.nodes.reconciler import reconciler_node
from src.nodes.variant_detector import needs_human_router, variant_detector_node
from src.state import AgentState


def should_retry(state: AgentState) -> Literal["patch_extractor", "end"]:
    """Self-correction conditional. After critic, retry if there are flags and
    we haven't exhausted the retry budget."""
    flags = state.get("critic_flags") or []
    retries = state.get("retry_count", 0)
    max_r = state.get("max_retries", 2)
    if flags and retries < max_r:
        return "patch_extractor"
    return "end"


def build_graph(use_checkpointer: bool = False):
    g = StateGraph(AgentState)

    g.add_node("load_pdfs", load_pdfs_node)
    g.add_node("extract_pdf1", extract_pdf1_node)
    g.add_node("extract_pdf2", extract_pdf2_node)
    g.add_node("parse_nepqa", parse_nepqa_node)
    g.add_node("variant_detector", variant_detector_node)
    g.add_node("human_in_loop", human_in_loop_node)
    g.add_node("auto_choose", auto_choose_node)
    g.add_node("reconciler", reconciler_node)
    g.add_node("nepqa_mapper", nepqa_mapper_node)
    g.add_node("drafter", drafter_node)
    g.add_node("critic", critic_node)
    g.add_node("patch_extractor", patch_extractor_node)

    g.add_edge(START, "load_pdfs")

    # Fan-out: 3 parallel extraction nodes
    g.add_edge("load_pdfs", "extract_pdf1")
    g.add_edge("load_pdfs", "extract_pdf2")
    g.add_edge("load_pdfs", "parse_nepqa")

    # Join at variant_detector (LangGraph waits for all 3)
    g.add_edge("extract_pdf1", "variant_detector")
    g.add_edge("extract_pdf2", "variant_detector")
    g.add_edge("parse_nepqa", "variant_detector")

    # Conditional: human or auto
    g.add_conditional_edges(
        "variant_detector",
        needs_human_router,
        {
            "human_in_loop": "human_in_loop",
            "auto_choose": "auto_choose",
        },
    )

    g.add_edge("human_in_loop", "reconciler")
    g.add_edge("auto_choose", "reconciler")
    g.add_edge("reconciler", "nepqa_mapper")
    g.add_edge("nepqa_mapper", "drafter")
    g.add_edge("drafter", "critic")

    # Self-correction conditional: loop back to patch_extractor or end
    g.add_conditional_edges(
        "critic",
        should_retry,
        {
            "patch_extractor": "patch_extractor",
            "end": END,
        },
    )
    g.add_edge("patch_extractor", "drafter")

    if use_checkpointer:
        return g.compile(
            checkpointer=MemorySaver(), interrupt_before=["human_in_loop"]
        )
    return g.compile()
