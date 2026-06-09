"""Classify the relationship between PDF1 and PDF2 ProductRecords.

Primary path: ReAct agent with 5 tools (compare_field, get_models,
check_factory_match, check_certifications_overlap, commit_decision).
Fallback: single-shot structured call if the agent never commits or its
args fail validation. A Python sanity-check then overrides any verdict
that contradicts disjoint-models + different-phase + different-family.
"""
from __future__ import annotations

from src.config import VARIANT_AGENT_MAX_TOOL_CALLS
from src.llm import current_on_token, get_llm, invoke_structured
from src.nodes.variant_tools import (
    build_decision_from_sink,
    build_variant_tools,
    format_tool_trace,
)
from src.prompts import (
    VARIANT_AGENT_SYSTEM,
    VARIANT_DETECTOR_SYSTEM,
    variant_agent_user,
    variant_detector_user,
)
from src.schemas import ProductRecord, VariantDecision, VariantRelationship
from src.state import AgentState


def _model_set(record: ProductRecord) -> set[str]:
    return {fc.value for fc in record.model_numbers if isinstance(fc.value, str)}


def _phase_value(record: ProductRecord) -> str | None:
    p = record.electrical.phase
    return str(p.value) if p is not None else None


def _sanity_check_different(p1: ProductRecord, p2: ProductRecord) -> bool:
    """True iff model sets disjoint, family labels differ, phase differs."""
    if _model_set(p1) & _model_set(p2):
        return False
    if p1.family_label == p2.family_label:
        return False
    phase1, phase2 = _phase_value(p1), _phase_value(p2)
    if not phase1 or not phase2 or phase1 == phase2:
        return False
    return True


def _emit(text: str) -> None:
    cb = current_on_token.get()
    if cb is not None:
        cb(text)


def _run_react_agent(p1: ProductRecord, p2: ProductRecord) -> VariantDecision | None:
    """Stream the ReAct loop; break the moment `commit_decision` populates
    the sink so the model can't re-run the evidence sequence after committing.
    Returns None on failure (caller falls back to single-shot)."""
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage

    sink: list[dict] = []
    tools = build_variant_tools(p1, p2, sink)
    llm = get_llm()

    _emit(
        f"\n[variant_detector] Starting ReAct loop "
        f"(max {VARIANT_AGENT_MAX_TOOL_CALLS} tool calls, "
        "early-break on commit)\n"
    )

    try:
        agent = create_react_agent(llm, tools=tools, prompt=VARIANT_AGENT_SYSTEM)
    except TypeError:
        # langgraph < 0.2.50 expects state_modifier=
        agent = create_react_agent(
            llm, tools=tools, state_modifier=VARIANT_AGENT_SYSTEM
        )

    initial = {
        "messages": [
            HumanMessage(
                content=variant_agent_user(
                    p1.model_dump_json(),
                    p2.model_dump_json(),
                )
            )
        ]
    }
    config = {"recursion_limit": 2 * VARIANT_AGENT_MAX_TOOL_CALLS + 4}

    final_messages: list = []
    early_stop = False
    try:
        for update in agent.stream(initial, config=config, stream_mode="values"):
            final_messages = update.get("messages", final_messages)
            if sink:
                early_stop = True
                break
    except Exception as e:
        _emit(f"\n[agent fallback: {type(e).__name__}: {e}]\n")
        return None

    trace = format_tool_trace(final_messages)
    if trace:
        _emit("\n" + trace + "\n")

    if not sink:
        _emit("\n[agent fallback: no commit_decision call]\n")
        return None

    if early_stop:
        _emit("\n[early-stop: commit_decision recorded, loop terminated]\n")

    try:
        decision = build_decision_from_sink(sink)
    except Exception as e:
        _emit(f"\n[agent fallback: commit args invalid — {e}]\n")
        return None

    if decision is None:
        return None

    _emit(
        f"\n[commit_decision] relationship={decision.relationship.value}, "
        f"requires_human_choice={decision.requires_human_choice}\n"
    )
    return decision


def _fallback_single_shot(p1: ProductRecord, p2: ProductRecord) -> VariantDecision:
    return invoke_structured(
        VariantDecision,
        VARIANT_DETECTOR_SYSTEM,
        variant_detector_user(
            p1.model_dump_json(),
            p2.model_dump_json(),
        ),
    )


def variant_detector_node(state: AgentState) -> AgentState:
    p1 = state["pdf1_record"]
    p2 = state["pdf2_record"]

    decision = _run_react_agent(p1, p2)
    if decision is None:
        decision = _fallback_single_shot(p1, p2)

    if _sanity_check_different(p1, p2):
        if decision.relationship in (
            VariantRelationship.SAME_PRODUCT,
            VariantRelationship.VARIANT,
        ):
            decision = decision.model_copy(
                update={
                    "relationship": VariantRelationship.DIFFERENT_FAMILY,
                    "requires_human_choice": True,
                    "reasoning": (
                        decision.reasoning
                        + " | Sanity override: model sets disjoint, family labels differ, phase differs."
                    ),
                }
            )
        else:
            decision = decision.model_copy(update={"requires_human_choice": True})

    return {"variant_decision": decision}


def needs_human_router(state: AgentState) -> str:
    decision = state.get("variant_decision")
    if decision is None:
        return "reconciler"
    return "human_in_loop" if decision.requires_human_choice else "auto_choose"
