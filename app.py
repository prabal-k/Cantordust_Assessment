"""Streamlit UI with live per-node progress + LLM token streaming.

Layout:
  - sidebar:   provider switch, input PDF paths, reset
  - main area: one card per node, split into 3 sections
                  Phase 1 (extraction)
                  Human-in-loop / auto-choose
                  Phase 2 (reconcile → map → draft → critic)
  - results:   tabs for draft preview, NEPQA coverage, mismatches, critic flags,
               downloads

Token streaming: when a Gemini/Groq node runs, its system prompt is augmented
with a strict JSON-only guard and we use `llm.stream()` to push token deltas
into the per-node code block. Updates are throttled to ~80-char repaints.

Human-in-loop is implemented as two separate generator passes
(`stream_phase_one`, `stream_phase_two`) split around the radio confirm button —
Streamlit's script-rerun model handles the pause naturally and we avoid
LangGraph's checkpointer interrupt machinery.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.nodes.human_in_loop import auto_choose_node
from src.schemas import HumanChoice
from src.streaming import (
    NODE_META,
    PHASE_ONE_ORDER,
    PHASE_TWO_ORDER,
    stream_phase_one,
    stream_phase_two,
)
from src.ui_nodes import (
    init_session_state,
    on_node_done,
    on_node_error,
    on_node_skipped,
    on_node_start,
    on_node_token,
    render_node_card,
    reset_node_status,
)


st.set_page_config(
    page_title="Nepal Compliance Drafter",
    layout="wide",
)

st.markdown(
    """
<style>
/* Slick node-card look */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
/* Pulsing green dot for currently-running header markers */
.running-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7);
    animation: pulse 1.5s infinite;
    margin-right: 6px;
    vertical-align: middle;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
    70%  { box-shadow: 0 0 0 10px rgba(34, 197, 94, 0); }
    100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
}
/* Phase subheaders cleaner */
h3 {
    border-bottom: 1px solid rgba(148, 163, 184, 0.18);
    padding-bottom: 4px;
    margin-top: 28px;
}
/* Slight glow on header markdown */
.node-header {
    font-weight: 600;
    font-size: 1.05rem;
}
</style>
    """,
    unsafe_allow_html=True,
)

st.title("Nepal Import Compliance Drafter")

init_session_state()


# --- Sidebar ------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    provider = st.radio(
        "LLM provider",
        ["gemini", "groq", "openrouter"],
        index=0,
        help=(
            "Switch backends; tokens stream from any provider. "
            "openrouter = OpenAI-API-compatible gateway w/ free Llama/DeepSeek "
            "(set OPENROUTER_API_KEY + OPENROUTER_MODEL in .env)."
        ),
    )
    os.environ["LLM_PROVIDER"] = provider
    # Live-read so the caption reflects what get_llm() will actually use, not
    # the radio default. Guards against stale-config-import bugs.
    from src.llm import get_active_provider

    _active = get_active_provider()
    _model_env = {
        "gemini": ("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        "groq": ("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "openrouter": ("OPENROUTER_MODEL", "qwen/qwen3-next-80b-a3b-instruct:free"),
    }[_active]
    _model = os.getenv(_model_env[0], _model_env[1])
    st.caption(f"Active: **{_active}** · model `{_model}`")

    max_retries = st.slider(
        "Critic self-correction retries",
        min_value=1,
        max_value=3,
        value=2,
        help=(
            "If the critic flags issues, the agent patches the flagged fields "
            "and re-drafts up to this many times. Default 2."
        ),
    )

    st.divider()
    st.subheader("Input PDFs")
    default_data = Path(r"C:\Users\praba\Desktop\Prabal\Docs\cantordust\data")
    pdf1_path = st.text_input(
        "Manufacturer doc 1 path",
        value=str(default_data / "DSS_GZES230100125901_combined-1.pdf"),
    )
    pdf2_path = st.text_input(
        "Manufacturer doc 2 path",
        value=str(default_data / "188_1115.pdf"),
    )
    nepqa_path = st.text_input(
        "Regulator doc (NEPQA) path",
        value=str(default_data / "nepqa_2025.pdf"),
    )

    st.divider()
    if st.button("Reset", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# --- Session state defaults ---------------------------------------------

st.session_state.setdefault("phase", "idle")     # idle | extracted | choice_made | done
st.session_state.setdefault("state", {})          # AgentState dict
st.session_state.setdefault("phase_one_ran", False)
st.session_state.setdefault("phase_two_ran", False)


# --- Top action bar -----------------------------------------------------

action = st.container()
with action:
    if st.session_state.phase == "idle":
        run_clicked = st.button("Run pipeline", type="primary", use_container_width=True)
    else:
        run_clicked = False
        st.caption(f"Current phase: `{st.session_state.phase}`")


# --- Render phase-one cards (only nodes that have already started) ------

if st.session_state.phase != "idle":
    st.subheader("Phase 1 — Extraction & variant detection")
    phase_one_cards = st.container()
    with phase_one_cards:
        from src.ui_nodes import PENDING

        for node_key in PHASE_ONE_ORDER:
            status = st.session_state.node_status.get(node_key, {})
            if status.get("state", PENDING) != PENDING:
                render_node_card(node_key)


# --- Phase 1 execution (when user clicks Run) ---------------------------

if run_clicked and st.session_state.phase == "idle":
    reset_node_status()
    st.subheader("Phase 1 — Extraction & variant detection")
    phase_one_cards = st.container()

    state: dict = {
        "pdf1_path": pdf1_path,
        "pdf2_path": pdf2_path,
        "nepqa_path": nepqa_path,
        "pdf1_name": Path(pdf1_path).stem,
        "pdf2_name": Path(pdf2_path).stem,
        "nepqa_name": Path(nepqa_path).stem,
        "interface": "streamlit",
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "max_retries": max_retries,
        "retry_count": 0,
        "patch_history": [],
        "best_attempt": {},
    }

    try:
        for ev in stream_phase_one(state):
            if ev.type == "node_start":
                # Lazy-render the card the FIRST time we hear about this node,
                # so the UI doesn't show empty placeholders for nodes that
                # haven't started yet.
                with phase_one_cards:
                    if ev.node not in st.session_state.placeholders:
                        render_node_card(ev.node)
                on_node_start(
                    ev.node,
                    ev.payload.get("desc", ""),
                    retry_attempt=ev.payload.get("retry_attempt"),
                    max_retries=ev.payload.get("max_retries"),
                )
            elif ev.type == "token":
                on_node_token(ev.node, ev.payload.get("text", ""))
            elif ev.type == "node_done":
                on_node_done(
                    ev.node,
                    summary=ev.payload.get("summary", ""),
                    elapsed=ev.payload.get("elapsed", 0.0),
                    final_buffer=ev.payload.get("buffer", ""),
                )
            elif ev.type == "node_skipped":
                on_node_skipped(ev.node, ev.payload.get("reason", ""))
            elif ev.type == "error":
                on_node_error(ev.node, ev.payload.get("exc", "unknown error"))
            elif ev.type == "phase_done":
                break
    except Exception as e:
        st.error(f"Phase 1 failed: {e}")
        st.stop()

    st.session_state.state = state
    st.session_state.phase = "extracted"
    st.session_state.phase_one_ran = True
    st.rerun()


# --- Variant decision + human-in-loop -----------------------------------

if st.session_state.phase in ("extracted", "choice_made", "done"):
    state = st.session_state.state
    p1 = state.get("pdf1_record")
    p2 = state.get("pdf2_record")
    decision = state.get("variant_decision")
    name1 = state.get("pdf1_name") or "pdf1"
    name2 = state.get("pdf2_name") or "pdf2"

    if p1 and p2 and decision:
        st.subheader("Extracted records")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{name1}** — {p1.family_label}")
            with st.expander("Show JSON", expanded=False):
                st.json(json.loads(p1.model_dump_json()))
        with c2:
            st.markdown(f"**{name2}** — {p2.family_label}")
            with st.expander("Show JSON", expanded=False):
                st.json(json.loads(p2.model_dump_json()))

        st.subheader("Variant decision")
        st.info(
            f"**Relationship**: `{decision.relationship.value}`  \n"
            f"**Reasoning**: {decision.reasoning}"
        )

        if st.session_state.phase == "extracted":
            if decision.requires_human_choice:
                st.warning(
                    "The two PDFs describe different product families. Pick the one this "
                    "shipment is about so the draft targets the right product."
                )
                sample1 = p1.model_numbers[0].value if p1.model_numbers else "—"
                sample2 = p2.model_numbers[0].value if p2.model_numbers else "—"
                def _pretty(label: str) -> str:
                    return label.replace("_", " ").title()
                choice = st.radio(
                    "Which product family is this shipment?",
                    options=["pdf1", "pdf2"],
                    format_func=lambda v: (
                        f"{name1} · {_pretty(p1.family_label)} ({sample1})"
                        if v == "pdf1"
                        else f"{name2} · {_pretty(p2.family_label)} ({sample2})"
                    ),
                    horizontal=True,
                )
                if st.button("Confirm choice", type="primary"):
                    chosen = p1 if choice == "pdf1" else p2
                    state["human_choice"] = HumanChoice(
                        chosen_family=choice, rationale="Streamlit radio"
                    )
                    state["chosen_record"] = chosen
                    st.session_state.phase = "choice_made"
                    st.rerun()
            else:
                if st.button("Continue (auto-choose)", type="primary"):
                    state.update(auto_choose_node(state))
                    st.session_state.phase = "choice_made"
                    st.rerun()


# --- Phase 2 execution --------------------------------------------------

if st.session_state.phase in ("choice_made", "done"):
    st.subheader("Phase 2 — Reconcile · Map · Draft · Critique")
    phase_two_cards = st.container()
    if st.session_state.phase == "done":
        from src.ui_nodes import PENDING

        with phase_two_cards:
            for node_key in PHASE_TWO_ORDER:
                status = st.session_state.node_status.get(node_key, {})
                if status.get("state", PENDING) != PENDING:
                    render_node_card(node_key)


if st.session_state.phase == "choice_made":
    state = st.session_state.state
    try:
        for ev in stream_phase_two(state):
            if ev.type == "node_start":
                with phase_two_cards:
                    if ev.node not in st.session_state.placeholders:
                        render_node_card(ev.node)
                on_node_start(
                    ev.node,
                    ev.payload.get("desc", ""),
                    retry_attempt=ev.payload.get("retry_attempt"),
                    max_retries=ev.payload.get("max_retries"),
                )
            elif ev.type == "token":
                on_node_token(ev.node, ev.payload.get("text", ""))
            elif ev.type == "node_done":
                on_node_done(
                    ev.node,
                    summary=ev.payload.get("summary", ""),
                    elapsed=ev.payload.get("elapsed", 0.0),
                    final_buffer=ev.payload.get("buffer", ""),
                )
            elif ev.type == "error":
                on_node_error(ev.node, ev.payload.get("exc", "unknown error"))
            elif ev.type == "phase_done":
                break
    except Exception as e:
        st.error(f"Phase 2 failed: {e}")
        st.stop()

    st.session_state.state = state
    st.session_state.phase = "done"
    st.session_state.phase_two_ran = True
    st.rerun()


# --- Results ------------------------------------------------------------

if st.session_state.phase == "done":
    state = st.session_state.state
    chosen = state["chosen_record"]
    st.success(
        f"Draft generated for **{chosen.family_label}** "
        f"({len(state.get('coverage', []))} NEPQA items checked)"
    )

    tabs = st.tabs([
        "Draft preview",
        "NEPQA coverage",
        "Mismatches",
        "Critic flags",
        "Downloads",
    ])

    with tabs[0]:
        st.markdown(state.get("draft_markdown", "_no draft_"))

    with tabs[1]:
        for cov in state.get("coverage", []):
            st.markdown(
                f"**{cov.item.clause_id}** — {cov.item.requirement_text}  \n"
                f"_status_: `{cov.status.value}` · _gap_: {cov.gap_note or '—'}"
            )

    with tabs[2]:
        mm = state.get("mismatches", [])
        if not mm:
            st.write("_no mismatches_")
        else:
            for m in mm:
                st.markdown(
                    f"**`{m.field_path}`** — `{m.severity.value}`  \n"
                    f"{state.get('pdf1_name') or 'pdf1'}: `{m.pdf1_value or '—'}` · "
                    f"{state.get('pdf2_name') or 'pdf2'}: `{m.pdf2_value or '—'}`  \n"
                    f"➡ {m.recommendation}"
                )

    with tabs[3]:
        flags = state.get("critic_flags", [])
        ask = state.get("ask_factory_list", [])
        if not flags and not ask:
            st.write("_no critic output_")
        if flags:
            st.subheader("Flags")
            for f in flags:
                st.markdown(
                    f"- **{f.section}**: {f.issue}  \n"
                    f"  > _{f.claim_excerpt}_  \n"
                    f"  ➡ {f.suggested_action}"
                )
        if ask:
            st.subheader("Ask the factory for")
            for item in ask:
                st.markdown(f"- {item}")

    with tabs[4]:
        md_path = state.get("draft_md_path")
        if md_path and Path(md_path).exists():
            st.download_button(
                "Download markdown",
                data=Path(md_path).read_bytes(),
                file_name=Path(md_path).name,
                mime="text/markdown",
            )
        pdf_path = state.get("draft_pdf_path")
        if pdf_path and Path(pdf_path).exists():
            st.download_button(
                "Download PDF",
                data=Path(pdf_path).read_bytes(),
                file_name=Path(pdf_path).name,
                mime="application/pdf",
            )
        else:
            st.caption(
                "PDF rendering skipped — both WeasyPrint and the PyMuPDF "
                "fallback failed. Markdown is still available above. Re-run "
                "the pipeline after a fresh start if you just upgraded."
            )
