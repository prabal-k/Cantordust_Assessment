"""Streamlit UI helpers for streaming node cards.

A single dict in `st.session_state.node_status` survives reruns; placeholders
themselves are rebuilt every rerun (Streamlit objects don't survive). Each card
shows: status icon + name + elapsed, description, optional token stream, final
summary.
"""
from __future__ import annotations

from typing import Iterable

import streamlit as st

from src.streaming import ALL_NODES, NODE_META


# Status flags stored in session_state.node_status[node]["state"]
PENDING = "pending"
RUNNING = "running"
DONE = "done"
SKIPPED = "skipped"
ERROR = "error"

STATE_ICON = {
    PENDING: "·",
    RUNNING: "🟢",
    DONE: "✅",
    SKIPPED: "⚪",
    ERROR: "❌",
}


STATE_COLOR = {
    PENDING: "#475569",   # slate
    RUNNING: "#22c55e",   # green-500
    DONE: "#16a34a",      # green-600
    SKIPPED: "#94a3b8",   # slate-400
    ERROR: "#ef4444",     # red-500
}


def init_session_state() -> None:
    """Idempotently set up containers keyed by node name."""
    if "node_status" not in st.session_state:
        st.session_state.node_status = {
            n: {"state": PENDING, "desc": "", "summary": "", "elapsed": None, "tokens": ""}
            for n in ALL_NODES
        }
    if "placeholders" not in st.session_state:
        st.session_state.placeholders = {}


def reset_node_status() -> None:
    st.session_state.node_status = {
        n: {"state": PENDING, "desc": "", "summary": "", "elapsed": None, "tokens": ""}
        for n in ALL_NODES
    }


def render_node_card(node_key: str) -> dict:
    """Create the placeholder containers for one node card. Returns the dict
    of placeholders so the consumer can update them mid-stream.

    For LLM nodes the token stream is shown in two surfaces:
      - `tokens_preview`: a compact code block capped to the last ~5 lines /
        400 chars. Always visible while the node is running.
      - `tokens_full`: full buffer (last 6 KB) inside an `st.expander` that
        is collapsed by default. The user clicks to drill down.
    """
    meta = NODE_META[node_key]
    icon = meta["icon"]
    with st.container(border=True):
        cols = st.columns([8, 2])
        with cols[0]:
            header = st.empty()
            desc = st.empty()
        with cols[1]:
            elapsed = st.empty()
        if meta["is_llm"]:
            tokens_preview = st.empty()
            with st.expander("Show full stream", expanded=False):
                tokens_full = st.empty()
        else:
            tokens_preview = None
            tokens_full = None
        summary = st.empty()
    placeholders = {
        "icon": icon,
        "header": header,
        "desc": desc,
        "tokens_preview": tokens_preview,
        "tokens_full": tokens_full,
        # Back-compat alias so old code paths keep working.
        "tokens": tokens_preview,
        "summary": summary,
        "elapsed": elapsed,
        "is_llm": meta["is_llm"],
    }
    _paint_from_status(node_key, placeholders)
    st.session_state.placeholders[node_key] = placeholders
    return placeholders


def _preview_text(text: str, max_lines: int = 5, max_chars: int = 400) -> str:
    """Compact tail of a streaming buffer for the always-visible preview."""
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) >= max_lines:
        return "\n".join(lines[-max_lines:])
    return text[-max_chars:]


def _paint_from_status(node_key: str, ph: dict) -> None:
    """Render the persisted status into a freshly-created placeholder set."""
    status = st.session_state.node_status.get(node_key, {})
    state = status.get("state", PENDING)
    state_icon = STATE_ICON[state]
    ph["header"].markdown(f"{state_icon} {ph['icon']} **{node_key}**")
    if status.get("desc"):
        ph["desc"].caption(status["desc"])
    if status.get("elapsed") is not None:
        ph["elapsed"].caption(f"{status['elapsed']:.1f}s")
    if ph.get("tokens_preview") is not None and status.get("tokens"):
        ph["tokens_preview"].code(
            _preview_text(status["tokens"]), language="json"
        )
    if ph.get("tokens_full") is not None and status.get("tokens"):
        ph["tokens_full"].code(status["tokens"][-6000:], language="json")
    if status.get("summary"):
        if state == ERROR:
            ph["summary"].error(status["summary"])
        elif state == DONE:
            ph["summary"].success(status["summary"])
        else:
            ph["summary"].info(status["summary"])


def render_all_cards(order: Iterable[str]) -> None:
    """Render every card in `order` (used for phase one or phase two block)."""
    for n in order:
        render_node_card(n)


# --- Live-update helpers used by the streaming consumer ------------------

def on_node_start(
    node_key: str,
    desc: str,
    retry_attempt: int | None = None,
    max_retries: int | None = None,
) -> None:
    status = st.session_state.node_status[node_key]
    status["state"] = RUNNING
    status["desc"] = desc
    status["tokens"] = ""
    status["summary"] = ""
    status["elapsed"] = None
    ph = st.session_state.placeholders.get(node_key)
    if ph is None:
        return
    suffix = ""
    if retry_attempt is not None and max_retries is not None and retry_attempt > 0:
        suffix = f" — retry {retry_attempt}/{max_retries}"
    ph["header"].markdown(
        f"<span class='running-dot'></span> {ph['icon']} "
        f"<span class='node-header'>{node_key}</span>{suffix}",
        unsafe_allow_html=True,
    )
    ph["desc"].caption(desc)


def on_node_token(node_key: str, text: str, *, throttle_chars: int = 80) -> None:
    """Accumulate token text; repaint preview + full surfaces every
    `throttle_chars` chars to keep the browser snappy."""
    status = st.session_state.node_status[node_key]
    prev_len = len(status.get("tokens", ""))
    status["tokens"] = status.get("tokens", "") + text
    new_len = len(status["tokens"])
    if new_len // throttle_chars == prev_len // throttle_chars:
        return
    ph = st.session_state.placeholders.get(node_key)
    if ph is None or ph.get("tokens_preview") is None:
        return
    full = status["tokens"]
    ph["tokens_preview"].code(_preview_text(full), language="json")
    if ph.get("tokens_full") is not None:
        ph["tokens_full"].code(full[-6000:], language="json")


def on_node_done(node_key: str, summary: str, elapsed: float, *, final_buffer: str = "") -> None:
    status = st.session_state.node_status[node_key]
    status["state"] = DONE
    status["summary"] = summary
    status["elapsed"] = elapsed
    if final_buffer:
        status["tokens"] = final_buffer
    ph = st.session_state.placeholders.get(node_key)
    if ph is None:
        return
    ph["header"].markdown(
        f"{STATE_ICON[DONE]} {ph['icon']} "
        f"<span class='node-header'>{node_key}</span>",
        unsafe_allow_html=True,
    )
    ph["elapsed"].caption(f"{elapsed:.1f}s")
    if ph.get("tokens_preview") is not None and final_buffer:
        ph["tokens_preview"].code(_preview_text(final_buffer), language="json")
    if ph.get("tokens_full") is not None and final_buffer:
        ph["tokens_full"].code(final_buffer[-6000:], language="json")
    ph["summary"].success(summary)


def on_node_error(node_key: str, exc_repr: str) -> None:
    status = st.session_state.node_status[node_key]
    status["state"] = ERROR
    status["summary"] = exc_repr
    ph = st.session_state.placeholders.get(node_key)
    if ph is None:
        return
    ph["header"].markdown(f"{STATE_ICON[ERROR]} {ph['icon']} **{node_key}**")
    ph["summary"].error(exc_repr)


def on_node_skipped(node_key: str, reason: str = "") -> None:
    status = st.session_state.node_status[node_key]
    status["state"] = SKIPPED
    status["summary"] = reason or "skipped"
    ph = st.session_state.placeholders.get(node_key)
    if ph is None:
        return
    ph["header"].markdown(f"{STATE_ICON[SKIPPED]} {ph['icon']} **{node_key}**")
    if reason:
        ph["summary"].info(reason)
