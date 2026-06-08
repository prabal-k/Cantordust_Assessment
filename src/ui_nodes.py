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
    RUNNING: "⏳",
    DONE: "✅",
    SKIPPED: "⚪",
    ERROR: "❌",
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
    of placeholders so the consumer can update them mid-stream."""
    meta = NODE_META[node_key]
    icon = meta["icon"]
    with st.container(border=True):
        cols = st.columns([8, 2])
        with cols[0]:
            header = st.empty()
            desc = st.empty()
        with cols[1]:
            elapsed = st.empty()
        tokens = st.empty() if meta["is_llm"] else None
        summary = st.empty()
    placeholders = {
        "icon": icon,
        "header": header,
        "desc": desc,
        "tokens": tokens,
        "summary": summary,
        "elapsed": elapsed,
        "is_llm": meta["is_llm"],
    }
    _paint_from_status(node_key, placeholders)
    st.session_state.placeholders[node_key] = placeholders
    return placeholders


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
    if ph["tokens"] is not None and status.get("tokens"):
        ph["tokens"].code(status["tokens"], language="json")
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
        f"{STATE_ICON[RUNNING]} {ph['icon']} **{node_key}**{suffix}"
    )
    ph["desc"].caption(desc)


def on_node_token(node_key: str, text: str, *, throttle_chars: int = 80) -> None:
    """Accumulate token text; repaint only every `throttle_chars` chars to keep
    the browser snappy."""
    status = st.session_state.node_status[node_key]
    prev_len = len(status.get("tokens", ""))
    status["tokens"] = status.get("tokens", "") + text
    new_len = len(status["tokens"])
    if new_len // throttle_chars == prev_len // throttle_chars:
        return
    ph = st.session_state.placeholders.get(node_key)
    if ph is None or ph["tokens"] is None:
        return
    # Trim very long buffers to last ~6 KB so the browser doesn't choke.
    display = status["tokens"][-6000:]
    ph["tokens"].code(display, language="json")


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
    ph["header"].markdown(f"{STATE_ICON[DONE]} {ph['icon']} **{node_key}**")
    ph["elapsed"].caption(f"{elapsed:.1f}s")
    if ph["tokens"] is not None and final_buffer:
        ph["tokens"].code(final_buffer[-6000:], language="json")
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
