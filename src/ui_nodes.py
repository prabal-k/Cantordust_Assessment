"""Streamlit UI helpers for streaming node cards.

`node_status` survives reruns; placeholders are recreated each rerun. Each
card has a status icon, description, optional token stream (live tail +
expandable full buffer / JSON tree on completion), and a final summary.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

import streamlit as st

from src.streaming import ALL_NODES, NODE_META


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_json_if_possible(text: str) -> Any:
    if not text:
        return None
    stripped = _FENCE_RE.sub("", text).strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        l = stripped.find(opener)
        r = stripped.rfind(closer)
        if l != -1 and r > l:
            try:
                return json.loads(stripped[l : r + 1])
            except Exception:
                continue
    return None


def _pretty_json(text: str) -> tuple[str, Any]:
    parsed = _parse_json_if_possible(text)
    if parsed is None:
        return text, None
    try:
        return json.dumps(parsed, indent=2, ensure_ascii=False), parsed
    except Exception:
        return text, None


# Status flags stored in session_state.node_status[node]["state"]
PENDING = "pending"
RUNNING = "running"
DONE = "done"
SKIPPED = "skipped"
ERROR = "error"

STATE_ICON = {
    PENDING: "·",
    RUNNING: "▶",
    DONE: "✓",
    SKIPPED: "—",
    ERROR: "✕",
}


STATE_COLOR = {
    PENDING: "#475569",   # slate
    RUNNING: "#22c55e",   # green-500
    DONE: "#16a34a",      # green-600
    SKIPPED: "#94a3b8",   # slate-400
    ERROR: "#ef4444",     # red-500
}


def init_session_state() -> None:
    if "node_status" not in st.session_state:
        st.session_state.node_status = {
            n: {
                "state": PENDING, "desc": "", "summary": "", "elapsed": None,
                "tokens": "", "tokens_parsed": None,
            }
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
    """Build the placeholder containers for one node card."""
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
        "tokens": tokens_preview,  # back-compat alias
        "summary": summary,
        "elapsed": elapsed,
        "is_llm": meta["is_llm"],
    }
    _paint_from_status(node_key, placeholders)
    st.session_state.placeholders[node_key] = placeholders
    return placeholders


def _preview_text(text: str, max_lines: int = 5, max_chars: int = 400) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) >= max_lines:
        return "\n".join(lines[-max_lines:])
    return text[-max_chars:]


def _paint_from_status(node_key: str, ph: dict) -> None:
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
            _preview_text(status["tokens"]), language="json", wrap_lines=True
        )
    if ph.get("tokens_full") is not None and status.get("tokens"):
        parsed = status.get("tokens_parsed")
        if state == DONE and parsed is not None:
            ph["tokens_full"].json(parsed, expanded=False)
        else:
            ph["tokens_full"].code(
                status["tokens"][-6000:], language="json", wrap_lines=True
            )
    if status.get("summary"):
        if state == ERROR:
            ph["summary"].error(status["summary"])
        elif state == DONE:
            ph["summary"].success(status["summary"])
        else:
            ph["summary"].info(status["summary"])


def render_all_cards(order: Iterable[str]) -> None:
    for n in order:
        render_node_card(n)

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
    status["tokens_parsed"] = None
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
    """Append text to the buffer; repaint every `throttle_chars` chars."""
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
    ph["tokens_preview"].code(_preview_text(full), language="json", wrap_lines=True)
    if ph.get("tokens_full") is not None:
        ph["tokens_full"].code(full[-6000:], language="json", wrap_lines=True)


def on_node_done(node_key: str, summary: str, elapsed: float, *, final_buffer: str = "") -> None:
    status = st.session_state.node_status[node_key]
    status["state"] = DONE
    status["summary"] = summary
    status["elapsed"] = elapsed
    if final_buffer:
        pretty, parsed = _pretty_json(final_buffer)
        status["tokens"] = pretty
        status["tokens_parsed"] = parsed
    ph = st.session_state.placeholders.get(node_key)
    if ph is None:
        return
    ph["header"].markdown(
        f"{STATE_ICON[DONE]} {ph['icon']} "
        f"<span class='node-header'>{node_key}</span>",
        unsafe_allow_html=True,
    )
    ph["elapsed"].caption(f"{elapsed:.1f}s")
    pretty = status.get("tokens", "")
    parsed = status.get("tokens_parsed")
    if ph.get("tokens_preview") is not None and pretty:
        ph["tokens_preview"].code(
            _preview_text(pretty), language="json", wrap_lines=True
        )
    if ph.get("tokens_full") is not None and pretty:
        if parsed is not None:
            ph["tokens_full"].json(parsed, expanded=False)
        else:
            ph["tokens_full"].code(
                pretty[-6000:], language="json", wrap_lines=True
            )
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
