"""Streaming + per-node progress wiring.

Wraps each node call with start/done events and pipes LLM token deltas through
the `current_on_token` ContextVar in `src.llm`. Streamlit consumes the event
stream and renders one card per node.

Two generators model the human-in-loop split:
  - stream_phase_one: load_pdfs → extract_pdf1 → extract_pdf2 → parse_nepqa
                      → variant_detector → (auto_choose if no human needed)
  - stream_phase_two: reconciler → nepqa_mapper → drafter → critic

The "parallel" trio (extract_pdf1/2 + parse_nepqa) runs sequentially here so we
get clean visible token streams without thread races; LangGraph still fan-outs
for the production CLI path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from src.llm import current_on_token
from src.nodes.critic import critic_node
from src.nodes.drafter import drafter_node
from src.nodes.extract_pdf1 import extract_pdf1_node
from src.nodes.extract_pdf2 import extract_pdf2_node
from src.nodes.human_in_loop import auto_choose_node
from src.nodes.load_pdfs import load_pdfs_node
from src.nodes.nepqa_mapper import nepqa_mapper_node
from src.nodes.parse_nepqa import parse_nepqa_node
from src.nodes.patch_extractor import patch_extractor_node
from src.nodes.reconciler import reconciler_node
from src.nodes.variant_detector import variant_detector_node


# --- Event ---------------------------------------------------------------

@dataclass
class Event:
    type: str  # "node_start" | "token" | "node_done" | "node_skipped" | "error" | "phase_done"
    node: str
    payload: dict[str, Any] = field(default_factory=dict)


# --- NODE_META: per-node descriptions + summaries ------------------------

def _safe(fn: Callable[[dict], str], state: dict) -> str:
    try:
        return fn(state)
    except Exception:
        return ""


def _running_load_pdfs(s):
    return "Loading 3 PDFs from disk"


def _done_load_pdfs(s):
    return (
        f"Loaded pages: pdf1={len(s.get('pdf1_pages', {}))}, "
        f"pdf2={len(s.get('pdf2_pages', {}))}, "
        f"nepqa={len(s.get('nepqa_pages', {}))}"
    )


def _running_extract_pdf1(s):
    name = s.get("pdf1_name") or "pdf1"
    return f"Reading {name} (pages 1-8) → structured ProductRecord"


def _done_extract(record_key: str):
    def _f(s):
        r = s.get(record_key)
        if r is None:
            return ""
        return (
            f"Extracted {len(r.model_numbers)} models, "
            f"{len(r.certifications)} certifications, family={r.family_label}"
        )
    return _f


def _running_extract_pdf2(s):
    name = s.get("pdf2_name") or "pdf2"
    return f"Reading {name} (pages 1-4) → structured ProductRecord"


def _running_parse_nepqa(s):
    return "Parsing NEPQA 2025 §1.4 (PV Inverter / Grid Connected Inverter)"


def _done_parse_nepqa(s):
    items = s.get("nepqa_checklist") or []
    return f"Extracted {len(items)} requirement items"


def _running_variant_detector(s):
    p1 = s.get("pdf1_record")
    p2 = s.get("pdf2_record")
    if p1 and p2:
        return f"Comparing {p1.family_label} vs {p2.family_label}"
    return "Comparing PDF1 vs PDF2 records"


def _done_variant_detector(s):
    d = s.get("variant_decision")
    if d is None:
        return ""
    return (
        f"Decision: {d.relationship.value} "
        f"(human_needed={d.requires_human_choice})"
    )


def _running_auto_choose(s):
    return "Auto-selecting (variant detector says no human input needed)"


def _running_human(s):
    return "Awaiting your choice of product family"


def _done_chosen(s):
    c = s.get("chosen_record")
    if c is None:
        return ""
    return f"Chose {c.family_label} from {c.source_doc}"


def _running_reconciler(s):
    n1 = s.get("pdf1_name") or "pdf1"
    n2 = s.get("pdf2_name") or "pdf2"
    return f"Diffing {n1} vs {n2} across scalar fields + certifications"


def _done_reconciler(s):
    mm = s.get("mismatches") or []
    if not mm:
        return "No mismatches"
    from collections import Counter
    counts = Counter(m.severity.value for m in mm)
    return (
        f"{counts.get('CRITICAL', 0)} critical · "
        f"{counts.get('WARNING', 0)} warning · "
        f"{counts.get('INFO', 0)} info"
    )


def _running_nepqa_mapper(s):
    n = len(s.get("nepqa_checklist") or [])
    return f"Mapping {n} NEPQA requirements against chosen product record"


def _done_nepqa_mapper(s):
    cov = s.get("coverage") or []
    from collections import Counter
    counts = Counter(c.status.value for c in cov)
    return (
        f"🟢 {counts.get('COVERED', 0)} · "
        f"🟡 {counts.get('PARTIAL', 0)} · "
        f"🔴 {counts.get('MISSING', 0)} · "
        f"⚪ {counts.get('NOT_APPLICABLE', 0)}"
    )


def _running_drafter(s):
    return "Rendering markdown + PDF compliance draft to outputs/"


def _done_drafter(s):
    md = s.get("draft_markdown") or ""
    md_path = s.get("draft_md_path") or "—"
    return f"Draft {len(md)} chars → {md_path.split(chr(92))[-1] if md_path else '—'}"


def _running_critic(s):
    rc = s.get("retry_count", 0)
    return f"Self-reviewing draft against NEPQA source pages (attempt {rc + 1})"


def _done_critic(s):
    flags = s.get("critic_flags") or []
    asks = s.get("ask_factory_list") or []
    return f"{len(flags)} flags · {len(asks)} factory asks"


def _running_patch_extractor(s):
    flags = s.get("critic_flags") or []
    rc = s.get("retry_count", 0)
    mr = s.get("max_retries", 2)
    src = s.get("chosen_record")
    if src is None:
        src_name = "?"
    elif src.source_doc == "pdf1":
        src_name = s.get("pdf1_name") or "pdf1"
    elif src.source_doc == "pdf2":
        src_name = s.get("pdf2_name") or "pdf2"
    else:
        src_name = src.source_doc
    return (
        f"Patching {len(flags)} flagged fields from {src_name} "
        f"(retry {rc + 1}/{mr})"
    )


def _done_patch_extractor(s):
    history = s.get("patch_history") or []
    if not history:
        return ""
    last = history[-1]
    return (
        f"Patched fields: {', '.join(last.get('flagged_sections', [])) or '—'} "
        f"(attempt {last.get('attempt', '?')})"
    )


NODE_META: dict[str, dict] = {
    "load_pdfs": {
        "icon": "📄",
        "is_llm": False,
        "running": _running_load_pdfs,
        "done": _done_load_pdfs,
    },
    "extract_pdf1": {
        "icon": "🔍",
        "is_llm": True,
        "running": _running_extract_pdf1,
        "done": _done_extract("pdf1_record"),
    },
    "extract_pdf2": {
        "icon": "🔍",
        "is_llm": True,
        "running": _running_extract_pdf2,
        "done": _done_extract("pdf2_record"),
    },
    "parse_nepqa": {
        "icon": "📋",
        "is_llm": True,
        "running": _running_parse_nepqa,
        "done": _done_parse_nepqa,
    },
    "variant_detector": {
        "icon": "🔀",
        "is_llm": True,
        "running": _running_variant_detector,
        "done": _done_variant_detector,
    },
    "human_in_loop": {
        "icon": "🙋",
        "is_llm": False,
        "running": _running_human,
        "done": _done_chosen,
    },
    "auto_choose": {
        "icon": "🤖",
        "is_llm": False,
        "running": _running_auto_choose,
        "done": _done_chosen,
    },
    "reconciler": {
        "icon": "🔗",
        "is_llm": False,
        "running": _running_reconciler,
        "done": _done_reconciler,
    },
    "nepqa_mapper": {
        "icon": "🗺",
        "is_llm": False,
        "running": _running_nepqa_mapper,
        "done": _done_nepqa_mapper,
    },
    "drafter": {
        "icon": "📝",
        "is_llm": False,
        "running": _running_drafter,
        "done": _done_drafter,
    },
    "critic": {
        "icon": "🔎",
        "is_llm": True,
        "running": _running_critic,
        "done": _done_critic,
    },
    "patch_extractor": {
        "icon": "🔧",
        "is_llm": True,
        "running": _running_patch_extractor,
        "done": _done_patch_extractor,
    },
}


# Display order for the UI
PHASE_ONE_ORDER = [
    "load_pdfs",
    "extract_pdf1",
    "extract_pdf2",
    "parse_nepqa",
    "variant_detector",
]

PHASE_TWO_ORDER = [
    "reconciler",
    "nepqa_mapper",
    "drafter",
    "critic",
    "patch_extractor",
]

ALL_NODES = PHASE_ONE_ORDER + ["human_in_loop", "auto_choose"] + PHASE_TWO_ORDER


_NODE_FNS: dict[str, Callable[[dict], dict]] = {
    "load_pdfs": load_pdfs_node,
    "extract_pdf1": extract_pdf1_node,
    "extract_pdf2": extract_pdf2_node,
    "parse_nepqa": parse_nepqa_node,
    "variant_detector": variant_detector_node,
    "reconciler": reconciler_node,
    "nepqa_mapper": nepqa_mapper_node,
    "drafter": drafter_node,
    "critic": critic_node,
    "patch_extractor": patch_extractor_node,
}


# --- Generator runners ---------------------------------------------------

def _run_node(node_key: str, state: dict) -> Iterator[Event]:
    """Run one node, yielding node_start, optional tokens, node_done/error.

    For nodes that participate in the self-correction loop (critic,
    patch_extractor, drafter), the node_start payload includes retry_attempt
    and max_retries so the Streamlit UI can render a "Retry N/M" header.
    """
    meta = NODE_META[node_key]
    fn = _NODE_FNS[node_key]
    desc = _safe(meta["running"], state)
    payload = {"desc": desc, "icon": meta["icon"]}
    if node_key in ("critic", "patch_extractor", "drafter"):
        payload["retry_attempt"] = state.get("retry_count", 0)
        payload["max_retries"] = state.get("max_retries", 2)
    yield Event("node_start", node_key, payload)

    t0 = time.perf_counter()

    if meta["is_llm"]:
        # Pipe streamed tokens through the ContextVar so the underlying
        # invoke_structured call hits its streaming branch.
        buf: list[str] = []
        deferred: list[Event] = []

        def on_tok(text: str) -> None:
            buf.append(text)
            deferred.append(Event("token", node_key, {"text": text}))

        token = current_on_token.set(on_tok)
        try:
            try:
                update = fn(state)
            except Exception as e:
                yield from deferred
                yield Event("error", node_key, {"exc": repr(e)})
                raise
        finally:
            current_on_token.reset(token)

        # Drain any buffered token events first, then the done event.
        yield from deferred
        state.update(update)
        yield Event(
            "node_done",
            node_key,
            {
                "summary": _safe(meta["done"], state),
                "elapsed": time.perf_counter() - t0,
                "buffer": "".join(buf),
            },
        )
        return

    # Pure Python node
    try:
        update = fn(state)
    except Exception as e:
        yield Event("error", node_key, {"exc": repr(e)})
        raise
    state.update(update)
    yield Event(
        "node_done",
        node_key,
        {
            "summary": _safe(meta["done"], state),
            "elapsed": time.perf_counter() - t0,
        },
    )


def stream_phase_one(state: dict) -> Iterator[Event]:
    """Up through variant_detector. Stops before the human-in-loop break."""
    for key in PHASE_ONE_ORDER:
        yield from _run_node(key, state)
    yield Event("phase_done", "phase_one")


def _snapshot_attempt(state: dict) -> dict:
    """Return a copy of the artifacts we may need to restore if a later attempt
    regresses."""
    return {
        "flag_count": len(state.get("critic_flags") or []),
        "draft_markdown": state.get("draft_markdown", ""),
        "draft_md_path": state.get("draft_md_path", ""),
        "draft_pdf_path": state.get("draft_pdf_path", ""),
        "chosen_record": state.get("chosen_record"),
        "critic_flags": list(state.get("critic_flags") or []),
        "ask_factory_list": list(state.get("ask_factory_list") or []),
    }


def _restore_attempt(state: dict, attempt: dict) -> None:
    """Restore a snapshotted attempt into state."""
    state["draft_markdown"] = attempt["draft_markdown"]
    state["draft_md_path"] = attempt["draft_md_path"]
    state["draft_pdf_path"] = attempt["draft_pdf_path"]
    state["chosen_record"] = attempt["chosen_record"]
    state["critic_flags"] = attempt["critic_flags"]
    state["ask_factory_list"] = attempt["ask_factory_list"]


def stream_phase_two(state: dict) -> Iterator[Event]:
    """After the human radio. Runs reconciler → mapper → drafter → critic, then
    enters the self-correction loop:

      while critic flags > 0 AND retry_count < max_retries:
          patch_extractor → drafter → critic

    Best-attempt safeguard: tracks the lowest-flag-count attempt across all
    iterations and restores it at the end if a later attempt regressed.
    """
    yield from _run_node("reconciler", state)
    yield from _run_node("nepqa_mapper", state)
    yield from _run_node("drafter", state)
    yield from _run_node("critic", state)

    best = _snapshot_attempt(state)
    state["best_attempt"] = {"flag_count": best["flag_count"]}

    max_retries = state.get("max_retries", 2)

    while (state.get("critic_flags") or []) and state.get("retry_count", 0) < max_retries:
        yield from _run_node("patch_extractor", state)
        yield from _run_node("drafter", state)
        yield from _run_node("critic", state)

        new_flag_count = len(state.get("critic_flags") or [])
        if new_flag_count < best["flag_count"]:
            best = _snapshot_attempt(state)
            state["best_attempt"] = {"flag_count": best["flag_count"]}

    # Restore best attempt if the current state regressed
    final_flag_count = len(state.get("critic_flags") or [])
    if best["flag_count"] < final_flag_count:
        _restore_attempt(state, best)
        yield Event(
            "node_done",
            "best_attempt_restored",
            {
                "summary": (
                    f"Restored best attempt ({best['flag_count']} flags) "
                    f"after later iteration regressed to {final_flag_count} flags."
                ),
                "elapsed": 0.0,
            },
        )

    yield Event("phase_done", "phase_two")
