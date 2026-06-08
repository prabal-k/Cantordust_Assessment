"""ReAct tools for variant_detector.

`build_variant_tools(p1, p2, sink)` returns the 5 LangChain tools bound to the
two ProductRecords plus a mutable `sink` list. The agent calls them in any
order; `commit_decision` is terminal and writes its args into the sink for the
node to read after the agent finishes.

The tools are pure functions over the two records — no LLM, no I/O. This makes
them deterministic + unit-testable without mocking.
"""
from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool, tool

from src.schemas import FieldClaim, ProductRecord


# --- Helpers --------------------------------------------------------------

def _resolve_field(record: ProductRecord, dotted: str) -> Any:
    """Walk a dotted attribute path; return the leaf object or None."""
    obj: Any = record
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _claim_value(obj: Any) -> str | None:
    """Unwrap FieldClaim → str; pass plain scalars through."""
    if obj is None:
        return None
    if isinstance(obj, FieldClaim):
        return str(obj.value)
    if isinstance(obj, list):
        return ", ".join(
            str(item.value) if isinstance(item, FieldClaim) else str(item)
            for item in obj
        )
    return str(obj)


def _all_cert_standards(record: ProductRecord) -> list[str]:
    return [
        str(c.standard.value)
        for c in record.certifications
        if isinstance(c.standard.value, str)
    ]


# --- Tool factory ---------------------------------------------------------

def build_variant_tools(
    p1: ProductRecord,
    p2: ProductRecord,
    sink: list[dict],
) -> list[BaseTool]:
    """Return the 5 tools bound to these two records and to a shared sink for
    `commit_decision` output."""

    @tool
    def compare_field(field_path: str) -> dict:
        """Compare one field across PDF1 and PDF2 by dotted path.

        field_path: dotted attribute path on ProductRecord. Examples:
            "electrical.ac_voltage_v", "electrical.phase",
            "electrical.rated_power_w", "mechanical.ip_rating",
            "mechanical.topology", "manufacturer", "factory",
            "family_label", "warranty_years".
        Returns: {"pdf1_value": str|None, "pdf2_value": str|None, "match": bool,
                   "path": str}.
        """
        v1 = _claim_value(_resolve_field(p1, field_path))
        v2 = _claim_value(_resolve_field(p2, field_path))
        match = (
            v1 is not None
            and v2 is not None
            and v1.strip().lower() == v2.strip().lower()
        )
        return {"pdf1_value": v1, "pdf2_value": v2, "match": match, "path": field_path}

    @tool
    def get_models(pdf: Literal["pdf1", "pdf2"]) -> list[str]:
        """Return the list of model SKU strings declared in the named record."""
        rec = p1 if pdf == "pdf1" else p2
        return [
            str(fc.value)
            for fc in rec.model_numbers
            if isinstance(fc.value, str)
        ]

    @tool
    def check_factory_match() -> dict:
        """Return {"pdf1_factory": str|None, "pdf2_factory": str|None, "match": bool}."""
        f1 = _claim_value(p1.factory)
        f2 = _claim_value(p2.factory)
        match = (
            f1 is not None
            and f2 is not None
            and f1.strip().lower() == f2.strip().lower()
        )
        return {"pdf1_factory": f1, "pdf2_factory": f2, "match": match}

    @tool
    def check_certifications_overlap() -> dict:
        """Return {"shared": [...], "only_pdf1": [...], "only_pdf2": [...]} of
        certification standard strings."""
        s1 = set(_all_cert_standards(p1))
        s2 = set(_all_cert_standards(p2))
        return {
            "shared": sorted(s1 & s2),
            "only_pdf1": sorted(s1 - s2),
            "only_pdf2": sorted(s2 - s1),
        }

    @tool
    def commit_decision(
        relationship: Literal[
            "SAME_PRODUCT", "VARIANT", "DIFFERENT_FAMILY", "OEM_SAME_FACTORY"
        ],
        reasoning: str,
        shared_attributes: list[str],
        distinguishing_attributes: list[str],
        requires_human_choice: bool,
    ) -> str:
        """Record the FINAL verdict. Call this exactly once at the end.

        relationship: one of SAME_PRODUCT, VARIANT, DIFFERENT_FAMILY,
            OEM_SAME_FACTORY (uppercase, case-sensitive).
        requires_human_choice: True when DIFFERENT_FAMILY or OEM_SAME_FACTORY.
        Returns the literal string 'committed' on success.
        """
        sink.append(
            {
                "relationship": relationship,
                "reasoning": reasoning,
                "shared_attributes": list(shared_attributes),
                "distinguishing_attributes": list(distinguishing_attributes),
                "requires_human_choice": bool(requires_human_choice),
            }
        )
        return "committed"

    return [
        compare_field,
        get_models,
        check_factory_match,
        check_certifications_overlap,
        commit_decision,
    ]


def build_decision_from_sink(sink: list[dict]):
    """Construct a VariantDecision from the last commit in the sink."""
    from src.schemas import VariantDecision, VariantRelationship

    if not sink:
        return None
    payload = sink[-1]
    rel = payload["relationship"]
    if isinstance(rel, str):
        rel = rel.upper()
    return VariantDecision(
        relationship=VariantRelationship(rel),
        reasoning=payload.get("reasoning", ""),
        shared_attributes=payload.get("shared_attributes", []) or [],
        distinguishing_attributes=payload.get("distinguishing_attributes", []) or [],
        requires_human_choice=payload.get("requires_human_choice", False),
    )


def format_tool_trace(messages: list) -> str:
    """Convert the agent's final messages list into a readable text trace.

    Format:  → tool_name(arg=...)
             ← <result truncated to 240 chars>
    """
    import json as _json

    lines: list[str] = []
    last_calls: dict[str, str] = {}
    for msg in messages:
        # AIMessage with tool_calls
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            tcid = (
                tc.get("id")
                if isinstance(tc, dict)
                else getattr(tc, "id", None)
            )
            try:
                args_str = _json.dumps(args, ensure_ascii=False)
            except Exception:
                args_str = str(args)
            lines.append(f"→ {name}({args_str})")
            if tcid:
                last_calls[tcid] = name
        # ToolMessage with the response
        if msg.__class__.__name__ == "ToolMessage":
            content = getattr(msg, "content", "")
            preview = str(content)
            if len(preview) > 240:
                preview = preview[:240] + "…"
            lines.append(f"  ← {preview}")
    return "\n".join(lines)
