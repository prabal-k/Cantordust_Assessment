"""Node 7: field-by-field diff between PDF1 and PDF2 records.

Pure Python — no LLM call. Deterministic, testable, fast.

Severity logic:
  - CRITICAL: scalar values present on both sides and they DISAGREE on a model
    they nominally share.
  - WARNING: same family, one side missing while other has it.
  - INFO: DIFFERENT_FAMILY relationship downgrades all missing-side mismatches
    to info (apples vs oranges).
"""
from __future__ import annotations

from typing import Any, Optional

from src.schemas import (
    FieldClaim,
    MismatchEntry,
    MismatchSeverity,
    ProductRecord,
    VariantDecision,
    VariantRelationship,
)
from src.state import AgentState


# Scalar fields to compare. (field_path, dotted_attribute_path)
SCALAR_PATHS: list[tuple[str, str]] = [
    ("manufacturer", "manufacturer"),
    ("factory", "factory"),
    ("electrical.ac_voltage_v", "electrical.ac_voltage_v"),
    ("electrical.ac_frequency_hz", "electrical.ac_frequency_hz"),
    ("electrical.rated_power_w", "electrical.rated_power_w"),
    ("electrical.phase", "electrical.phase"),
    ("electrical.power_factor", "electrical.power_factor"),
    ("electrical.thd_pct", "electrical.thd_pct"),
    ("electrical.max_efficiency_pct", "electrical.max_efficiency_pct"),
    ("mechanical.ip_rating", "mechanical.ip_rating"),
    ("mechanical.weight_kg", "mechanical.weight_kg"),
    ("mechanical.topology", "mechanical.topology"),
    ("mechanical.protective_class", "mechanical.protective_class"),
    ("warranty_years", "warranty_years"),
]


def _resolve(record: ProductRecord, dotted: str) -> Optional[FieldClaim]:
    obj: Any = record
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if isinstance(obj, FieldClaim) else None


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def reconcile(
    p1: ProductRecord,
    p2: ProductRecord,
    decision: Optional[VariantDecision] = None,
) -> list[MismatchEntry]:
    is_different_family = (
        decision is not None
        and decision.relationship
        in (VariantRelationship.DIFFERENT_FAMILY, VariantRelationship.OEM_SAME_FACTORY)
    )

    out: list[MismatchEntry] = []

    for field_path, dotted in SCALAR_PATHS:
        fc1 = _resolve(p1, dotted)
        fc2 = _resolve(p2, dotted)

        if fc1 is None and fc2 is None:
            continue

        if fc1 is None or fc2 is None:
            # Missing on one side
            severity = (
                MismatchSeverity.INFO if is_different_family else MismatchSeverity.WARNING
            )
            out.append(
                MismatchEntry(
                    field_path=field_path,
                    pdf1_value=str(fc1.value) if fc1 else None,
                    pdf2_value=str(fc2.value) if fc2 else None,
                    severity=severity,
                    recommendation=(
                        f"Field present only in {'pdf1' if fc1 else 'pdf2'}. "
                        + (
                            "Expected — different product families."
                            if is_different_family
                            else "Ask factory to provide the missing value."
                        )
                    ),
                )
            )
            continue

        if _normalize(fc1.value) != _normalize(fc2.value):
            severity = (
                MismatchSeverity.INFO if is_different_family else MismatchSeverity.CRITICAL
            )
            out.append(
                MismatchEntry(
                    field_path=field_path,
                    pdf1_value=str(fc1.value),
                    pdf2_value=str(fc2.value),
                    severity=severity,
                    recommendation=(
                        "Different product families — divergence expected."
                        if is_different_family
                        else "Specs disagree across sources — request factory clarification before submission."
                    ),
                )
            )

    # Certifications: union vs intersection
    std1 = {
        c.standard.value for c in p1.certifications if isinstance(c.standard.value, str)
    }
    std2 = {
        c.standard.value for c in p2.certifications if isinstance(c.standard.value, str)
    }
    only_p1 = std1 - std2
    only_p2 = std2 - std1
    if only_p1 or only_p2:
        severity = (
            MismatchSeverity.INFO if is_different_family else MismatchSeverity.WARNING
        )
        out.append(
            MismatchEntry(
                field_path="certifications",
                pdf1_value=", ".join(sorted(only_p1)) or None,
                pdf2_value=", ".join(sorted(only_p2)) or None,
                severity=severity,
                recommendation=(
                    "Standards differ by document. Combined coverage may still be sufficient — "
                    "see NEPQA coverage matrix."
                ),
            )
        )

    return out


def reconciler_node(state: AgentState) -> AgentState:
    p1 = state["pdf1_record"]
    p2 = state["pdf2_record"]
    decision = state.get("variant_decision")
    return {"mismatches": reconcile(p1, p2, decision)}
