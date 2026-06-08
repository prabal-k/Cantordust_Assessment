"""Post-extraction normalizers.

The LLM sometimes mislabels `family_label` and `electrical.phase` when the
voltage field is written ambiguously (e.g. `3L/N/PE 230/400V` parsed as just
"230"). We recompute these two fields deterministically from the rated power
and AC voltage so the downstream variant detector + UI don't carry the
mislabel.

Pure Python — no LLM call. Cheap, testable, idempotent.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from src.schemas import FieldClaim, ProductRecord


_THREE_PHASE_MARKERS = (
    "3l/n/pe",
    "3l-n-pe",
    "3 phase",
    "three phase",
    "3-phase",
    "three-phase",
    "400v",
    "230/400",
)

_SINGLE_PHASE_MARKERS = (
    "single phase",
    "1-phase",
    "single-phase",
)


def _voltage_text(record: ProductRecord) -> str:
    fc = record.electrical.ac_voltage_v
    if fc is None or not isinstance(fc.value, str):
        return ""
    return fc.value.lower()


def _rated_power_kw(record: ProductRecord) -> Optional[float]:
    fc = record.electrical.rated_power_w
    if fc is None or not isinstance(fc.value, str):
        return None
    # Parse first number; tolerate "3000", "3.0kW", "300, 500, ...", "3000 W"
    m = re.search(r"(\d+(?:\.\d+)?)", fc.value)
    if not m:
        return None
    n = float(m.group(1))
    # Treat 3-digit and higher as W; <100 we assume kW
    if n >= 100:
        return n / 1000.0
    return n


def _infer_phase(record: ProductRecord) -> Optional[Literal["single", "three"]]:
    # Trust existing extraction first when unambiguous
    phase_fc = record.electrical.phase
    if phase_fc is not None and isinstance(phase_fc.value, str):
        p = phase_fc.value.strip().lower()
        if "three" in p or p == "3":
            return "three"
        if "single" in p or p == "1":
            return "single"

    v = _voltage_text(record)
    if any(m in v for m in _THREE_PHASE_MARKERS):
        return "three"
    if any(m in v for m in _SINGLE_PHASE_MARKERS):
        return "single"
    return None


def _infer_family(phase: Optional[str], power_kw: Optional[float]) -> str:
    """Mirror the prompt's decision rules in code.

    (a) three-phase + power > 2 kW       → three_phase_string_inverter
    (b) single-phase + all power ≤ 2 kW  → microinverter
    (c) single-phase + power > 2 kW      → single_phase_string_inverter
    """
    if phase == "three":
        return "three_phase_string_inverter"
    if phase == "single":
        if power_kw is None:
            return "single_phase_string_inverter"
        return "microinverter" if power_kw <= 2.0 else "single_phase_string_inverter"
    return "unknown"


def normalize_record(record: ProductRecord, expected_source: str) -> ProductRecord:
    """Force source_doc + recompute family_label/phase to match the data.

    Keeps every FieldClaim intact; only mutates the small fixed fields.
    """
    updates = {}

    if record.source_doc != expected_source:
        updates["source_doc"] = expected_source

    inferred_phase = _infer_phase(record)
    inferred_power = _rated_power_kw(record)
    inferred_family = _infer_family(inferred_phase, inferred_power)

    # Patch family_label only if the LLM's guess disagrees with the data.
    if inferred_family != "unknown" and record.family_label != inferred_family:
        updates["family_label"] = inferred_family

    # Patch phase FieldClaim if we have a confident phase from voltage markers
    # but the existing phase field is missing or wrong.
    if inferred_phase is not None:
        current_phase = record.electrical.phase
        if current_phase is None or (
            isinstance(current_phase.value, str)
            and inferred_phase not in current_phase.value.lower()
        ):
            new_phase = FieldClaim(
                value=inferred_phase,
                source_doc=expected_source,  # type: ignore[arg-type]
                source_page=current_phase.source_page if current_phase else 1,
                confidence=0.85,
                notes="Inferred from AC voltage markers in post-extraction normalizer",
            )
            new_electrical = record.electrical.model_copy(
                update={"phase": new_phase}
            )
            updates["electrical"] = new_electrical

    if updates:
        record = record.model_copy(update=updates)
    return record
