"""Node 8: map chosen product against NEPQA Section 1.4 requirements.

Per item, look for evidence in the chosen ProductRecord and classify:
  COVERED: clear evidence found.
  PARTIAL: related evidence found but doesn't fully satisfy.
  MISSING: no evidence.
  NOT_APPLICABLE: requirement irrelevant for this product family (rare).

Pure Python heuristic — no LLM call. Keeps the node deterministic and testable.
The drafter node uses the natural-language fields to write prose; the mapper
just needs to be reliable about coverage classification.
"""
from __future__ import annotations

import re
from typing import Optional

from src.schemas import (
    CoverageResult,
    CoverageStatus,
    FieldClaim,
    NEPQAItem,
    NEPQAItemType,
    ProductRecord,
)
from src.state import AgentState


# Map common NEPQA technical thresholds to ProductRecord attribute paths.
TECH_KEYWORDS = {
    "voltage": "electrical.ac_voltage_v",
    "frequency": "electrical.ac_frequency_hz",
    "efficiency": "electrical.max_efficiency_pct",
    "euro": "electrical.euro_efficiency_pct",
    "mppt": "electrical.mppt_efficiency_pct",
    "thd": "electrical.thd_pct",
    "power factor": "electrical.power_factor",
    "ip65": "mechanical.ip_rating",
    "ip protection": "mechanical.ip_rating",
    "ingress": "mechanical.ip_rating",
    "warranty": "warranty_years",
    "cooling": "mechanical.cooling",
    # "transformer" intentionally NOT here: it appears in many no-load-loss
    # clauses where the real requirement is loss-%, not topology presence.
    # Map it explicitly only when the clause is about topology itself.
    "topology": "mechanical.topology",
}


# Evidence with FieldClaim.confidence below this threshold gets demoted from
# COVERED → PARTIAL. Keyword match alone is not enough; the LLM must have
# extracted the value confidently.
CONFIDENCE_FLOOR_FOR_COVERED = 0.80


def _max_conf(claims: list[FieldClaim]) -> float:
    return max((c.confidence for c in claims), default=0.0)


def _resolve(record: ProductRecord, dotted: str) -> Optional[FieldClaim]:
    obj = record
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if isinstance(obj, FieldClaim) else None


def _all_cert_standards(record: ProductRecord) -> list[FieldClaim]:
    return [c.standard for c in record.certifications]


def _normalize_std(s: str) -> str:
    # Drop year + part suffixes for soft matching ("IEC 62109-1:2010" → "iec 62109")
    s = s.lower()
    s = re.sub(r":\s*\d{4}", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _check_document_item(item: NEPQAItem, record: ProductRecord) -> CoverageResult:
    expected = (item.expected_value or item.requirement_text or "").strip()
    if not expected:
        return CoverageResult(item=item, status=CoverageStatus.MISSING)

    expected_norm = _normalize_std(expected)
    matches: list[FieldClaim] = []
    for std in _all_cert_standards(record):
        if not isinstance(std.value, str):
            continue
        std_norm = _normalize_std(std.value)
        # exact match OR substring match on the IEC number
        core_match = re.search(r"iec\s*\d+(?:-\d+)?", expected_norm)
        if core_match and core_match.group(0) in std_norm:
            matches.append(std)

    if matches:
        # Demote to PARTIAL if best evidence confidence is low.
        if _max_conf(matches) < 0.80:
            return CoverageResult(
                item=item,
                status=CoverageStatus.PARTIAL,
                evidence=matches,
                gap_note=(
                    f"Standard match confidence {_max_conf(matches):.2f} below "
                    "0.80 threshold; verify cert against the source PDF."
                ),
            )
        return CoverageResult(item=item, status=CoverageStatus.COVERED, evidence=matches)
    return CoverageResult(
        item=item,
        status=CoverageStatus.MISSING,
        gap_note=f"No certification matching {expected} found in extracted record.",
    )


def _check_technical_item(item: NEPQAItem, record: ProductRecord) -> CoverageResult:
    req_text = (item.requirement_text or "").lower()
    matched_attr: Optional[str] = None
    for kw, attr in TECH_KEYWORDS.items():
        if kw in req_text:
            matched_attr = attr
            break
    if matched_attr is None:
        return CoverageResult(
            item=item,
            status=CoverageStatus.PARTIAL,
            gap_note="Technical clause not auto-mapped; manual review needed.",
        )
    fc = _resolve(record, matched_attr)
    if fc is None:
        return CoverageResult(
            item=item,
            status=CoverageStatus.MISSING,
            gap_note=f"Value for {matched_attr} not present in extracted record. Ask factory.",
        )
    # Demote to PARTIAL if evidence confidence is low — keyword match alone is
    # not enough to claim COVERED.
    if fc.confidence < CONFIDENCE_FLOOR_FOR_COVERED:
        return CoverageResult(
            item=item,
            status=CoverageStatus.PARTIAL,
            evidence=[fc],
            gap_note=(
                f"Evidence found but confidence {fc.confidence:.2f} below "
                f"{CONFIDENCE_FLOOR_FOR_COVERED} threshold; manual review needed."
            ),
        )
    return CoverageResult(item=item, status=CoverageStatus.COVERED, evidence=[fc])


def _check_label_item(item: NEPQAItem, record: ProductRecord) -> CoverageResult:
    """Label-content requirements need a nameplate photo to be honestly covered.

    Token-overlap matches against table headers in the certificate are too weak
    to mark COVERED — the certificate is not the nameplate. We always report
    PARTIAL with a "request nameplate photo" gap note when token matches exist,
    MISSING otherwise.
    """
    req_text = (item.requirement_text or "").lower()
    matches: list[FieldClaim] = []
    for lbl in record.labeling_items:
        if not isinstance(lbl.value, str):
            continue
        tokens = [t for t in re.split(r"[\s,;:/]+", req_text) if len(t) > 3]
        if any(t in lbl.value.lower() for t in tokens):
            matches.append(lbl)

    high_conf_matches = [m for m in matches if m.confidence >= CONFIDENCE_FLOOR_FOR_COVERED]

    if high_conf_matches:
        return CoverageResult(
            item=item,
            status=CoverageStatus.COVERED,
            evidence=high_conf_matches,
        )
    if matches:
        return CoverageResult(
            item=item,
            status=CoverageStatus.PARTIAL,
            evidence=matches,
            gap_note=(
                "Weak match against certificate text; cannot confirm without a "
                "nameplate photo from the factory."
            ),
        )
    return CoverageResult(
        item=item,
        status=CoverageStatus.PARTIAL,
        gap_note="No label-photo evidence in extracted record. Request nameplate photo from factory.",
    )


def map_coverage(
    record: ProductRecord, checklist: list[NEPQAItem]
) -> list[CoverageResult]:
    results: list[CoverageResult] = []
    for item in checklist:
        if item.item_type == NEPQAItemType.DOCUMENT:
            results.append(_check_document_item(item, record))
        elif item.item_type == NEPQAItemType.TECHNICAL:
            results.append(_check_technical_item(item, record))
        elif item.item_type == NEPQAItemType.LABEL:
            results.append(_check_label_item(item, record))
        else:  # GENERAL
            results.append(
                CoverageResult(item=item, status=CoverageStatus.NOT_APPLICABLE)
            )
    return results


def nepqa_mapper_node(state: AgentState) -> AgentState:
    chosen = state["chosen_record"]
    checklist = state["nepqa_checklist"]
    return {"coverage": map_coverage(chosen, checklist)}
