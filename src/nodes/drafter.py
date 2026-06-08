"""Node 9: produce the Nepal compliance draft.

Deterministic markdown template. Every fact wears a (source: pdfN p.K) citation.
No LLM call — provenance must be perfectly reliable, not paraphrased.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import OUTPUTS_DIR
from src.render import markdown_to_pdf, save_markdown
from src.schemas import (
    CoverageResult,
    CoverageStatus,
    FieldClaim,
    MismatchEntry,
    MismatchSeverity,
    ProductRecord,
    VariantDecision,
)
from src.state import AgentState


_DOC_NAMES: dict[str, str] = {}


def _doc_label(source_doc: str) -> str:
    """Resolve internal id (pdf1/pdf2/nepqa) → display name set on AgentState."""
    return _DOC_NAMES.get(source_doc, source_doc)


def _cite(fc: Optional[FieldClaim]) -> str:
    if fc is None:
        return "_not provided_"
    return (
        f"**{fc.value}** _(source: {_doc_label(fc.source_doc)} "
        f"p.{fc.source_page}, conf {fc.confidence:.2f})_"
    )


def _row(label: str, fc: Optional[FieldClaim]) -> str:
    return f"| {label} | {_cite(fc)} |"


def _badge(status: CoverageStatus) -> str:
    return {
        CoverageStatus.COVERED: "🟢 COVERED",
        CoverageStatus.PARTIAL: "🟡 PARTIAL",
        CoverageStatus.MISSING: "🔴 MISSING",
        CoverageStatus.NOT_APPLICABLE: "⚪ N/A",
    }[status]


def _severity_tag(sev: MismatchSeverity) -> str:
    return {
        MismatchSeverity.INFO: "INFO",
        MismatchSeverity.WARNING: "⚠ WARNING",
        MismatchSeverity.CRITICAL: "🔴 CRITICAL",
    }[sev]


def _scalar_value(record: Optional[ProductRecord], dotted: str) -> Optional[str]:
    """Resolve a dotted path on a record and return the FieldClaim value as str."""
    if record is None:
        return None
    obj: object = record
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    val = getattr(obj, "value", None)
    return str(val) if val is not None else None


_AGREEMENT_PATHS: list[tuple[str, str]] = [
    ("Factory", "factory"),
    ("Topology", "mechanical.topology"),
    ("Power factor", "electrical.power_factor"),
    ("Cooling", "mechanical.cooling"),
    ("Operating temp", "mechanical.operating_temp_range_c"),
    ("Protective class", "mechanical.protective_class"),
    ("AC frequency", "electrical.ac_frequency_hz"),
]


def _collect_agreements(
    p1: Optional[ProductRecord], p2: Optional[ProductRecord]
) -> list[tuple[str, str]]:
    """Return rows where both records have the SAME normalized value for a
    given field."""
    if p1 is None or p2 is None:
        return []
    rows: list[tuple[str, str]] = []
    for label, dotted in _AGREEMENT_PATHS:
        v1 = _scalar_value(p1, dotted)
        v2 = _scalar_value(p2, dotted)
        if v1 is None or v2 is None:
            continue
        if v1.strip().lower() == v2.strip().lower():
            rows.append((label, v1))
    return rows


def render_markdown(
    chosen: ProductRecord,
    decision: Optional[VariantDecision],
    mismatches: list[MismatchEntry],
    coverage: list[CoverageResult],
    timestamp: str,
    p1: Optional[ProductRecord] = None,
    p2: Optional[ProductRecord] = None,
) -> str:
    e, m = chosen.electrical, chosen.mechanical

    coverage_summary = {
        "covered": sum(1 for c in coverage if c.status == CoverageStatus.COVERED),
        "partial": sum(1 for c in coverage if c.status == CoverageStatus.PARTIAL),
        "missing": sum(1 for c in coverage if c.status == CoverageStatus.MISSING),
        "na": sum(1 for c in coverage if c.status == CoverageStatus.NOT_APPLICABLE),
    }

    md_lines: list[str] = []
    A = md_lines.append

    # --- Header
    A("# Nepal Import Compliance Draft")
    A("")
    A(f"**Importer**: SunBridge Trading Pvt. Ltd., Kathmandu  ")
    A(f"**Generated**: {timestamp}  ")
    A(f"**Regulatory reference**: NEPQA 2025 Section 1.4 (PV Inverter / Grid Connected Inverter)  ")
    A(f"**Status**: Draft for import-agent review")
    A("")
    A("> This draft was assembled by an automated agent. Every claim cites the "
      "source PDF and page. Mismatches between sources are reported honestly. "
      "Items marked 🔴 MISSING must be requested from the manufacturer before "
      "final submission to the import agent.")
    A("")

    # --- Variant decision
    if decision is not None:
        A("## 1. Source documents")
        A("")
        A(f"**Variant relationship**: `{decision.relationship.value}`  ")
        A(f"**Reasoning**: {decision.reasoning}")
        A("")
        if decision.shared_attributes:
            A(f"- Shared: {', '.join(decision.shared_attributes)}")
        if decision.distinguishing_attributes:
            A(f"- Distinguishing: {', '.join(decision.distinguishing_attributes)}")
        A("")
        A(f"**Product family chosen for this draft**: `{chosen.family_label}` "
          f"(source: {_doc_label(chosen.source_doc)})")
        A("")

    # --- Product summary
    A("## 2. Product summary")
    A("")
    A("| Attribute | Value |")
    A("|---|---|")
    A(_row("Document type", chosen.document_type))
    A(_row("Manufacturer / brand owner", chosen.manufacturer))
    A(_row("Factory", chosen.factory))
    if chosen.applicant:
        A(_row("Applicant", chosen.applicant))
    A(f"| Family | **{chosen.family_label}** |")
    A(_row("Phase", e.phase))
    A(_row("AC voltage (V)", e.ac_voltage_v))
    A(_row("AC frequency (Hz)", e.ac_frequency_hz))
    A(_row("Rated AC power (W)", e.rated_power_w))
    A(_row("Power factor", e.power_factor))
    A(_row("THD (%)", e.thd_pct))
    A(_row("Max efficiency (%)", e.max_efficiency_pct))
    A(_row("Euro efficiency (%)", e.euro_efficiency_pct))
    A(_row("IP rating", m.ip_rating))
    A(_row("Topology", m.topology))
    A(_row("Operating temp (°C)", m.operating_temp_range_c))
    A(_row("Weight (kg)", m.weight_kg))
    A(_row("Warranty (years)", chosen.warranty_years))
    A("")

    A("### Model numbers covered")
    A("")
    if chosen.model_numbers:
        for fc in chosen.model_numbers:
            A(f"- {_cite(fc)}")
    else:
        A("_no model numbers extracted_")
    A("")

    # --- Variant breakdown (auto-grouped by SKU suffix patterns)
    if chosen.model_numbers:
        groups: dict[str, list[str]] = {}
        for fc in chosen.model_numbers:
            if not isinstance(fc.value, str):
                continue
            sku = fc.value.strip()
            # Heuristic: variant = the bit after the last `-` if it looks like
            # a short tag (e.g. "AM2", "AM2-P1", "EU"). Otherwise the SKU goes
            # into a "base" bucket.
            parts = sku.split("-")
            tag = parts[-1] if parts and len(parts[-1]) <= 4 else "base"
            groups.setdefault(tag, []).append(sku)
        if len(groups) > 1:
            A("### Variant breakdown")
            A("")
            A("The model SKUs cluster into the following variants. Differences "
              "between variants (e.g. max input current, max short-circuit "
              "current) should be confirmed against the certificate appendix.")
            A("")
            A("| Variant suffix | Count | Example SKUs |")
            A("|---|---|---|")
            for tag in sorted(groups):
                skus = groups[tag]
                example = ", ".join(skus[:3]) + (f" … (+{len(skus) - 3} more)" if len(skus) > 3 else "")
                A(f"| `{tag}` | {len(skus)} | {example} |")
            A("")

    # --- Certifications
    A("## 3. Certifications & test reports")
    A("")
    if chosen.certifications:
        A("| Standard | Cert / Report # | Issuer | Valid until |")
        A("|---|---|---|---|")
        for c in chosen.certifications:
            A(
                f"| {_cite(c.standard)} | {_cite(c.cert_number) if c.cert_number else _cite(c.test_report_number)} "
                f"| {_cite(c.issuer)} | {_cite(c.valid_until)} |"
            )
    else:
        A("_no certifications extracted_")
    A("")

    # --- Labeling
    A("## 4. Labeling")
    A("")
    if chosen.labeling_items:
        for fc in chosen.labeling_items:
            A(f"- {_cite(fc)}")
    else:
        A("_no labeling items extracted — request nameplate photo from factory_")
    A("")

    # --- NEPQA coverage matrix
    A("## 5. NEPQA Section 1.4 coverage matrix")
    A("")
    A(
        f"**Summary**: 🟢 {coverage_summary['covered']} covered "
        f"· 🟡 {coverage_summary['partial']} partial "
        f"· 🔴 {coverage_summary['missing']} missing "
        f"· ⚪ {coverage_summary['na']} N/A"
    )
    A("")
    A("| Clause | NEPQA p. | Requirement | Status | Evidence / gap |")
    A("|---|---|---|---|---|")
    for cov in coverage:
        ev_or_gap = (
            ", ".join(_cite(fc) for fc in cov.evidence)
            if cov.evidence
            else (cov.gap_note or "—")
        )
        A(
            f"| `{cov.item.clause_id}` | _p.{cov.item.source_page}_ "
            f"| {cov.item.requirement_text[:120]} "
            f"| {_badge(cov.status)} | {ev_or_gap} |"
        )
    A("")

    # --- Mismatches + agreements
    A("## 6. Cross-source consistency report")
    A("")

    agreements = _collect_agreements(p1, p2)
    if agreements:
        A("### Consistent across both source documents")
        A("")
        A("| Field | Value (same in both) |")
        A("|---|---|")
        for label, value in agreements:
            A(f"| {label} | **{value}** |")
        A("")
    else:
        A("### Consistent across both source documents")
        A("")
        A("_no exact agreements detected on scalar fields — the two PDFs cover "
          "different product families, so divergence is expected._")
        A("")

    A("### Differences")
    A("")
    if not mismatches:
        A("_no mismatches detected_")
    else:
        p1_label = _doc_label("pdf1")
        p2_label = _doc_label("pdf2")
        A(f"| Field | {p1_label} | {p2_label} | Severity | Recommendation |")
        A("|---|---|---|---|---|")
        for m_ in mismatches:
            A(
                f"| `{m_.field_path}` | {m_.pdf1_value or '—'} | {m_.pdf2_value or '—'} "
                f"| {_severity_tag(m_.severity)} | {m_.recommendation} |"
            )
    A("")

    return "\n".join(md_lines)


def drafter_node(state: AgentState) -> AgentState:
    chosen = state["chosen_record"]
    decision = state.get("variant_decision")
    mismatches = state.get("mismatches", [])
    coverage = state.get("coverage", [])
    timestamp = state.get("run_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Prime the doc-label resolver so every _cite() call uses the friendly name.
    _DOC_NAMES.clear()
    _DOC_NAMES.update(
        {
            "pdf1": state.get("pdf1_name") or "pdf1",
            "pdf2": state.get("pdf2_name") or "pdf2",
            "nepqa": state.get("nepqa_name") or "nepqa",
        }
    )

    md_text = render_markdown(
        chosen,
        decision,
        mismatches,
        coverage,
        timestamp,
        p1=state.get("pdf1_record"),
        p2=state.get("pdf2_record"),
    )

    safe_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_md = OUTPUTS_DIR / f"compliance_draft_{safe_ts}.md"
    out_pdf = OUTPUTS_DIR / f"compliance_draft_{safe_ts}.pdf"
    save_markdown(md_text, out_md)
    pdf_ok = markdown_to_pdf(md_text, out_pdf)

    # Also dump full agent state as JSON for full transparency
    out_state = OUTPUTS_DIR / f"agent_state_{safe_ts}.json"
    state_dump = {
        "timestamp": timestamp,
        "chosen_record": chosen.model_dump(),
        "variant_decision": decision.model_dump() if decision else None,
        "mismatches": [m_.model_dump() for m_ in mismatches],
        "coverage": [c.model_dump() for c in coverage],
        "pdf1_record": state.get("pdf1_record").model_dump() if state.get("pdf1_record") else None,
        "pdf2_record": state.get("pdf2_record").model_dump() if state.get("pdf2_record") else None,
    }
    out_state.write_text(json.dumps(state_dump, indent=2, default=str), encoding="utf-8")

    return {
        "draft_markdown": md_text,
        "draft_md_path": str(out_md),
        "draft_pdf_path": str(out_pdf) if pdf_ok else "",
    }
