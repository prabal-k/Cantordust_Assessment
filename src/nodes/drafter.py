"""Build the Nepal compliance draft.

Deterministic markdown template + LLM prose blocks (cover note, methodology,
gap narrative, mismatch framing). Called twice per run: first pass for the
critic, second pass after the critic produces ask_factory_list so §8 is
populated in the final draft.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional

from src.config import OUTPUTS_DIR
from src.llm import invoke_structured
from src.prompts import DRAFTER_PROSE_SYSTEM, drafter_prose_user
from src.render import markdown_to_pdf, save_markdown
from src.schemas import (
    CoverageResult,
    CoverageStatus,
    CriticFlag,
    DrafterProse,
    FieldClaim,
    MismatchEntry,
    MismatchSeverity,
    ProductRecord,
    VariantDecision,
)
from src.state import AgentState


_DOC_NAMES: dict[str, str] = {}


def _doc_label(source_doc: str) -> str:
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


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

_CITATION_PATTERNS = [
    re.compile(r"\(source:[^)]*\)", re.IGNORECASE),
    re.compile(r"\bp\.\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bconf\s+0?\.\d+\b", re.IGNORECASE),
]

# Small numbers the LLM may use grammatically ("two PDFs") + NEPQA section ref
# even when not present in the structured input.
_STRUCTURAL_DIGITS = {str(n) for n in range(0, 13)} | {"1.4", "2025"}

_PROSE_FIELDS = ("cover_note", "methodology_note", "gap_narrative", "mismatch_framing")


def _allowed_digits(
    chosen: ProductRecord,
    coverage: list[CoverageResult],
    mismatches: list[MismatchEntry],
    decision: Optional[VariantDecision],
) -> set[str]:
    blob = json.dumps(
        {
            "chosen": chosen.model_dump(),
            "coverage": [c.model_dump() for c in coverage],
            "mismatches": [m_.model_dump() for m_ in mismatches],
            "decision": decision.model_dump() if decision else None,
        },
        default=str,
    )
    return set(_NUM_RE.findall(blob)) | _STRUCTURAL_DIGITS


def _sanitize_prose(text: str) -> str:
    out = text
    for pat in _CITATION_PATTERNS:
        out = pat.sub("", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def _orphan_digits(text: str, allowed: set[str]) -> list[str]:
    return [tok for tok in _NUM_RE.findall(text) if tok not in allowed]


def _build_prose_input(
    chosen: ProductRecord,
    decision: Optional[VariantDecision],
    mismatches: list[MismatchEntry],
    coverage: list[CoverageResult],
) -> str:
    coverage_counts = {
        "covered": sum(1 for c in coverage if c.status == CoverageStatus.COVERED),
        "partial": sum(1 for c in coverage if c.status == CoverageStatus.PARTIAL),
        "missing": sum(1 for c in coverage if c.status == CoverageStatus.MISSING),
        "not_applicable": sum(
            1 for c in coverage if c.status == CoverageStatus.NOT_APPLICABLE
        ),
    }
    payload = {
        "product": {
            "family_label": chosen.family_label,
            "manufacturer": chosen.manufacturer.value if chosen.manufacturer else None,
            "factory": chosen.factory.value if chosen.factory else None,
            "model_count": len(chosen.model_numbers),
            "certification_count": len(chosen.certifications),
            "chosen_from": chosen.source_doc,
        },
        "coverage_counts": coverage_counts,
        "partial_or_missing_clauses": [
            {
                "clause_id": c.item.clause_id,
                "requirement": c.item.requirement_text[:160],
                "status": c.status.value,
                "gap_note": c.gap_note,
            }
            for c in coverage
            if c.status in (CoverageStatus.PARTIAL, CoverageStatus.MISSING)
        ],
        "mismatches": [
            {
                "field": m_.field_path,
                "pdf1": m_.pdf1_value,
                "pdf2": m_.pdf2_value,
                "severity": m_.severity.value,
                "recommendation": m_.recommendation,
            }
            for m_ in mismatches
        ],
        "variant_relationship": decision.relationship.value if decision else None,
    }
    return json.dumps(payload, separators=(",", ":"), default=str)


def _empty_prose() -> DrafterProse:
    return DrafterProse(
        cover_note="", methodology_note="", gap_narrative="", mismatch_framing=""
    )


def _call_prose_llm(
    chosen: ProductRecord,
    decision: Optional[VariantDecision],
    mismatches: list[MismatchEntry],
    coverage: list[CoverageResult],
    max_attempts: int = 2,
) -> DrafterProse:
    structured = _build_prose_input(chosen, decision, mismatches, coverage)
    allowed = _allowed_digits(chosen, coverage, mismatches, decision)

    user = drafter_prose_user(structured)
    system = DRAFTER_PROSE_SYSTEM

    for _ in range(max_attempts):
        try:
            prose = invoke_structured(
                DrafterProse, system, user, temperature=0.0
            )
        except Exception:
            return _empty_prose()

        cleaned_kwargs = {
            f: _sanitize_prose(getattr(prose, f)) for f in _PROSE_FIELDS
        }
        cleaned = DrafterProse(**cleaned_kwargs)

        orphans: list[str] = []
        for f in _PROSE_FIELDS:
            orphans.extend(_orphan_digits(getattr(cleaned, f), allowed))

        if not orphans:
            return cleaned

        user = drafter_prose_user(structured) + (
            f"\n\nPREVIOUS ATTEMPT REJECTED: it contained these numbers not "
            f"present in the input: {sorted(set(orphans))}. Do NOT write any "
            "number that is not in the STRUCTURED INPUT above."
        )

    return _empty_prose()

_FALLBACK_COVER_NOTE = (
    "This draft has been prepared for the Nepal import agent reviewing a "
    "shipment of grid-tied solar inverters from China on behalf of SunBridge "
    "Trading Pvt. Ltd. (Kathmandu). It pulls together product details, "
    "manufacturer information, test certifications, and labeling from the "
    "two manufacturer PDFs supplied, and aligns them against the NEPQA 2025 "
    "indicative checklist. Cross-source mismatches are reported in §6 and "
    "items still to be confirmed with the factory are listed in §8."
)

_FALLBACK_METHODOLOGY = (
    "Two manufacturer PDFs were read page by page; a typed ProductRecord was "
    "extracted from each, with every field carrying a (source: pdfN p.K) "
    "citation. NEPQA 2025 Section 1.4 was parsed once and used as an "
    "indicative import-side reference, not as a filing form to copy "
    "section-by-section. A variant-relationship pass classified how the two "
    "PDFs relate; where they describe different product families, only one "
    "was carried forward to this draft (see §1). A self-review pass then "
    "re-read this draft against NEPQA source pages and produced the "
    "ask-the-factory list in §8."
)

def render_markdown(
    chosen: ProductRecord,
    decision: Optional[VariantDecision],
    mismatches: list[MismatchEntry],
    coverage: list[CoverageResult],
    timestamp: str,
    p1: Optional[ProductRecord] = None,
    p2: Optional[ProductRecord] = None,
    prose: Optional[DrafterProse] = None,
    ask_factory: Optional[list[str]] = None,
    critic_flags: Optional[list[CriticFlag]] = None,
    retry_count: int = 0,
    max_retries: int = 0,
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

    A("# Nepal Import Compliance Draft — Grid-Tied Solar Inverter")
    A("")
    A("| Field | Value |")
    A("|---|---|")
    A("| **Importer** | SunBridge Trading Pvt. Ltd., Kathmandu |")
    A("| **Recipient** | Nepal import agent (for review) |")
    A("| **Document type** | Draft compliance file — for agent review, not a final filing |")
    A("| **Regulatory reference** | NEPQA 2025 §1.4 (PV Inverter / Grid Connected Inverter) — used as indicative import-side guideline |")
    A(f"| **Generated** | {timestamp} |")
    A("| **Status** | Draft for import-agent review |")
    A("| **Prepared by** | Automated compliance-drafting agent (LangGraph orchestrated; see §9 for methodology) |")
    A("")
    A("---")
    A("")
    A("## Cover note for the import agent")
    A("")
    if prose is not None and prose.cover_note:
        A(prose.cover_note)
    else:
        A(_FALLBACK_COVER_NOTE)
    A("")
    A("## How this draft was assembled")
    A("")
    if prose is not None and prose.methodology_note:
        A(prose.methodology_note)
    else:
        A(_FALLBACK_METHODOLOGY)
    A("")
    A("## 1. Product and variant identification")
    A("")
    A(f"**Product family chosen for this draft**: `{chosen.family_label}` "
      f"(extracted from {_doc_label(chosen.source_doc)})")
    A("")
    if decision is not None:
        A(f"**Variant relationship between the two manufacturer PDFs**: "
          f"`{decision.relationship.value}`")
        A("")
        A(f"**Reasoning**: {decision.reasoning}")
        A("")
        if decision.shared_attributes:
            A(f"- **Shared attributes**: {', '.join(decision.shared_attributes)}")
        if decision.distinguishing_attributes:
            A(f"- **Distinguishing attributes**: {', '.join(decision.distinguishing_attributes)}")
        A("")
    else:
        A("_Variant relationship not classified._")
        A("")
    A("## 2. Manufacturer and factory")
    A("")
    A("| Attribute | Value |")
    A("|---|---|")
    A(_row("Manufacturer / brand owner", chosen.manufacturer))
    A(_row("Factory", chosen.factory))
    if chosen.applicant:
        A(_row("Applicant on test report", chosen.applicant))
    A(_row("Source document type", chosen.document_type))
    A("")
    A("## 3. Product specifications")
    A("")
    A("### Electrical")
    A("")
    A("| Attribute | Value |")
    A("|---|---|")
    A(_row("Phase", e.phase))
    A(_row("AC voltage (V)", e.ac_voltage_v))
    A(_row("AC frequency (Hz)", e.ac_frequency_hz))
    A(_row("Rated AC power (W)", e.rated_power_w))
    A(_row("Power factor", e.power_factor))
    A(_row("THD (%)", e.thd_pct))
    A(_row("Max efficiency (%)", e.max_efficiency_pct))
    A(_row("Euro efficiency (%)", e.euro_efficiency_pct))
    A(_row("MPPT efficiency (%)", e.mppt_efficiency_pct))
    A(_row("Max DC input voltage (V)", e.max_dc_input_voltage_v))
    A(_row("MPPT voltage range (V)", e.mppt_voltage_range_v))
    A("")
    A("### Mechanical")
    A("")
    A("| Attribute | Value |")
    A("|---|---|")
    A(_row("IP rating", m.ip_rating))
    A(_row("Topology", m.topology))
    A(_row("Operating temp (°C)", m.operating_temp_range_c))
    A(_row("Weight (kg)", m.weight_kg))
    A(_row("Dimensions (mm)", m.dimensions_mm))
    A(_row("Cooling", m.cooling))
    A(_row("Protective class", m.protective_class))
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
    if chosen.model_numbers:
        groups: dict[str, list[str]] = {}
        for fc in chosen.model_numbers:
            if not isinstance(fc.value, str):
                continue
            sku = fc.value.strip()
            parts = sku.split("-")
            tag = parts[-1] if parts and len(parts[-1]) <= 4 else "base"
            groups.setdefault(tag, []).append(sku)
        if len(groups) > 1:
            A("### Variant breakdown within the chosen family")
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
    A("## 4. Test information and certifications")
    A("")
    if chosen.certifications:
        A("| Standard | Cert / Report # | Issuer | Valid until |")
        A("|---|---|---|---|")
        for c in chosen.certifications:
            cert_or_report = c.cert_number if c.cert_number else c.test_report_number
            A(
                f"| {_cite(c.standard)} | {_cite(cert_or_report)} "
                f"| {_cite(c.issuer)} | {_cite(c.valid_until)} |"
            )
    else:
        A("_no certifications extracted — request full test report set from factory_")
    A("")
    A("## 5. Labeling")
    A("")
    if chosen.labeling_items:
        A("Items the manufacturer's documentation states must appear on the "
          "product nameplate / label:")
        A("")
        for fc in chosen.labeling_items:
            A(f"- {_cite(fc)}")
    else:
        A("_no labeling items extracted from the supplied PDFs — request a "
          "nameplate photograph from the factory before final submission._")
    A("")
    A("## 6. Cross-source consistency")
    A("")
    A("How the two manufacturer PDFs agree and disagree. Mismatches are not "
      "silently merged — they are surfaced here for the agent to adjudicate.")
    A("")

    agreements = _collect_agreements(p1, p2)
    A("### Items consistent across both source documents")
    A("")
    if agreements:
        A("| Field | Value (same in both) |")
        A("|---|---|")
        for label, value in agreements:
            A(f"| {label} | **{value}** |")
    else:
        A("_no exact agreements detected on scalar fields — the two PDFs cover "
          "different product families, so divergence is expected._")
    A("")

    A("### Differences (mismatches)")
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

    if prose is not None and prose.mismatch_framing:
        A("### Why this matters")
        A("")
        A(prose.mismatch_framing)
        A("")
    A("## 7. Items relevant to Nepal import review (NEPQA 2025)")
    A("")
    A("> NEPQA 2025 Section 1.4 is used here as an indicative reference of "
      "what a Nepal import review for solar inverters tends to ask for, per "
      "the assessment guidance. It is NOT being filled in as a "
      "section-by-section filing form. Use the matrix below to spot what is "
      "well-evidenced (🟢), partial (🟡), and missing (🔴).")
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

    if prose is not None and prose.gap_narrative:
        A("### What the gaps mean")
        A("")
        A(prose.gap_narrative)
        A("")
    A("## 8. What's still unclear — items to confirm with the factory")
    A("")
    if ask_factory:
        A("Before final submission to the import agent, SunBridge should ask "
          "the factory for the following:")
        A("")
        for item in ask_factory:
            A(f"- {item}")
        A("")
    else:
        A("_(populated after the self-review pass — see §9 if this draft was "
          "rendered before the critic ran.)_")
        A("")

    if critic_flags:
        A("### Self-review flags raised on this draft")
        A("")
        A("The critic node also flagged the following items in this draft "
          "that may need a second look:")
        A("")
        A("| Section | Excerpt | Issue | Suggested action |")
        A("|---|---|---|---|")
        for f in critic_flags:
            A(
                f"| {f.section} | {f.claim_excerpt[:80]} "
                f"| {f.issue} | {f.suggested_action} |"
            )
        A("")
    A("## 9. Drafter notes & limitations")
    A("")
    A("**How this was built (one paragraph for the agent)**: An automated "
      "agent read the two manufacturer PDFs page-by-page, extracted a typed "
      "product record from each with provenance attached to every field, "
      "classified whether the two PDFs describe the same product or different "
      "products, asked a human to pick which family this shipment is about, "
      "reconciled the records, mapped them against NEPQA 2025 §1.4 as an "
      "indicative reference, and rendered this draft. A self-review pass "
      "then re-read the draft against the NEPQA source pages and produced "
      "the ask-the-factory list in §8.")
    A("")
    A("**Provenance**: every value in §2–§7 carries a `(source: pdfN p.K, "
      "conf 0.XX)` citation. The agent never silently merges conflicting "
      "facts — mismatches go to §6, not into the product spec tables.")
    A("")
    A("**What was NOT verified by this draft**:")
    A("")
    A("- No physical inspection of the product or factory.")
    A("- No phone call with the manufacturer to clarify ambiguous entries.")
    A("- No check that the cited standard revisions (e.g. IEC 62109-1:2010) "
      "are still the current version Nepal accepts.")
    A("- No customs HS code, no NPR duty calculation, no port-of-entry "
      "documentation.")
    A("- The NEPQA 2025 coverage matrix is best-effort matching, not a "
      "regulator-issued conformity declaration.")
    A("")
    if max_retries:
        A(f"**Self-review loop**: critic ran with a retry budget of "
          f"{max_retries}; this draft was finalised after {retry_count} "
          f"patch cycle(s).")
        A("")

    A("---")
    A("")
    A("**Prepared by**: Automated compliance-drafting agent  ")
    A("**For**: SunBridge Trading Pvt. Ltd., Kathmandu  ")
    A("**Recipient**: Nepal import agent (review only)  ")
    A(f"**Date**: {timestamp}")
    A("")
    A("_End of draft._")

    return "\n".join(md_lines)


def drafter_node(state: AgentState) -> AgentState:
    chosen = state["chosen_record"]
    decision = state.get("variant_decision")
    mismatches = state.get("mismatches", [])
    coverage = state.get("coverage", [])
    timestamp = state.get("run_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Deterministic per-run filename — first drafter call sets it, subsequent
    # calls overwrite the SAME .md / .pdf / .json so the final draft on disk
    # is the one the critic helped produce.
    safe_ts = state.get("draft_safe_ts")
    if not safe_ts:
        safe_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prime the doc-label resolver so every _cite() call uses the friendly name.
    _DOC_NAMES.clear()
    _DOC_NAMES.update(
        {
            "pdf1": state.get("pdf1_name") or "pdf1",
            "pdf2": state.get("pdf2_name") or "pdf2",
            "nepqa": state.get("nepqa_name") or "nepqa",
        }
    )

    prose = _call_prose_llm(chosen, decision, mismatches, coverage)

    md_text = render_markdown(
        chosen,
        decision,
        mismatches,
        coverage,
        timestamp,
        p1=state.get("pdf1_record"),
        p2=state.get("pdf2_record"),
        prose=prose,
        ask_factory=state.get("ask_factory_list"),
        critic_flags=state.get("critic_flags"),
        retry_count=state.get("retry_count", 0),
        max_retries=state.get("max_retries", 0),
    )

    out_md = OUTPUTS_DIR / f"compliance_draft_{safe_ts}.md"
    out_pdf = OUTPUTS_DIR / f"compliance_draft_{safe_ts}.pdf"
    save_markdown(md_text, out_md)
    pdf_ok = markdown_to_pdf(md_text, out_pdf)

    out_state = OUTPUTS_DIR / f"agent_state_{safe_ts}.json"
    state_dump = {
        "timestamp": timestamp,
        "chosen_record": chosen.model_dump(),
        "variant_decision": decision.model_dump() if decision else None,
        "mismatches": [m_.model_dump() for m_ in mismatches],
        "coverage": [c.model_dump() for c in coverage],
        "pdf1_record": state.get("pdf1_record").model_dump() if state.get("pdf1_record") else None,
        "pdf2_record": state.get("pdf2_record").model_dump() if state.get("pdf2_record") else None,
        "critic_flags": [f.model_dump() for f in (state.get("critic_flags") or [])],
        "ask_factory_list": state.get("ask_factory_list") or [],
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 0),
    }
    out_state.write_text(json.dumps(state_dump, indent=2, default=str), encoding="utf-8")

    return {
        "draft_markdown": md_text,
        "draft_md_path": str(out_md),
        "draft_pdf_path": str(out_pdf) if pdf_ok else "",
        "draft_safe_ts": safe_ts,
    }
