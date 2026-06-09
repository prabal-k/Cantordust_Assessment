"""Hybrid drafter prose tests.

Covers:
  - prose blocks inject at the correct insertion points in render_markdown
  - the digit-leak guard rejects fabricated numbers + retries
  - persistent violation falls back to empty prose (deterministic core still ships)
  - sanitizer strips citation patterns the LLM was told not to write
  - new submission-grade sections (cover note, methodology, §8 ask-factory,
    §9 drafter notes, sign-off block) render correctly
"""
from __future__ import annotations

from unittest.mock import patch

from src.nodes.drafter import (
    _allowed_digits,
    _call_prose_llm,
    _orphan_digits,
    _sanitize_prose,
    render_markdown,
)
from src.schemas import (
    CoverageResult,
    CoverageStatus,
    CriticFlag,
    DrafterProse,
    MismatchEntry,
    MismatchSeverity,
    VariantDecision,
    VariantRelationship,
)


def _make_coverage(items):
    return [
        CoverageResult(item=item, status=CoverageStatus.COVERED, evidence=[])
        for item in items
    ]


def _decision_diff_family():
    return VariantDecision(
        relationship=VariantRelationship.DIFFERENT_FAMILY,
        reasoning="Disjoint model sets, different phase, different power range.",
        shared_attributes=["factory"],
        distinguishing_attributes=["family_label", "phase"],
        requires_human_choice=True,
    )


def _full_prose():
    return DrafterProse(
        cover_note="COVER_NOTE_MARKER body.",
        methodology_note="METHOD_NOTE_MARKER body.",
        gap_narrative="GAP_NARRATIVE_MARKER body.",
        mismatch_framing="MISMATCH_FRAMING_MARKER body.",
    )


# --- Sanitizer ----------------------------------------------------------

def test_sanitize_strips_source_citation():
    s = "The product is rated 230V (source: pdf1 p.7) per the test report."
    assert "(source:" not in _sanitize_prose(s)
    assert "p.7" not in _sanitize_prose(s)


def test_sanitize_strips_conf_pattern():
    s = "Manufacturer name confirmed conf 0.95 across both docs."
    assert "conf 0.95" not in _sanitize_prose(s)


# --- Allow-list + orphan detection --------------------------------------

def test_orphan_digits_finds_fabrication(sample_record_pdf1, nepqa_items_basic):
    allowed = _allowed_digits(
        sample_record_pdf1, _make_coverage(nepqa_items_basic), [], None
    )
    orphans = _orphan_digits("The factory shipped 999 units.", allowed)
    assert "999" in orphans


def test_orphan_digits_allows_input_numbers(sample_record_pdf1, nepqa_items_basic):
    allowed = _allowed_digits(
        sample_record_pdf1, _make_coverage(nepqa_items_basic), [], None
    )
    assert _orphan_digits("Rated at 230V single phase.", allowed) == []


def test_structural_small_digits_allowed(sample_record_pdf1, nepqa_items_basic):
    allowed = _allowed_digits(
        sample_record_pdf1, _make_coverage(nepqa_items_basic), [], None
    )
    assert _orphan_digits("There are 2 source PDFs and 3 prose blocks.", allowed) == []


def test_nepqa_2025_year_is_structural(sample_record_pdf1, nepqa_items_basic):
    """LLM can reference 'NEPQA 2025' in prose without tripping the guard."""
    allowed = _allowed_digits(
        sample_record_pdf1, _make_coverage(nepqa_items_basic), [], None
    )
    assert _orphan_digits("Aligned against the NEPQA 2025 reference.", allowed) == []


# --- Submission-grade template assertions -------------------------------

def test_render_markdown_has_doc_control_header(sample_record_pdf1, nepqa_items_basic):
    md = render_markdown(
        sample_record_pdf1,
        _decision_diff_family(),
        [],
        _make_coverage(nepqa_items_basic),
        "2026-06-09 12:00:00",
    )
    assert "# Nepal Import Compliance Draft" in md
    assert "**Importer**" in md
    assert "SunBridge Trading" in md
    assert "**Recipient**" in md
    assert "**Generated**" in md
    assert "**Prepared by**" in md


def test_render_markdown_has_all_ten_sections(sample_record_pdf1, nepqa_items_basic):
    md = render_markdown(
        sample_record_pdf1,
        _decision_diff_family(),
        [],
        _make_coverage(nepqa_items_basic),
        "2026-06-09 12:00:00",
    )
    expected = [
        "## Cover note for the import agent",
        "## How this draft was assembled",
        "## 1. Product and variant identification",
        "## 2. Manufacturer and factory",
        "## 3. Product specifications",
        "## 4. Test information and certifications",
        "## 5. Labeling",
        "## 6. Cross-source consistency",
        "## 7. Items relevant to Nepal import review",
        "## 8. What's still unclear",
        "## 9. Drafter notes & limitations",
    ]
    for header in expected:
        assert header in md, f"missing section header: {header}"


def test_section_order_is_correct(sample_record_pdf1, nepqa_items_basic):
    md = render_markdown(
        sample_record_pdf1,
        _decision_diff_family(),
        [],
        _make_coverage(nepqa_items_basic),
        "2026-06-09 12:00:00",
    )
    # Cross-source (§6) before NEPQA (§7) — assessment criterion ordering
    assert md.index("## 6. Cross-source consistency") < md.index(
        "## 7. Items relevant to Nepal import review"
    )
    # §8 ask-factory before §9 drafter notes
    assert md.index("## 8. What's still unclear") < md.index(
        "## 9. Drafter notes"
    )


def test_render_markdown_has_signoff_block(sample_record_pdf1, nepqa_items_basic):
    md = render_markdown(
        sample_record_pdf1,
        _decision_diff_family(),
        [],
        _make_coverage(nepqa_items_basic),
        "2026-06-09 12:00:00",
    )
    assert "Prepared by" in md
    assert "_End of draft._" in md


def test_render_markdown_uses_fallback_cover_when_prose_empty(
    sample_record_pdf1, nepqa_items_basic
):
    md = render_markdown(
        sample_record_pdf1,
        _decision_diff_family(),
        [],
        _make_coverage(nepqa_items_basic),
        "2026-06-09 12:00:00",
        prose=None,
    )
    # Cover + methodology sections appear even without LLM prose
    assert "## Cover note for the import agent" in md
    assert "## How this draft was assembled" in md
    assert "indicative import-side guideline" in md or "indicative reference" in md.lower()


def test_nepqa_section_carries_indicative_disclaimer(
    sample_record_pdf1, nepqa_items_basic
):
    """§7 must signal NEPQA is reference-only, not a section-by-section form."""
    md = render_markdown(
        sample_record_pdf1,
        _decision_diff_family(),
        [],
        _make_coverage(nepqa_items_basic),
        "2026-06-09 12:00:00",
    )
    assert "indicative reference" in md.lower()
    assert "section-by-section" in md.lower()


# --- Prose injection ----------------------------------------------------

def test_all_four_prose_blocks_inject_at_correct_points(
    sample_record_pdf1, nepqa_items_basic
):
    mismatches = [
        MismatchEntry(
            field_path="electrical.phase",
            pdf1_value="single",
            pdf2_value="three",
            severity=MismatchSeverity.CRITICAL,
            recommendation="Confirm which family the shipment covers.",
        )
    ]
    md = render_markdown(
        sample_record_pdf1,
        _decision_diff_family(),
        mismatches,
        _make_coverage(nepqa_items_basic),
        "2026-06-09 12:00:00",
        prose=_full_prose(),
    )
    assert "COVER_NOTE_MARKER" in md
    assert "METHOD_NOTE_MARKER" in md
    assert "GAP_NARRATIVE_MARKER" in md
    assert "MISMATCH_FRAMING_MARKER" in md

    # cover_note appears in §"Cover note", methodology in §"How", gap in §7, framing in §6
    assert md.index("COVER_NOTE_MARKER") < md.index("METHOD_NOTE_MARKER")
    assert md.index("METHOD_NOTE_MARKER") < md.index("## 1.")
    assert md.index("MISMATCH_FRAMING_MARKER") > md.index("### Differences")
    assert md.index("MISMATCH_FRAMING_MARKER") < md.index("## 7.")
    assert md.index("GAP_NARRATIVE_MARKER") > md.index("## 7.")


# --- Section 8 ask_factory + critic_flags -------------------------------

def test_section_8_renders_ask_factory_list(sample_record_pdf1, nepqa_items_basic):
    ask = [
        "Photograph of nameplate label",
        "Confirmation of warranty period in writing",
    ]
    md = render_markdown(
        sample_record_pdf1,
        _decision_diff_family(),
        [],
        _make_coverage(nepqa_items_basic),
        "2026-06-09 12:00:00",
        ask_factory=ask,
    )
    assert "## 8. What's still unclear" in md
    for item in ask:
        assert item in md


def test_section_8_placeholder_when_no_ask_factory_yet(
    sample_record_pdf1, nepqa_items_basic
):
    """First drafter pass (before critic) renders §8 as a placeholder."""
    md = render_markdown(
        sample_record_pdf1,
        _decision_diff_family(),
        [],
        _make_coverage(nepqa_items_basic),
        "2026-06-09 12:00:00",
        ask_factory=None,
    )
    assert "populated after the self-review pass" in md


def test_section_8_surfaces_critic_flags(sample_record_pdf1, nepqa_items_basic):
    flags = [
        CriticFlag(
            section="§3 Product specifications",
            claim_excerpt="THD < 5%",
            issue="THD value not present in supplied source pages.",
            suggested_action="Request manufacturer test report appendix.",
        )
    ]
    md = render_markdown(
        sample_record_pdf1,
        _decision_diff_family(),
        [],
        _make_coverage(nepqa_items_basic),
        "2026-06-09 12:00:00",
        critic_flags=flags,
    )
    assert "Self-review flags" in md
    assert "§3 Product specifications" in md
    assert "Request manufacturer test report appendix" in md


# --- LLM call + guard ---------------------------------------------------

def test_call_prose_llm_returns_clean_prose(sample_record_pdf1, nepqa_items_basic):
    coverage = _make_coverage(nepqa_items_basic)
    clean = DrafterProse(
        cover_note="The shipment is a microinverter product line.",
        methodology_note="Two manufacturer PDFs were read.",
        gap_narrative="Required test reports were verified against the source.",
        mismatch_framing="Both documents describe the same product line.",
    )
    with patch("src.nodes.drafter.invoke_structured", return_value=clean) as mock_llm:
        result = _call_prose_llm(sample_record_pdf1, None, [], coverage)
    assert mock_llm.call_count == 1
    assert result.cover_note.startswith("The shipment")


def test_call_prose_llm_retries_on_orphan_digit(sample_record_pdf1, nepqa_items_basic):
    coverage = _make_coverage(nepqa_items_basic)
    bad = DrafterProse(
        cover_note="The factory shipped 999 units last year.",
        methodology_note="",
        gap_narrative="",
        mismatch_framing="",
    )
    good = DrafterProse(
        cover_note="The shipment is a microinverter product line.",
        methodology_note="",
        gap_narrative="",
        mismatch_framing="",
    )
    with patch(
        "src.nodes.drafter.invoke_structured", side_effect=[bad, good]
    ) as mock_llm:
        result = _call_prose_llm(sample_record_pdf1, None, [], coverage)
    assert mock_llm.call_count == 2
    assert "999" not in result.cover_note
    assert result.cover_note == good.cover_note


def test_call_prose_llm_falls_back_to_empty_after_max_attempts(
    sample_record_pdf1, nepqa_items_basic
):
    coverage = _make_coverage(nepqa_items_basic)
    bad = DrafterProse(
        cover_note="They sold 7777 units in 8888 markets.",
        methodology_note="",
        gap_narrative="",
        mismatch_framing="",
    )
    with patch("src.nodes.drafter.invoke_structured", return_value=bad):
        result = _call_prose_llm(sample_record_pdf1, None, [], coverage, max_attempts=2)
    assert result.cover_note == ""
    assert result.methodology_note == ""
    assert result.gap_narrative == ""
    assert result.mismatch_framing == ""


def test_call_prose_llm_falls_back_on_llm_exception(
    sample_record_pdf1, nepqa_items_basic
):
    coverage = _make_coverage(nepqa_items_basic)
    with patch(
        "src.nodes.drafter.invoke_structured", side_effect=RuntimeError("rate limit")
    ):
        result = _call_prose_llm(sample_record_pdf1, None, [], coverage)
    assert result.cover_note == ""
