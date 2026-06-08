"""Tests for the ReAct-agent path + fallback path in variant_detector_node.

Approach: monkeypatch `_run_react_agent` and `_fallback_single_shot` to return
canned VariantDecision values, then assert behavior:
  - When agent path succeeds → that decision is used (subject to sanity override).
  - When agent path returns None → fallback path runs.
  - Sanity override still fires after agent path on obviously different families.
"""
from __future__ import annotations

from src.schemas import VariantDecision, VariantRelationship


def _make_decision(rel: VariantRelationship, requires_human: bool = False) -> VariantDecision:
    return VariantDecision(
        relationship=rel,
        reasoning="test",
        shared_attributes=[],
        distinguishing_attributes=[],
        requires_human_choice=requires_human,
    )


def test_agent_path_returns_decision(
    monkeypatch, sample_record_pdf1, sample_record_pdf2
):
    from src.nodes import variant_detector

    monkeypatch.setattr(
        variant_detector,
        "_run_react_agent",
        lambda p1, p2: _make_decision(VariantRelationship.DIFFERENT_FAMILY, True),
    )
    # fallback should NOT be called if agent succeeds
    def _boom(*a, **kw):
        raise AssertionError("fallback should not run when agent succeeds")

    monkeypatch.setattr(variant_detector, "_fallback_single_shot", _boom)

    state = {"pdf1_record": sample_record_pdf1, "pdf2_record": sample_record_pdf2}
    update = variant_detector.variant_detector_node(state)
    decision = update["variant_decision"]
    assert decision.relationship == VariantRelationship.DIFFERENT_FAMILY
    # sanity check still fires & agrees — requires_human stays True
    assert decision.requires_human_choice is True


def test_fallback_when_agent_returns_none(
    monkeypatch, sample_record_pdf1, sample_record_pdf2
):
    from src.nodes import variant_detector

    monkeypatch.setattr(variant_detector, "_run_react_agent", lambda p1, p2: None)

    called: list[bool] = []

    def _fake_fallback(p1, p2):
        called.append(True)
        return _make_decision(VariantRelationship.DIFFERENT_FAMILY, True)

    monkeypatch.setattr(variant_detector, "_fallback_single_shot", _fake_fallback)

    state = {"pdf1_record": sample_record_pdf1, "pdf2_record": sample_record_pdf2}
    update = variant_detector.variant_detector_node(state)
    assert called == [True]
    assert update["variant_decision"].relationship == VariantRelationship.DIFFERENT_FAMILY


def test_sanity_override_fires_when_agent_picks_same_product(
    monkeypatch, sample_record_pdf1, sample_record_pdf2
):
    """Agent over-confidently picks SAME_PRODUCT; sanity override must flip it."""
    from src.nodes import variant_detector

    monkeypatch.setattr(
        variant_detector,
        "_run_react_agent",
        lambda p1, p2: _make_decision(VariantRelationship.SAME_PRODUCT, False),
    )
    monkeypatch.setattr(variant_detector, "_fallback_single_shot", lambda *a: None)

    state = {"pdf1_record": sample_record_pdf1, "pdf2_record": sample_record_pdf2}
    update = variant_detector.variant_detector_node(state)
    decision = update["variant_decision"]
    # The fixtures are intentionally different families (microinverter vs
    # string_inverter, single vs three phase, disjoint models).
    assert decision.relationship == VariantRelationship.DIFFERENT_FAMILY
    assert decision.requires_human_choice is True
    assert "Sanity override" in decision.reasoning
