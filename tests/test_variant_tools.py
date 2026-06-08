"""Unit tests for the 5 variant_detector tools.

Tools are pure functions over two ProductRecords + a shared sink. No LLM, no
mocks needed.
"""
from __future__ import annotations

from src.nodes.variant_tools import (
    build_decision_from_sink,
    build_variant_tools,
)
from src.schemas import VariantRelationship


def test_compare_field_match(sample_record_pdf1, sample_record_pdf2):
    sink: list[dict] = []
    tools = build_variant_tools(sample_record_pdf1, sample_record_pdf2, sink)
    compare = next(t for t in tools if t.name == "compare_field")
    result = compare.invoke({"field_path": "mechanical.topology"})
    assert result["match"] is True
    assert result["pdf1_value"] == "transformerless"
    assert result["pdf2_value"] == "transformerless"


def test_compare_field_mismatch(sample_record_pdf1, sample_record_pdf2):
    sink: list[dict] = []
    tools = build_variant_tools(sample_record_pdf1, sample_record_pdf2, sink)
    compare = next(t for t in tools if t.name == "compare_field")
    result = compare.invoke({"field_path": "electrical.phase"})
    assert result["match"] is False
    assert result["pdf1_value"] == "single"
    assert result["pdf2_value"] == "three"


def test_get_models(sample_record_pdf1, sample_record_pdf2):
    sink: list[dict] = []
    tools = build_variant_tools(sample_record_pdf1, sample_record_pdf2, sink)
    get_models = next(t for t in tools if t.name == "get_models")
    assert get_models.invoke({"pdf": "pdf1"}) == ["CE-1P5001G-230-EU"]
    assert get_models.invoke({"pdf": "pdf2"}) == ["SUN-5K-G06P3-EU-AM2-P1"]


def test_check_factory_match(sample_record_pdf1, sample_record_pdf2):
    sink: list[dict] = []
    tools = build_variant_tools(sample_record_pdf1, sample_record_pdf2, sink)
    check = next(t for t in tools if t.name == "check_factory_match")
    result = check.invoke({})
    assert result["match"] is True
    assert "deye" in result["pdf1_factory"].lower()
    assert "deye" in result["pdf2_factory"].lower()


def test_check_certifications_overlap(sample_record_pdf1, sample_record_pdf2):
    sink: list[dict] = []
    tools = build_variant_tools(sample_record_pdf1, sample_record_pdf2, sink)
    check = next(t for t in tools if t.name == "check_certifications_overlap")
    result = check.invoke({})
    assert result["shared"] == []
    assert result["only_pdf1"] == ["IEC 62109-1:2010"]
    assert "IEC 62116:2014" in result["only_pdf2"]
    assert "IEC 61727:2004" in result["only_pdf2"]


def test_commit_decision_then_build_from_sink(
    sample_record_pdf1, sample_record_pdf2
):
    sink: list[dict] = []
    tools = build_variant_tools(sample_record_pdf1, sample_record_pdf2, sink)
    commit = next(t for t in tools if t.name == "commit_decision")
    result = commit.invoke(
        {
            "relationship": "DIFFERENT_FAMILY",
            "reasoning": "Different phase + model sets + standards.",
            "shared_attributes": ["factory"],
            "distinguishing_attributes": ["phase", "model_numbers", "certifications"],
            "requires_human_choice": True,
        }
    )
    assert result == "committed"
    decision = build_decision_from_sink(sink)
    assert decision is not None
    assert decision.relationship == VariantRelationship.DIFFERENT_FAMILY
    assert decision.requires_human_choice is True
    assert "factory" in decision.shared_attributes


def test_build_decision_from_empty_sink_returns_none():
    assert build_decision_from_sink([]) is None
