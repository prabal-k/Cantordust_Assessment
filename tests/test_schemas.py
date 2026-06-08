"""Schema validation tests. No LLM calls."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.schemas import FieldClaim, ProductRecord


def test_fieldclaim_requires_provenance():
    # source_page missing
    with pytest.raises(ValidationError):
        FieldClaim(value="230V", source_doc="pdf1", confidence=0.9)
    # source_doc missing
    with pytest.raises(ValidationError):
        FieldClaim(value="230V", source_page=7, confidence=0.9)
    # confidence > 1.0 rejected
    with pytest.raises(ValidationError):
        FieldClaim(value="230V", source_doc="pdf1", source_page=7, confidence=1.5)
    # confidence < 0 rejected
    with pytest.raises(ValidationError):
        FieldClaim(value="230V", source_doc="pdf1", source_page=7, confidence=-0.1)
    # source_page must be >= 1
    with pytest.raises(ValidationError):
        FieldClaim(value="230V", source_doc="pdf1", source_page=0, confidence=0.9)


def test_productrecord_round_trip(sample_record_pdf1):
    js = sample_record_pdf1.model_dump_json()
    parsed = json.loads(js)
    assert parsed["family_label"] == "microinverter"
    rebuilt = ProductRecord(**parsed)
    assert rebuilt == sample_record_pdf1
