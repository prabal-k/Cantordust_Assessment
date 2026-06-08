"""NEPQA mapper coverage logic tests."""
from __future__ import annotations

from src.nodes.nepqa_mapper import map_coverage
from src.schemas import CoverageStatus


def test_coverage_covered_when_threshold_met(sample_record_pdf1, nepqa_items_basic):
    """PDF1 record has IEC 62109-1 cert → first NEPQA item COVERED."""
    results = map_coverage(sample_record_pdf1, nepqa_items_basic)
    iec_62109_1 = [r for r in results if r.item.clause_id == "1.4.2.d"][0]
    assert iec_62109_1.status == CoverageStatus.COVERED
    assert iec_62109_1.evidence, "covered result must carry evidence FieldClaims"


def test_coverage_missing_when_no_evidence(sample_record_pdf1, nepqa_items_basic):
    """PDF1 record has no IEC 62891 cert → MISSING with gap_note."""
    results = map_coverage(sample_record_pdf1, nepqa_items_basic)
    iec_62891 = [r for r in results if r.item.clause_id == "1.4.2.c"][0]
    assert iec_62891.status == CoverageStatus.MISSING
    assert iec_62891.gap_note is not None
    assert "62891" in iec_62891.gap_note
