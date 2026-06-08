"""Reconciler diff logic tests."""
from __future__ import annotations

from src.nodes.reconciler import reconcile
from src.schemas import (
    FieldClaim,
    MismatchSeverity,
    VariantDecision,
    VariantRelationship,
)


def _patch_voltage(record, voltage):
    new_electrical = record.electrical.model_copy(
        update={
            "ac_voltage_v": FieldClaim(
                value=str(voltage), source_doc=record.source_doc, source_page=2, confidence=0.95
            )
        }
    )
    return record.model_copy(update={"electrical": new_electrical})


def test_reconciler_emits_mismatch_when_voltages_differ(
    sample_record_pdf1, sample_record_pdf2
):
    # Force a critical conflict: same nominal product, different AC voltage value
    p1 = _patch_voltage(sample_record_pdf1, 230.0)
    p2 = _patch_voltage(sample_record_pdf2, 220.0)
    decision = VariantDecision(
        relationship=VariantRelationship.SAME_PRODUCT,
        reasoning="same family for test",
        requires_human_choice=False,
    )
    mismatches = reconcile(p1, p2, decision)
    voltage_mismatches = [m for m in mismatches if m.field_path == "electrical.ac_voltage_v"]
    assert len(voltage_mismatches) == 1
    assert voltage_mismatches[0].severity == MismatchSeverity.CRITICAL
    assert voltage_mismatches[0].pdf1_value == "230.0"
    assert voltage_mismatches[0].pdf2_value == "220.0"


def test_reconciler_marks_different_family_as_info(
    sample_record_pdf1, sample_record_pdf2
):
    decision = VariantDecision(
        relationship=VariantRelationship.DIFFERENT_FAMILY,
        reasoning="microinverter vs string inverter",
        requires_human_choice=True,
    )
    mismatches = reconcile(sample_record_pdf1, sample_record_pdf2, decision)
    assert mismatches, "expected at least one mismatch given the divergent records"
    # No CRITICAL or WARNING in DIFFERENT_FAMILY mode — everything downgrades to INFO
    severities = {m.severity for m in mismatches}
    assert MismatchSeverity.CRITICAL not in severities
    assert MismatchSeverity.WARNING not in severities
    assert MismatchSeverity.INFO in severities
