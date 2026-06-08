"""Tests for the family_label / phase normalizer."""
from __future__ import annotations

from src.nodes.extract_utils import normalize_record
from src.schemas import (
    Certification,
    ElectricalSpecs,
    FieldClaim,
    MechanicalSpecs,
    ProductRecord,
)


def _claim(value, doc="pdf2", page=1, conf=0.9):
    return FieldClaim(value=str(value), source_doc=doc, source_page=page, confidence=conf)


def _record(
    *,
    family_label="unknown",
    source_doc="pdf2",
    phase_value=None,
    voltage_value=None,
    power_value=None,
):
    electrical = ElectricalSpecs(
        phase=_claim(phase_value, doc=source_doc) if phase_value else None,
        ac_voltage_v=_claim(voltage_value, doc=source_doc) if voltage_value else None,
        rated_power_w=_claim(power_value, doc=source_doc) if power_value else None,
    )
    return ProductRecord(
        family_label=family_label,
        source_doc=source_doc,
        document_type=_claim("certificate", doc=source_doc),
        model_numbers=[],
        manufacturer=_claim("acme", doc=source_doc),
        electrical=electrical,
        mechanical=MechanicalSpecs(),
        certifications=[],
    )


def test_three_phase_400v_classified_as_three_phase_string_inverter():
    """The Deye SUN-XK case: '3L/N/PE 230/400V', 3000W rated."""
    r = _record(
        family_label="single_phase_string_inverter",
        source_doc="pdf2",
        voltage_value="3L/N/PE  230/400V",
        power_value="3000",
        phase_value=None,
    )
    out = normalize_record(r, expected_source="pdf2")
    assert out.family_label == "three_phase_string_inverter"
    assert out.electrical.phase is not None
    assert out.electrical.phase.value == "three"


def test_single_phase_low_power_classified_as_microinverter():
    """The Chisage CE-1Pxxxx case: 230V single phase, 500W."""
    r = _record(
        family_label="single_phase_string_inverter",
        source_doc="pdf1",
        voltage_value="230V",
        power_value="500",
        phase_value="single",
    )
    out = normalize_record(r, expected_source="pdf1")
    assert out.family_label == "microinverter"


def test_single_phase_high_power_classified_as_single_phase_string():
    r = _record(
        family_label="microinverter",
        source_doc="pdf1",
        voltage_value="230V",
        power_value="3000",
        phase_value="single",
    )
    out = normalize_record(r, expected_source="pdf1")
    assert out.family_label == "single_phase_string_inverter"


def test_source_doc_force_correction():
    r = _record(source_doc="pdf1")  # built saying pdf1
    out = normalize_record(r, expected_source="pdf2")
    assert out.source_doc == "pdf2"
