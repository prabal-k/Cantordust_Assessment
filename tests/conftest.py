"""Shared test fixtures."""
from __future__ import annotations

import pytest

from src.schemas import (
    Certification,
    ElectricalSpecs,
    FieldClaim,
    MechanicalSpecs,
    NEPQAItem,
    NEPQAItemType,
    ProductRecord,
)


@pytest.fixture
def fc_pdf1():
    def make(value, page=1, conf=0.9):
        return FieldClaim(value=str(value), source_doc="pdf1", source_page=page, confidence=conf)
    return make


@pytest.fixture
def fc_pdf2():
    def make(value, page=1, conf=0.9):
        return FieldClaim(value=str(value), source_doc="pdf2", source_page=page, confidence=conf)
    return make


@pytest.fixture
def sample_record_pdf1(fc_pdf1):
    """Minimal Chisage microinverter record from pdf1."""
    return ProductRecord(
        family_label="microinverter",
        source_doc="pdf1",
        document_type=fc_pdf1("test_report", page=1),
        model_numbers=[fc_pdf1("CE-1P5001G-230-EU", page=7)],
        manufacturer=fc_pdf1("Zhejiang CHISAGE New Energy Technology Co., Ltd", page=2),
        factory=fc_pdf1("NingBo Deye Inverter Technology Co., Ltd.", page=6),
        applicant=fc_pdf1("Zhejiang CHISAGE New Energy Technology Co., Ltd", page=1),
        electrical=ElectricalSpecs(
            ac_voltage_v=fc_pdf1(230.0, page=7),
            ac_frequency_hz=fc_pdf1(50.0, page=7),
            rated_power_w=fc_pdf1(500.0, page=7),
            phase=fc_pdf1("single", page=2),
            power_factor=fc_pdf1(0.99, page=7),
        ),
        mechanical=MechanicalSpecs(
            ip_rating=fc_pdf1("IP67", page=5),
            weight_kg=fc_pdf1(3.5, page=5),
            topology=fc_pdf1("transformerless", page=7),
        ),
        certifications=[
            Certification(
                standard=fc_pdf1("IEC 62109-1:2010", page=1),
                test_report_number=fc_pdf1("GZES230100125901", page=1),
                issuer=fc_pdf1("SGS-CSTC Standards Technical Services Co., Ltd.", page=1),
            ),
        ],
    )


@pytest.fixture
def sample_record_pdf2(fc_pdf2):
    """Minimal Deye string inverter record from pdf2."""
    return ProductRecord(
        family_label="string_inverter",
        source_doc="pdf2",
        document_type=fc_pdf2("certificate_of_conformity", page=1),
        model_numbers=[fc_pdf2("SUN-5K-G06P3-EU-AM2-P1", page=2)],
        manufacturer=fc_pdf2("NingBo Deye Inverter Technology Co., Ltd.", page=1),
        factory=fc_pdf2("NingBo Deye Inverter Technology Co., Ltd.", page=1),
        electrical=ElectricalSpecs(
            ac_voltage_v=fc_pdf2(400.0, page=2),
            ac_frequency_hz=fc_pdf2(50.0, page=2),
            rated_power_w=fc_pdf2(5000.0, page=2),
            phase=fc_pdf2("three", page=2),
            power_factor=fc_pdf2(0.8, page=2),
        ),
        mechanical=MechanicalSpecs(
            ip_rating=fc_pdf2("IP65", page=2),
            topology=fc_pdf2("transformerless", page=2),
        ),
        certifications=[
            Certification(
                standard=fc_pdf2("IEC 62116:2014", page=1),
                cert_number=fc_pdf2("PCS-24-1022", page=1),
                issuer=fc_pdf2("SGS Testing & Control Services Singapore Pte Ltd", page=1),
            ),
            Certification(
                standard=fc_pdf2("IEC 61727:2004", page=1),
                cert_number=fc_pdf2("PCS-24-1022", page=1),
            ),
        ],
    )


@pytest.fixture
def nepqa_items_basic():
    """Two NEPQA items for coverage testing."""
    return [
        NEPQAItem(
            clause_id="1.4.2.d",
            requirement_text="Inverter must be tested per IEC 62109-1:2010",
            item_type=NEPQAItemType.DOCUMENT,
            expected_value="IEC 62109-1:2010",
            source_page=18,
        ),
        NEPQAItem(
            clause_id="1.4.2.c",
            requirement_text="Inverter must be tested per IEC 62891:2020 (MPPT efficiency)",
            item_type=NEPQAItemType.DOCUMENT,
            expected_value="IEC 62891:2020",
            source_page=18,
        ),
    ]
