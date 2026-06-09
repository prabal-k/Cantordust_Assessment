"""Pydantic v2 contracts. Every extracted value is a FieldClaim (value +
source_doc + source_page + confidence) so every claim in the draft carries
its own citation."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


SourceDoc = Literal["pdf1", "pdf2", "nepqa"]


class FieldClaim(BaseModel):
    """Atomic typed fact with provenance. `value` is str-typed so Gemini's
    Schema proto (which rejects `anyOf`) accepts it; numerics are stringified
    by the LLM and rendered as text in citations downstream."""

    model_config = ConfigDict(extra="forbid")

    value: str
    source_doc: SourceDoc
    source_page: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    notes: Optional[str] = None


class ElectricalSpecs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ac_voltage_v: Optional[FieldClaim] = None
    ac_frequency_hz: Optional[FieldClaim] = None
    rated_power_w: Optional[FieldClaim] = None
    phase: Optional[FieldClaim] = None
    max_efficiency_pct: Optional[FieldClaim] = None
    euro_efficiency_pct: Optional[FieldClaim] = None
    mppt_efficiency_pct: Optional[FieldClaim] = None
    thd_pct: Optional[FieldClaim] = None
    power_factor: Optional[FieldClaim] = None
    max_dc_input_voltage_v: Optional[FieldClaim] = None
    mppt_voltage_range_v: Optional[FieldClaim] = None


class MechanicalSpecs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ip_rating: Optional[FieldClaim] = None
    operating_temp_range_c: Optional[FieldClaim] = None
    weight_kg: Optional[FieldClaim] = None
    dimensions_mm: Optional[FieldClaim] = None
    cooling: Optional[FieldClaim] = None
    topology: Optional[FieldClaim] = None
    protective_class: Optional[FieldClaim] = None


class Certification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    standard: FieldClaim
    cert_number: Optional[FieldClaim] = None
    issuer: Optional[FieldClaim] = None
    valid_until: Optional[FieldClaim] = None
    test_report_number: Optional[FieldClaim] = None


class ProductRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family_label: str = Field(
        description="Short tag: 'microinverter', 'string_inverter', 'hybrid_inverter', etc."
    )
    source_doc: Literal["pdf1", "pdf2"]
    document_type: FieldClaim = Field(
        description="Document type: 'test_report', 'certificate_of_conformity', 'datasheet', etc."
    )
    model_numbers: list[FieldClaim] = Field(default_factory=list)
    manufacturer: FieldClaim
    factory: Optional[FieldClaim] = None
    applicant: Optional[FieldClaim] = None
    electrical: ElectricalSpecs = Field(default_factory=ElectricalSpecs)
    mechanical: MechanicalSpecs = Field(default_factory=MechanicalSpecs)
    certifications: list[Certification] = Field(default_factory=list)
    warranty_years: Optional[FieldClaim] = None
    labeling_items: list[FieldClaim] = Field(default_factory=list)


class NEPQAItemType(str, Enum):
    # Uppercase values match Groq's Llama-3.3 tool-call enum naming.
    DOCUMENT = "DOCUMENT"
    TECHNICAL = "TECHNICAL"
    LABEL = "LABEL"
    GENERAL = "GENERAL"


class NEPQAItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clause_id: str = Field(description="e.g. '1.4.2.a' or '1.4.3.ix'")
    requirement_text: str
    item_type: NEPQAItemType
    expected_value: Optional[str] = Field(
        default=None,
        description="Parsed threshold for technical items (e.g. 'THD < 5%', 'IP65').",
    )
    source_page: int = Field(ge=1)


class CoverageStatus(str, Enum):
    COVERED = "COVERED"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CoverageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: NEPQAItem
    status: CoverageStatus
    evidence: list[FieldClaim] = Field(default_factory=list)
    gap_note: Optional[str] = None


class MismatchSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class MismatchEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_path: str
    pdf1_value: Optional[str] = None
    pdf2_value: Optional[str] = None
    severity: MismatchSeverity
    recommendation: str


class VariantRelationship(str, Enum):
    SAME_PRODUCT = "SAME_PRODUCT"
    VARIANT = "VARIANT"
    DIFFERENT_FAMILY = "DIFFERENT_FAMILY"
    OEM_SAME_FACTORY = "OEM_SAME_FACTORY"


class VariantDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relationship: VariantRelationship
    reasoning: str
    shared_attributes: list[str] = Field(default_factory=list)
    distinguishing_attributes: list[str] = Field(default_factory=list)
    requires_human_choice: bool


class HumanChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chosen_family: Literal["pdf1", "pdf2"]
    rationale: Optional[str] = None


class CriticFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section: str
    claim_excerpt: str
    issue: str
    suggested_action: str


class NEPQAChecklist(BaseModel):
    """Wrapper so structured-output APIs that need a top-level object have one."""

    model_config = ConfigDict(extra="forbid")

    items: list[NEPQAItem]


class CriticReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flags: list[CriticFlag] = Field(default_factory=list)
    ask_factory: list[str] = Field(default_factory=list)


class DrafterProse(BaseModel):
    """Four flat strings (no nested lists/tables) so Gemini's Schema proto
    accepts them. Empty string = no synthesis possible; template skips that
    block."""

    model_config = ConfigDict(extra="forbid")

    cover_note: str = Field(
        description=(
            "4-6 sentences addressed to the Nepal import agent: what is in this "
            "draft, which product was chosen, how to read it. Opens the document."
        )
    )
    methodology_note: str = Field(
        description=(
            "3-5 sentences explaining how the draft was assembled — sources read, "
            "what was reconciled, what was deferred. Addresses Ramesh's ask for "
            "'a short note on how you approached it'."
        )
    )
    gap_narrative: str = Field(
        description="Plain-English grouping of partial/missing NEPQA items and what they imply."
    )
    mismatch_framing: str = Field(
        description="Why the cross-source differences matter for THIS shipment."
    )
