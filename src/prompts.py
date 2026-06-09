"""Centralized prompts for every LLM node.

Each prompt is token-optimized: imperative rules, no conversational framing,
schema-critical strings (field names, enum values) kept exact. Every
extraction prompt enforces:
  - FieldClaim.source_page from `=== page N ===` markers
  - null for missing optional fields (no fabrication)
  - honest confidence (lower for fuzzy matches)
"""
from __future__ import annotations


JSON_ONLY_GUARD = """

OUTPUT: one JSON object validating the schema below. No prose, no fences,
no trailing commas. Use lowercase `null` for missing optionals. Enum values
case-sensitive exact.

SCHEMA:
{json_schema}
"""


PRODUCT_EXTRACTION_SYSTEM = """\
Extract a typed ProductRecord from solar PV inverter compliance PDF text.

RULES:
1. Every FieldClaim cites source_page from the `=== page N ===` marker before
   the text where you found the value. Never invent a page.
2. Missing field → null. Never fabricate values.
3. confidence: ~0.95 verbatim, ~0.7 inferred/reformatted, lower for ambiguous.
4. family_label — pick ONE using these rules in order:
   (a) phase="three" OR 3-phase wiring (e.g. 3L/N/PE, 400V three-phase) AND
       any rated power > 2kW → "three_phase_string_inverter"
   (b) phase="single" AND all rated power ≤ 2kW (2000W) → "microinverter"
   (c) phase="single" AND any rated power > 2kW → "single_phase_string_inverter"
   (d) battery/hybrid topology mentioned → "hybrid_inverter"
   (e) else → "unknown"
   Derive from THIS document only — don't pull from prior knowledge or the
   other PDF.
5. model_numbers: every SKU, each its own FieldClaim with its page.
6. certifications: every distinct standard (e.g. "IEC 62109-1:2010") with
   optional cert/test report number, issuer, validity.
7. labeling_items: every nameplate/label item the document specifies, each
   its own FieldClaim.
"""


def product_extraction_user(source_doc: str, sliced_text: str) -> str:
    return f"""\
Text below sliced from `{source_doc}`. Use `=== page N ===` markers for
source_page. source_doc field MUST equal `{source_doc}`.

--- DOCUMENT TEXT ---
{sliced_text}
--- END ---
"""


NEPQA_EXTRACTION_SYSTEM = """\
Extract the import-side checklist for grid-tied PV inverters from NEPQA 2025
Section 1.4 (PV Inverter / Grid Connected Inverter).

RULES:
1. One NEPQAItem per atomic requirement.
2. clause_id uses document numbering:
     1.4.2.x for required documents (a, b, c...)
     1.4.3.x for technical reqs (i, ii, iii...)
     1.4.3.xvii.x for the label-content sublist inside xvii
3. item_type:
     DOCUMENT  — test certs, standards, importer agreements, datasheets
     TECHNICAL — numeric/qualitative spec thresholds (efficiency, THD, IP,
                 voltage, frequency, warranty years)
     LABEL     — nameplate items
     GENERAL   — scope/applicability paragraphs
4. expected_value: terse threshold for TECHNICAL ("THD < 5% at full load",
   "IP65", "≥ 5 years"); standard ref for DOCUMENT ("IEC 62109-1:2010");
   null for GENERAL.
5. source_page from `=== page N ===` marker before each clause.
6. requirement_text: one-sentence paraphrase/quote.
"""


def nepqa_extraction_user(sliced_text: str) -> str:
    return f"""\
Extract every clause from NEPQA Section 1.4 below as a list of NEPQAItem.

--- NEPQA SECTION 1.4 ---
{sliced_text}
--- END ---
"""


VARIANT_AGENT_SYSTEM = """\
ReAct agent. Classify the relationship between two ProductRecords from one
shipment.

TOOLS:
  compare_field(field_path)              diff one field
  get_models(pdf)                        list SKUs from one record
  check_factory_match()                  factories identical?
  check_certifications_overlap()         shared / only_pdf1 / only_pdf2 standards
  commit_decision(relationship, reasoning, shared_attributes,
                  distinguishing_attributes, requires_human_choice)
                                         FINAL — records verdict, terminates.

WORKFLOW:
1. ≤6 read-only tool calls to gather evidence. Suggested set: get_models x2,
   check_factory_match, check_certifications_overlap, compare_field(phase),
   compare_field(rated_power_w or family_label).
2. Pick ONE relationship (UPPERCASE exact):
     SAME_PRODUCT      identical models AND specs
     VARIANT           same family, minor delta (e.g. AM2 vs AM2-P1)
     DIFFERENT_FAMILY  clearly different lines (microinverter vs string), even
                       same factory
     OEM_SAME_FACTORY  same factory address, different brand owners, different
                       product lines (common in Chinese solar OEM)
3. requires_human_choice = True for DIFFERENT_FAMILY / OEM_SAME_FACTORY,
   False for SAME_PRODUCT / VARIANT.
4. End by calling commit_decision EXACTLY ONCE. After it returns 'committed',
   STOP. Do NOT call any tool again. Do NOT re-run the evidence-gathering
   sequence. Reply with the literal word 'done' and nothing else.
5. If uncertain, still commit with best guess + low-confidence reasoning.
   Never skip the commit.
6. If a tool returns 'already_committed_stop', you already committed — STOP
   immediately, reply 'done'.
"""


def variant_agent_user(pdf1_record_json: str, pdf2_record_json: str) -> str:
    return f"""\
Classify the two ProductRecords below and commit your verdict.

PDF1:
{pdf1_record_json}

PDF2:
{pdf2_record_json}
"""


VARIANT_DETECTOR_SYSTEM = """\
Classify the relationship between two ProductRecords from one shipment.

Relationships:
  SAME_PRODUCT      identical models AND specs
  VARIANT           same family, minor delta (e.g. AM2 vs AM2-P1)
  DIFFERENT_FAMILY  different product lines, even if same factory
  OEM_SAME_FACTORY  same factory, distinct brand owners, distinct product lines

requires_human_choice = True for DIFFERENT_FAMILY / OEM_SAME_FACTORY (one
family must be picked for the Nepal draft). False for SAME_PRODUCT / VARIANT.

reasoning: 2-3 sentences. shared_attributes and distinguishing_attributes:
explicit lists.
"""


def variant_detector_user(pdf1_record_json: str, pdf2_record_json: str) -> str:
    return f"""\
Classify the relationship between PDF1 and PDF2 records below.

PDF1:
{pdf1_record_json}

PDF2:
{pdf2_record_json}
"""


PATCH_EXTRACTOR_SYSTEM = """\
Revise a previously-extracted ProductRecord to address critic flags.

RULES:
1. Return a complete ProductRecord — same family_label, same source_doc, same
   document_type, all fields present.
2. Non-flagged fields: copy FieldClaim VERBATIM (same value, source_doc,
   source_page, confidence). Do not paraphrase or relower confidence.
3. Flagged fields: re-read the source pages and either correct value, lower
   confidence honestly, or null if unsupported. source_page only from
   `=== page N ===` markers — never invent.
4. New entry for a missing flagged field only if you actually find it in the
   source pages; else null.
5. Null with low confidence beats fabricated value.
"""


def patch_extractor_user(
    record_json: str,
    flags_json: str,
    source_text: str,
) -> str:
    return f"""\
Re-extract ONLY the flagged fields. Return the COMPLETE patched ProductRecord;
copy non-flagged FieldClaims verbatim.

CURRENT RECORD:
{record_json}

CRITIC FLAGS:
{flags_json}

SOURCE PAGES:
{source_text}
"""


CRITIC_SYSTEM = """\
Strict compliance reviewer. Read the draft Nepal import compliance document
and flag:
  1. Any numeric claim (%, V, A, kg, years) not present in supplied source pages.
  2. Any FieldClaim with confidence < 0.7 stated as fact in the draft.
  3. Any NEPQA item marked covered with weak/indirect evidence.
  4. Anything important that is silently missing.

Each flag: section, short claim excerpt, one-sentence issue, concrete action.

Also produce ask_factory: 3-8 specific items SunBridge should request from the
manufacturer before final submission.
"""


DRAFTER_PROSE_SYSTEM = """\
Write FOUR short prose blocks for a Nepal import compliance draft (importer:
SunBridge Trading Pvt. Ltd., Kathmandu). Numeric tables + citations come from
a Python template — your job is plain-English synthesis for a Nepali customs
agent who is not an electrical engineer.

HARD RULES (violation → rejected):
1. Use ONLY values in the STRUCTURED INPUT. Do NOT invent numbers,
   percentages, cert numbers, page refs, model IDs, or dates.
2. No citation patterns: no "(source: pdfN p.K)", no "p.4", no "conf 0.95".
3. Plain prose only — no tables, lists, or headers.
4. Voice: terse, factual, professional. No hype, no filler.
5. Empty string OK if no data for a block. Fabrication is not.
6. Each block ≤ 6 sentences.

BLOCKS:
- cover_note: opens the doc, addressed to the Nepal import agent. State
  product chosen (family + manufacturer + factory), variant decision in one
  sentence, and where in the doc to find mismatches + items still to confirm.
  This is an agent-prepared draft for review, not a finished filing.
- methodology_note: 3-5 sentences on how the draft was assembled. Two
  manufacturer PDFs read; NEPQA 2025 used as indicative reference (not a
  section-by-section form); every claim cites source PDF + page; conflicts
  between PDFs surfaced (not silently merged). Addresses client ask "short
  note on how you approached it".
- gap_narrative: group partial/missing NEPQA items by theme (e.g. missing
  test reports, no nameplate photo, warranty not stated) and explain why
  each matters to Nepali customs clearance.
- mismatch_framing: explain why §6 cross-source differences matter for THIS
  shipment. If variant relationship is DIFFERENT_FAMILY or OEM_SAME_FACTORY,
  remind the reader only one product was chosen as the draft basis.
"""


def drafter_prose_user(structured_input_json: str) -> str:
    return f"""\
Synthesize the four prose blocks from the STRUCTURED INPUT below. No
fabricated numbers, no citation patterns, no tables — plain prose only.

--- STRUCTURED INPUT ---
{structured_input_json}
--- END ---
"""


def critic_user(draft_markdown: str, nepqa_text: str, evidence_json: str) -> str:
    return f"""\
DRAFT:
{draft_markdown}

NEPQA SOURCE (ground truth for threshold checks):
{nepqa_text}

EVIDENCE (every FieldClaim used in the draft):
{evidence_json}
"""
