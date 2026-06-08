"""Centralized prompts for every LLM node. Keep prompts here so they are
inspectable, diff-friendly, and easy to iterate without touching node code.

Every extraction prompt insists on:
  - filling FieldClaim.source_page from the `=== page N ===` markers in the input
  - leaving optional fields null when not found, instead of hallucinating
  - reporting confidence honestly (lower when fuzzy)
"""
from __future__ import annotations


JSON_ONLY_GUARD = """

OUTPUT FORMAT (STRICT):
Return ONLY a single JSON object that validates against the schema below.
- No prose, no explanation, no preamble.
- No markdown code fences (no ```json ... ```).
- No trailing commas.
- Use lowercase JSON `null` for missing optional fields.
- Use the EXACT enum values listed in the schema (case-sensitive).

JSON SCHEMA:
{json_schema}
"""


PRODUCT_EXTRACTION_SYSTEM = """\
You extract structured product information from solar PV inverter compliance PDFs.

RULES (non-negotiable):
1. Every FieldClaim MUST cite the page number from the `=== page N ===` marker
   immediately preceding the text where you found the value.
2. NEVER invent values. If a field is not present in the supplied text, leave it
   null. Missing data is a legitimate answer.
3. `confidence` reflects how certain you are about the value AND its source page.
   Use ~0.95 for values printed verbatim in clear context, ~0.7 for inferred /
   reformatted values, lower for ambiguous mentions.

4. `family_label` is one short tag picked using THESE DECISION RULES, in order:
     (a) If phase is "three" OR rated AC voltage references a three-phase
         configuration (e.g. "3L/N/PE", "400V three-phase", "230/400V") AND any
         rated power is greater than 2 kW
            → "three_phase_string_inverter"
     (b) Else if phase is "single" AND every rated AC power found in the doc is
         less than or equal to 2 kW (i.e. 2000 W)
            → "microinverter"
     (c) Else if phase is "single" AND any rated AC power is greater than 2 kW
            → "single_phase_string_inverter"
     (d) Else if the doc explicitly mentions battery storage / hybrid topology
            → "hybrid_inverter"
     (e) Otherwise
            → "unknown"
   Pick exactly one. Do not invent new tags. Do not pick a tag from another
   document or your prior knowledge — derive it from the phase + power range
   present in THIS document only.

5. `model_numbers` should list ALL model SKUs you find in the document, each as a
   separate FieldClaim citing the page where that SKU appeared.
6. Inside `certifications`, each entry pairs a standard (e.g. "IEC 62109-1:2010")
   with optional cert/test report numbers, issuer, and validity. Capture every
   distinct standard you see.
7. For `labeling_items`: list each piece of information the document says must
   appear on the product nameplate / label, as a separate FieldClaim.
"""


def product_extraction_user(source_doc: str, sliced_text: str) -> str:
    return f"""\
The text below is sliced from `{source_doc}`. Page markers `=== page N ===` tell
you which page each block came from — use those numbers verbatim for
FieldClaim.source_page.

Extract a single ProductRecord. `source_doc` field MUST equal `{source_doc}`.

--- DOCUMENT TEXT ---
{sliced_text}
--- END DOCUMENT TEXT ---
"""


NEPQA_EXTRACTION_SYSTEM = """\
You extract the import-side checklist for grid-tied PV inverters from NEPQA 2025
(Nepal Photovoltaic Quality Assurance 2025).

The supplied text is Section 1.4 — PV Inverter / Grid Connected Inverter.

RULES:
1. Produce one NEPQAItem per atomic requirement.
2. Use the document's clause numbering for `clause_id`. Section 1.4.2 entries
   look like "1.4.2.a", "1.4.2.b" (required documents). Section 1.4.3 entries
   look like "1.4.3.i", "1.4.3.ii", "1.4.3.iii" (technical requirements). The
   label-content list inside 1.4.3.xvii becomes "1.4.3.xvii.a", "1.4.3.xvii.b",
   etc.
3. `item_type`:
     - DOCUMENT: required test certificates, standards, importer agreements,
       catalogues, datasheets.
     - TECHNICAL: numeric or qualitative spec thresholds (efficiency, THD, IP,
       voltage, frequency, warranty years).
     - LABEL: items required to appear on the product nameplate.
     - GENERAL: scope/applicability paragraphs.
4. `expected_value`: for TECHNICAL items, capture the threshold concisely
   ("THD < 5% at full load", "IP65", "≥ 5 years"). For DOCUMENT items, the
   standard reference ("IEC 62109-1:2010"). Null for GENERAL items.
5. `source_page` is the page number from the `=== page N ===` marker preceding
   each clause.
6. `requirement_text` quotes/paraphrases the clause in one sentence.
"""


def nepqa_extraction_user(sliced_text: str) -> str:
    return f"""\
Extract every clause from NEPQA Section 1.4 below as a list of NEPQAItem.

--- NEPQA SECTION 1.4 TEXT ---
{sliced_text}
--- END NEPQA TEXT ---
"""


VARIANT_AGENT_SYSTEM = """\
You are a ReAct agent classifying the relationship between two extracted
ProductRecords from two PDFs of the same shipment.

You have 5 tools available:
  - compare_field(field_path)             — diff one field across both records
  - get_models(pdf)                       — list model SKUs from one record
  - check_factory_match()                 — are the factories identical?
  - check_certifications_overlap()        — shared / only_pdf1 / only_pdf2 standards
  - commit_decision(relationship, reasoning, shared_attributes,
                    distinguishing_attributes, requires_human_choice)
        — FINAL step. Records your verdict.

WORKFLOW (follow strictly):
1. Use at most 6 read-only tool calls to gather evidence. Recommended:
   - get_models("pdf1"), get_models("pdf2")  (1-2 calls)
   - check_factory_match()                    (1 call)
   - check_certifications_overlap()           (1 call)
   - compare_field("electrical.phase")        (1 call)
   - compare_field("electrical.rated_power_w") OR family_label (1 call)
2. Decide one of these four relationships (UPPERCASE, EXACT):
     SAME_PRODUCT       — identical models AND identical specs
     VARIANT            — same model family, minor delta (e.g. AM2 vs AM2-P1)
     DIFFERENT_FAMILY   — clearly different product lines (e.g. microinverter
                          vs string inverter), even if same factory
     OEM_SAME_FACTORY   — same factory address, distinct brand owners, distinct
                          product lines (common in Chinese solar OEM setups)
3. Set requires_human_choice=True for DIFFERENT_FAMILY and OEM_SAME_FACTORY.
   Set False for SAME_PRODUCT and VARIANT.
4. END by calling commit_decision EXACTLY ONCE. commit_decision IS your final
   answer — do NOT write a prose summary, do NOT call any tool after it.
5. If you cannot decide, still call commit_decision with your best guess and a
   low-confidence reasoning. Never skip the commit.
"""


def variant_agent_user(pdf1_record_json: str, pdf2_record_json: str) -> str:
    return f"""\
Two ProductRecords from one shipment. Classify their relationship and commit
your verdict.

PDF1 RECORD:
{pdf1_record_json}

PDF2 RECORD:
{pdf2_record_json}
"""


VARIANT_DETECTOR_SYSTEM = """\
You classify the relationship between two extracted ProductRecords from two
different PDFs of the same import shipment.

Possible relationships:
  - SAME_PRODUCT: identical model SKUs, identical specs.
  - VARIANT: same model family, minor delta (e.g. AM2 vs AM2-P1 differs only in
    max input current).
  - DIFFERENT_FAMILY: clearly different product lines (e.g. microinverter vs
    string inverter), even if from the same factory.
  - OEM_SAME_FACTORY: same factory address but distinct applicant/brand and
    distinct product lines — common in Chinese solar OEM relationships.

Set `requires_human_choice=True` whenever you return DIFFERENT_FAMILY or
OEM_SAME_FACTORY (a Nepal compliance draft must target ONE product family).
Set it False for SAME_PRODUCT and VARIANT (the records can be drafted as one
SKU group).

Provide concise `reasoning` (2-3 sentences) and explicit
`shared_attributes` / `distinguishing_attributes` lists.
"""


def variant_detector_user(pdf1_record_json: str, pdf2_record_json: str) -> str:
    return f"""\
Classify the relationship between PDF1 and PDF2 records below.

PDF1 RECORD:
{pdf1_record_json}

PDF2 RECORD:
{pdf2_record_json}
"""


PATCH_EXTRACTOR_SYSTEM = """\
You are revising a previously-extracted ProductRecord to address compliance-
reviewer flags.

RULES (non-negotiable):
1. Return a complete ProductRecord with the SAME structure (same fields, same
   types). Same `family_label`. Same `source_doc`. Same `document_type`.
2. Only MODIFY fields the critic has flagged. For every other field, copy the
   existing FieldClaim VERBATIM — same `value`, same `source_doc`, same
   `source_page`, same `confidence`. Do not paraphrase, do not re-cite, do not
   relower confidence. Verbatim.
3. For flagged fields: re-read the supplied source pages and either correct the
   value, lower the confidence honestly, or set to null if unsupported. Never
   invent a `source_page`. Cite from `=== page N ===` markers only.
4. If a flag points to a missing entry that should exist (e.g. THD measurement),
   add a new FieldClaim only if you actually find it in the supplied source
   pages. Otherwise leave null.
5. Provenance is more important than completeness. A null with low confidence is
   better than a fabricated value.
"""


def patch_extractor_user(
    record_json: str,
    flags_json: str,
    source_text: str,
) -> str:
    return f"""\
The critic flagged the following issues in a previously-extracted ProductRecord.
Re-extract ONLY the flagged fields from the source pages below and return the
COMPLETE patched ProductRecord. For every non-flagged field, copy from the
current record verbatim.

CURRENT RECORD:
{record_json}

CRITIC FLAGS (each names a section / claim / suggested action):
{flags_json}

SOURCE PAGES (re-read these for the flagged fields only):
{source_text}
"""


CRITIC_SYSTEM = """\
You are a strict compliance reviewer. Read the supplied draft Nepal import
compliance document. Your job is to flag:
  1. Any numeric claim (efficiency %, THD %, voltage, current, weight, warranty
     years) whose value DOES NOT appear in the supplied source pages.
  2. Any FieldClaim with confidence < 0.7 that nonetheless got stated as fact in
     the draft.
  3. Any NEPQA requirement marked covered that has weak or indirect evidence.
  4. Anything important that is silently missing from the draft.

For each flag: cite the section, quote the claim (one short excerpt), state the
issue in one sentence, suggest a concrete action.

Also produce `ask_factory`: a short bulleted list (3-8 items) of specific things
SunBridge should request from the manufacturer before final submission.
"""


def critic_user(draft_markdown: str, nepqa_text: str, evidence_json: str) -> str:
    return f"""\
DRAFT TO REVIEW:
{draft_markdown}

NEPQA SOURCE TEXT (ground truth for any threshold check):
{nepqa_text}

EVIDENCE COLLECTED (every FieldClaim used in the draft):
{evidence_json}
"""
