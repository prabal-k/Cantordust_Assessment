# Nepal Import Compliance Drafter

A agent that reads two messy factory PDFs and Nepal's solar regulator checklist, then writes a draft compliance file the importer's customs agent can actually review.

Built as my Task 1 submission for Cantordust's AI Engineer take-home.

## The problem in one paragraph

A Kathmandu trader called SunBridge wants to import grid-tied solar inverters from China. The factory sends some paperwork, but it's written for China — different layouts, different terminology, sometimes for two different products mixed together. Nepal expects the inverter to be tested to a specific NEPQA 2025 checklist. Someone has to sit down, read everything, line it up against Nepal's rules, and produce one clean draft to hand to the import agent. That's tedious and easy to get wrong by hand. So I wrote an agent to do it.

The interesting twist, the two factory PDFs describe **two different products** from the same factory in Ningbo. PDF1 is a small single-phase Chisage microinverter line. PDF2 is the big three-phase Deye string-inverter line. Merging facts from both into one draft would be wrong and the import agent would catch it. So the agent has to detect the mismatch, ask which product the shipment is actually about, and only then write the draft.

## What it does

- Reads the three PDFs (`PyMuPDF`, page-indexed)
- Extracts a typed `ProductRecord` from each manufacturer doc using an LLM + Pydantic schema
- Pulls the NEPQA Section 1.4 (PV Inverter) checklist out of the regulator PDF the same way
- A **ReAct agent** classifies the relationship between the two records using 4 read-only tools and a terminal `commit_decision` tool
- If the records describe different families, the user picks which one the shipment is about (CLI prompt, or Streamlit radio)
- Reconciles the two records field-by-field, maps the chosen one to NEPQA, drafts the final markdown + PDF
- A critic node re-reads the draft and flags anything unsupported; if it finds flags, a `patch_extractor` node re-extracts only the flagged fields and the loop runs again (configurable retry budget, default 2)
- Every value in the final draft cites its source PDF and page

## Architecture

![graph](docs/graph.png)

The system is a multi-agent LangGraph workflow with 8 LLM-driven agents and 5 deterministic glue nodes. Three **extraction agents** read each PDF independently and return Pydantic records where every field cites its source page. A **ReAct variant-detector agent** then reasons over both records using five tools and either commits a verdict on its own or hands off to a **human-in-the-loop node** when it is uncertain or when a Python sanity check overrides it.

A **hybrid drafter agent** writes the compliance file — numeric tables and citations from a deterministic template, four short prose blocks (cover note, methodology, gap narrative, mismatch framing) from a constrained LLM call (temperature 0, structured output, post-call guard that rejects fabricated numbers). A **critic agent** then re-reads the draft against NEPQA source pages and flags anything unsupported, a **patch-extractor agent** re-extracts only the flagged fields, and the drafter runs again — up to `max_retries` cycles, with a best-attempt safeguard so a later iteration cannot regress the final draft. A final drafter pass after the loop ensures the critic's ask-the-factory list lands in §8 of the file on disk.

## Setup

Python 3.11 or higher.

```powershell
git clone <this-repo>
cd Cantordust_Assessment

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# Open .env, paste your GEMINI_API_KEY (free at https://aistudio.google.com/apikey)
# OR set LLM_PROVIDER=groq and paste GROQ_API_KEY (also free)
```

If `pip install` chokes on WeasyPrint on Windows, just comment that line out — the agent will skip PDF rendering and still write the markdown draft. WeasyPrint needs the GTK3 runtime on Windows which is more pain than it's worth for a demo.

### Optional: LangSmith tracing

If you want to inspect every LLM call after a run:

1. Make an account at https://smith.langchain.com, grab an API key
2. In `.env`, uncomment the three `LANGCHAIN_*` lines and fill in the key
3. Restart Streamlit — the sidebar caption will flip to **🔵 LangSmith tracing ON**

Disabled by default, so a reviewer without a LangSmith account isn't affected.

## Running it

CLI:

```powershell
python run.py `
  --pdf1 data/DSS_GZES230100125901_combined-1.pdf `
  --pdf2 data/188_1115.pdf `
  --nepqa data/nepqa_2025.pdf `
  --retries 2
```

Streamlit:

```powershell
streamlit run app.py
```

Both write into `outputs/`:

- `compliance_draft_<timestamp>.md` — the actual draft
- `compliance_draft_<timestamp>.pdf` — the rendered PDF if WeasyPrint is installed
- `agent_state_<timestamp>.json` — full state dump for audit

The Streamlit version is the one to demo. It streams tokens live into each node card, shows the variant decision, asks for human input only when needed, and shows the critic's retry loop play out.

## Tests

```powershell
pytest tests/
```

There are 42 tests covering schemas, reconciler severity logic, NEPQA coverage classification, the patch-extractor retry counter, all five variant-detector tools, the ReAct agent's commit path and fallback, the hybrid drafter's prose injection + digit-leak guard (rejection of fabricated numbers + fallback to empty prose), and the submission-grade template structure (10 sections, doc-control header, ask-factory §8, sign-off block, NEPQA-as-indicative-reference disclaimer).

## Sample report output

A complete sample run is committed to the repo so a reviewer can read the
final draft without needing API keys or executing the pipeline.

### Final draft (the file the customs agent reads)

The drafter writes a stable-named copy on every run. Whatever the latest
pipeline produced, the canonical "final draft" is always at this path:

- **[`outputs/compliance_draft_for_human_agent.pdf`](outputs/compliance_draft_for_human_agent.pdf)** — final PDF, ready to share with the Nepal import agent
- [`outputs/compliance_draft_for_human_agent.md`](outputs/compliance_draft_for_human_agent.md) — markdown source

### Timestamped sample (preserved for the reviewer)

A single representative run is also kept at a timestamp-free name so you can
compare it against a fresh run:

- [`outputs/compliance_draft_sample.md`](outputs/compliance_draft_sample.md)
- [`outputs/compliance_draft_sample.pdf`](outputs/compliance_draft_sample.pdf)

The sample shows the full 10-section structure: doc-control header, cover note,
how-this-was-assembled, product + variant identification, manufacturer + factory,
specifications, certifications, labeling, cross-source consistency (with the
critical microinverter vs string-inverter mismatch surfaced honestly), the NEPQA
2025 coverage matrix, the ask-the-factory list in §8, and the drafter
limitations + sign-off block.

## Demo

Video walkthrough
