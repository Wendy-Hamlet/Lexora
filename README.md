# Lexora — AI Tool for Digital Trade Regulatory Analysis

UN Global Hackathon on AI for Digital Trade Regulatory Analysis
Team: **Verbatim Trade** | Round: 1
Last updated: 2026-06-22

> A verifiable AI system for mapping digital-trade regulations to the UN ESCAP **RDTII 2.1** framework.
>
> **No canonical span, no claim.** — every machine-emitted claim is a verbatim quote bound to a specific document, page, and structural URI, and is validated before it reaches a human.

---

## What This Tool Does

Lexora automates the two tasks the RDTII requires for its **Step 1** (evidence
production). It does **not** assign scores — scoring is a downstream human step.

**Task 1 — Automated Evidence Discovery.** Given an economy and a regulatory topic,
Lexora crawls the official government legal portal (no URL is handed in), ranks the
candidates, fetches the full-text PDF/HTML (including scanned/image PDFs via OCR), and
extracts clean, structured, verbatim text.

**Task 2 — Intelligent Mapping & Citation.** The structured text is mapped to specific
RDTII indicator IDs. Each matched provision is recorded with an exact article-level
citation, a verbatim snippet, a `mapping_rationale`, and a Discovery Tag marking whether
it was found independently (**NEW**) or matched a known example (**KNOWN**).

**Mandatory scope:** Pillar 6 (Cross-Border Data Policies) and Pillar 7 (Domestic Data
Protection & Privacy) — 9 in-scope regulatory indicators (P6-I1…I4, P7-I1…I5; P6-I5 is a
non-regulatory indicator and is auto-excluded).
**Economies covered (Round 1):** Singapore, Australia, Malaysia.

---

## Quick Start

⚠ **Required for Round 1.** A reviewer with basic Python should be able to run this in
under 10 minutes.

```bash
# 1. Clone
git clone https://github.com/Wendy-Hamlet/Lexora.git
cd Lexora

# 2. Environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev,embeddings,ocr]"

# 3. Configure the LLM endpoint (any OpenAI-compatible server)
cp .env.example .env        # then edit .env — see "Swapping the LLM" below

# 4. Run the autonomous submission pipeline for one economy
LEXORA_LIVE=1 LEXORA_OCR=1 python scripts/run_submission.py -j sg --budget 20
```

**Output:** `outputs/submission_round1.csv` (official 13-column schema) +
`outputs/submission_round1.jsonld` + `outputs/submission_round1.summary.json`.

> **Always set `LEXORA_OCR=1` for real runs.** OCR is gated off by default; without it,
> scanned-only statutes (e.g. Malaysia's gazette PDFs) silently extract 0 clauses.

---

## Full Usage

```bash
# All three Round-1 economies into one official CSV (+ JSON-LD + summary)
LEXORA_LIVE=1 LEXORA_OCR=1 python scripts/run_submission.py -j all --budget 20 --secondary

# Faster (parallelism knobs; the safe fast config is --jobs 3 --llm-workers 8)
LEXORA_LIVE=1 LEXORA_OCR=1 python scripts/run_submission.py -j all --jobs 3 --llm-workers 8

# Optional LLM lanes (need an OpenAI-compatible endpoint configured in .env)
... --rationale-llm        # LLM-written mapping rationale (quote text still copied verbatim)
... --metadata-llm         # LLM-assisted Last-Amended / Law-Number metadata
... --verify-cells         # per-(clause×indicator) verifier (tightening only; currently over-strict, off by default)

# Inspect discovery only (search → rank, no mapping)
lexora discover -j sg

# Manual fallbacks (bypass the crawler)
lexora demo -j sg --url <pdf-or-html-url> --browser     # live-fetch one URL
lexora demo -j sg --pdf path/to/act.pdf --source-url <url>   # local PDF

# Evals
python scripts/eval_discovery.py     # anti-overfitting discovery eval
python scripts/eval_mapping.py       # section-level mapping hit@k (BM25 vs fused A/B)
```

Optional extras: `pip install -e ".[browser]" && playwright install chromium` (Singapore
SSO, which 403s plain HTTP). Network-touching tests are offline by default
(`httpx.MockTransport`); run the live ones with `LEXORA_LIVE=1 pytest -m live`.

---

## Architecture — six stages

```
┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌────────┐   ┌─────────┐
│ collect  │ → │ extract  │ → │ structure  │ → │ classify │ → │  cite  │ → │ export  │
└──────────┘   └──────────┘   └────────────┘   └──────────┘   └────────┘   └─────────┘
  crawl +        OCR/HTML/PDF   article/§ tree   BM25 retrieval  verbatim     CSV (13-col),
  discovery     + offsets       (schedule-aware) + LLM verifier  quote check  JSON-LD
```

| Stage | Package | Description |
| :---- | :---- | :---- |
| Collect | `src/lexora/collect/` | Portal crawler, autonomous discovery + ranking, per-portal strategies (SG SSO browser, AU OData, MY Fess), hashing; `collect/secondary/` = third-party trackers used only to *guide* discovery, never cited |
| Extract | `src/lexora/extract/` | HTML / PDF text / OCR (RapidOCR), verbatim char-offset invariant |
| Structure | `src/lexora/structure/` | Format-general legal parser (dotted SG/MY, spaced AU, civil-law articles), schedule-aware |
| Classify | `src/lexora/classify/` | Clause retrieval (BM25 default, dense opt-in) + constrained LLM verifier |
| Cite | `src/lexora/cite/` | Citation builder, verbatim validator, mapping rationale, metadata |
| Export | `src/lexora/export/` | Official 13-column CSV + JSON-LD |

See [`docs/architecture.md`](docs/architecture.md) and
[`docs/anti_hallucination.md`](docs/anti_hallucination.md) for the verbatim contract.

---

## Swapping the LLM (No Vendor Lock-in)

Lexora talks to **any OpenAI-compatible endpoint** — swap the model by changing `.env`
only, no code change. The LLM is optional: with no endpoint configured, the pipeline
degrades gracefully to BM25 + the verbatim validator.

```ini
# .env — OpenAI / vLLM / llama.cpp / Ollama all expose an OpenAI-compatible /v1
LEXORA_LLM_BASE_URL=http://localhost:11434/v1     # Ollama; or https://api.openai.com/v1
LEXORA_LLM_MODEL=llama3                            # or gpt-4o, Qwen2.5-7B-Instruct, …
LEXORA_LLM_API_KEY=not-needed-for-local            # real key for cloud
```

The client is abstracted in `src/lexora/classify/llm_client.py`.

## Swapping the OCR Engine

| Engine | Config | Notes |
| :---- | :---- | :---- |
| RapidOCR (ONNX) | `LEXORA_OCR_ENGINE=rapidocr` (default) | Bundled, CPU-fast, stable on Windows + py3.13 |
| PaddleOCR | `LEXORA_OCR_ENGINE=paddleocr` | GPU-server path (`paddlepaddle-gpu`) |

OCR is enabled with `LEXORA_OCR=1` and language-configured with `LEXORA_OCR_LANG`.

---

## Output Format

### CSV — official 13 columns, exact order (judges validate programmatically)

`economy`, `law_name`, `law_number_ref`, `last_amended`, `indicator_id`, `article`,
`discovery_tag`, `location_reference`, `verbatim_snippet`, `mapping_rationale`,
`source_url`, `confidence`, `notes`.

### JSON-LD

`run_submission.py` also emits a JSON-LD dump of every citation and a `.summary.json` run
report. *(The official 6-field technical JSON sidecar — `ocr_quality_cer`,
`processing_time`, `model_version`, `raw_context`, `provisions[]`, `source_pdf_path` — is
in progress; see Known Limitations.)*

---

## Project layout

```
Lexora/
├── src/lexora/
│   ├── models/        # Pydantic data contracts (citation, clause, source, indicator, secondary)
│   ├── collect/       # Stage 1 — crawler, discovery, per-portal strategies, secondary/ trackers
│   ├── extract/       # Stage 2 — HTML / PDF text / OCR
│   ├── structure/     # Stage 3 — legal parser
│   ├── classify/      # Stage 4 — retrieval + LLM verifier + boundary rules + lifecycle
│   ├── cite/          # Stage 5 — citation builder, validator, rationale, metadata
│   ├── export/        # Stage 6 — CSV / JSON-LD
│   ├── storage/ api/  # SQLite + object store; FastAPI app
│   └── config.py      # env-driven runtime config (LexoraConfig)
├── configs/
│   ├── jurisdictions/        # one YAML per economy (sg, au, my, _template)
│   ├── rdtii_indicators.yaml # 9 in-scope P6/P7 indicators
│   ├── secondary_sources.yaml
│   └── eval/                 # gold inventory, mapping gold, intrinsic parser fixtures
├── scripts/           # run_submission.py (Round-1 entry), run_pipeline.py, eval_*.py
├── docs/              # architecture, anti_hallucination, citation_schema, jurisdiction_profile
└── tests/
```

Add a new economy by writing one YAML under `configs/jurisdictions/` — see
[`docs/jurisdiction_profile.md`](docs/jurisdiction_profile.md).

---

## Known Limitations

Honest by design — these guide where to be cautious.

- **OCR is opt-in:** real runs **must** set `LEXORA_OCR=1`, or scanned-only statutes drop
  to 0 clauses with no flag. Submission runs set it explicitly.
- **Official 6-field JSON sidecar:** CSV + JSON-LD ship now; the official technical JSON
  copy (CER, processing time, raw context, …) is being finalized.
- **Per-cell LLM verifier (`--verify-cells`):** kills the "empty-attribution law scored
  against all 9 indicators" false positives, but is currently over-strict (drops some
  correct cells) — **off by default**, pending calibration.
- **Australia portal anti-bot:** `legislation.gov.au` applies cumulative per-IP throttling
  and can serve an HTML decoy under load; mitigated with per-host throttle + `--serial-fetch`.
- **Confidence is relative, not calibrated:** treat scores below 0.80 as review-flagged.
- **Secondary sources are never citable:** third-party trackers only *guide* discovery
  (USE 1/2/3); they never become a citation.

---

## Running the Test Suite

```bash
pytest                       # offline; ~325 tests
LEXORA_LIVE=1 pytest -m live # live portal tests
```

---

## Team

Team **Verbatim Trade** — Bohan LIU, Chiheng JIN, Wenxiang SHI, Yiying TANG, Zijie Oscar WEI.

## License

Apache 2.0 — see [`LICENSE`](LICENSE). All required runtime dependencies are permissively
licensed.

## Key Dates

| Date | Milestone |
| :---- | :---- |
| **20 July 2026** | **Round 1 submission deadline** |
| 31 July 2026 | 20 teams shortlisted |
| 3 August 2026 | Live online pitch |
| 5 August 2026 | 5 finalists announced |
| October 2026 | Grand Finale — Bangkok |

Built for the [UN ESCAP × KMITL Global Hackathon on AI for Digital Trade Regulatory Analysis](https://unitescap.medium.com/global-hackathon-on-using-ai-for-digital-trade-regulatory-analysis-3c6213ddffa3) (2026).
