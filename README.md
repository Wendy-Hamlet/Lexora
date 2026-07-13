# Lexora — AI Tool for Digital Trade Regulatory Analysis

UN Global Hackathon on AI for Digital Trade Regulatory Analysis
Team: **Verbatim Trade** | Round: 1
Last updated: 2026-07-13

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
under 10 minutes, using only the steps below.

```bash
# 1. Clone
git clone https://github.com/Wendy-Hamlet/Lexora.git
cd Lexora

# 2. Environment (Python 3.10+; verified on 3.12)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # pinned; includes the LLM client AND the OCR stack
playwright install chromium       # Singapore's SSO portal 403s a plain HTTP client

# 3. Configure the LLM endpoint (any OpenAI-compatible server)
cp .env.example .env              # then edit .env — see "Swapping the LLM" below

# 4. Run it — one command, no manual steps
python main.py --economy Singapore --pillar 6
```

**Output** (both files, always):

```
outputs/Singapore_P6_<timestamp>.csv     official 13 columns, one row per provision
outputs/Singapore_P6_<timestamp>.json    same rows + OCR / timing / context metadata
```

**No API key?** Add `--no-llm`. The engine still crawls, OCRs, parses, maps (BM25) and
writes both files — you get complete output with template rationales, so you can verify
the pipeline end-to-end before configuring any endpoint.

The economy argument is forgiving: `Singapore`, `SG`, `sg` and even `Singapre` all
resolve (fuzzy-matched, with a note); an unrecognisable one exits with the supported list
rather than a stack trace.

---

## Full Usage

```bash
python main.py \
  --economy "Malaysia" \
  --pillar 6 \              # 6, 7, or all (default)
  --output-dir outputs/ \
  --budget 20 \             # max instruments to map
  --llm-workers 8           # parallel LLM calls
```

### All three Round-1 economies in one file

```bash
LEXORA_LIVE=1 LEXORA_OCR=1 python scripts/run_submission.py -j all --budget 20 \
    --verify-clauses --rationale-llm --metadata-llm --amendment-llm --llm-workers 16
```

| Flag | What it adds |
| :---- | :---- |
| `--verify-clauses` | **The relevance decision.** Per-clause 9-in-1 LLM judge: each candidate clause is judged yes/no against *all* indicators at once. This is what `main.py` uses by default. |
| `--verify-cells` | Older per-(clause × indicator) verifier. Tightening only. |
| `--rationale-llm` | LLM-written Mapping Rationale (the quote itself is still copied verbatim from the parsed clause, never generated). |
| `--metadata-llm` | LLM-assisted Law Number / Last Amended extraction, source-verified. |
| `--amendment-llm` | Amendment-currency adjudication (flags provisions whose cited law was since amended). |
| `--secondary` | Use third-party trackers to *guide* discovery. They are never cited. |

### Other entry points

```bash
lexora discover -j sg                                        # discovery only (search → rank)
lexora demo -j sg --url <pdf-or-html-url> --browser          # map one live URL
lexora demo -j sg --pdf path/to/act.pdf --source-url <url>   # map a local PDF (bypass crawler)

python scripts/eval_discovery.py    # discovery eval vs the official legal inventory
python scripts/eval_mapping.py      # section-level mapping eval vs the provision gold
```

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

Every run writes **both** deliverable files, named `<Economy>_P<pillar>_<timestamp>`.

### CSV — official 13 columns, exact order

Header text and order match `OUTPUT_TEMPLATE_31MAY.xlsx` byte for byte (judges validate
programmatically — do not rename or reorder).

| # | Column | Required | Notes |
| :---- | :---- | :---- | :---- |
| 1 | `Economy` | Required | Official UN country name |
| 2 | `Law Name` | Required | Full official statute name + year |
| 3 | `Law Number / Ref` | Optional | e.g. `Act 709` |
| 4 | `Last Amended` | Required | Year; blank if never amended |
| 5 | `Indicator ID` | Required | RDTII code, e.g. `P6-I4` |
| 6 | `Article / Section` | Required | Article **and** paragraph, e.g. `s. 26(1)` |
| 7 | `Discovery Tag` | Required | `NEW` = found independently; `KNOWN` = in the sample kit |
| 8 | `Location Reference` | Optional | PDF page no. \| HTML anchor |
| 9 | `Verbatim Snippet` | Required | Copied from the parsed clause — never model-generated |
| 10 | `Mapping Rationale` | Optional | Why it maps, max 300 chars |
| 11 | `Source URL` | Required | Direct link on the official portal |
| 12 | `Confidence` | Optional | 0.00–1.00 |
| 13 | `Notes` | Optional | OCR issues, bilingual source, cross-references |

### JSON — same rows, richer metadata

One flat object per provision (JSON rows == CSV rows), carrying what CSV cannot hold:
`source_pdf_path`, `pdf_is_scanned`, `ocr_quality_cer`, `processing_time_seconds`,
`model_version`, `retrieval_method`, `raw_context_before` / `raw_context_after`.

`scripts/run_submission.py` additionally emits a JSON-LD dump and a `.summary.json` run
report (instruments discovered, NEW/KNOWN split, indicators covered).

---

## Actual Cost Per Document

*Required by the hackathon rubric for UN sustainability assessment. **Measured from a real
run, not estimated** — token counts come from the live LLM client accounting, page counts
from the parsed PDF, time from a wall-clock around the run.*

Our stack is almost entirely **self-hosted**, so its marginal API cost is ~$0; the LLM is
the only metered API. Reproduce on any local PDF:

```bash
PYTHONPATH=src python tools/cost_logger.py \
  --pdf data/raw/my/<hash>.pdf --economy my --pillar 6
# writes logs/cost_report.json + prints the table below
```

### Measured results

| Component | Engine used | Measured cost |
| :---- | :---- | :---- |
| OCR | RapidOCR (bundled ONNX, local CPU) | $0.0000 |
| Embedding | BAAI/bge-m3 (local; dense channel off by default) | $0.0000 |
| LLM mapping | gpt-5.4 (metadata + rationale + per-cell verify) | $0.0179 |
| Crawling | self-hosted (requests / browser) | $0.0000 |
| **Total (current stack)** |  | **$0.018 per document** |
| **Total (open-weight swap)** | Llama-class + Tesseract (all self-hosted) | **$0.000 per document** |

**Measured on:** 2026-06-22 · **Benchmark document:** Malaysia PDPA 2010 (Act 709), 100 pages,
~149k characters, scanned (OCR ran on all 100 pages) · **Token counts:** Input 24,548 · Output
3,722 (27 LLM calls) · **Wall-clock:** 150.3 s per document.

Only the per-token **rate** is a parameter — measured tokens × rate. The figure above prices
the LLM at a reference commercial rate ($0.50 / $1.50 per 1M input/output tokens); **our own
endpoint is not billed per token** (500M tokens/day, no per-call charge), so our *actual*
marginal cost is ~$0. Everything except the LLM is self-hosted, so the open-weight swap is
**$0 API per document** (compute only).

### Cost log excerpt (`logs/cost_report.json`)

```json
{
  "document": "MY_PDPA_Act709_100pages.pdf",
  "measured_on": "2026-06-22",
  "pages": 100,
  "ocr":       { "engine": "rapidocr",   "pages": 100, "scanned": true, "cost_usd": 0.0 },
  "embedding": { "model": "BAAI/bge-m3", "tokens": 0, "cost_usd": 0.0 },
  "llm":       { "model": "gpt-5.4", "calls": 27, "input_tokens": 24548,
                 "output_tokens": 3722, "cost_usd": 0.0179 },
  "total_cost_usd": 0.0179,
  "total_cost_usd_open_weight_swap": 0.0,
  "citations": 9,
  "processing_time_seconds": 150.3
}
```

---

## Project layout

```
Lexora/
├── main.py            # ← reviewer entry point: --economy <name> --pillar <6|7|all>
├── requirements.txt   # pinned runtime deps (engine + LLM client + OCR + browser)
├── .env.example       # copy to .env; LLM endpoint + feature switches
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

- **Recall is bounded by what the judge sees, not by the parser.** Retrieval pools the top
  `LEXORA_MAP_POOL_K` (default 40) clauses per indicator and the LLM judge decides
  membership over that pool. A relevant provision ranked below the pool is never judged.
  Measured retrieval ceiling on our flagship gold: pool 3 → 38%, 20 → 76%, 40 → 95%.
- **Confidence is relative, not calibrated:** it is a normalised retrieval score, not a
  probability. Treat < 0.80 as review-flagged.
- **Delegated legislation:** the engine retrieves principal statutes and discovers
  amendments, but does not exhaustively follow cross-references into subordinate
  regulations.
- **Australia portal anti-bot:** `legislation.gov.au` applies cumulative per-IP throttling
  and can serve an HTML decoy under load; mitigated with a per-host throttle and
  `--serial-fetch`.
- **Bilingual corpora:** Malaysia's portal mixes English and Malay; non-English provisions
  in a mixed PDF may be missed. Non-Latin scripts (CJK, Thai) are tokenised but not yet
  validated against gold.
- **Secondary sources are never citable:** third-party trackers only *guide* discovery;
  they never become a citation.
- **Windows, non-ASCII paths:** `pip` decodes `requirements.txt` with the *locale* codec,
  so we keep that file pure ASCII (a stray em dash makes `pip install -r` die with
  `UnicodeDecodeError` on a cp936/cp932 box). An editable install (`pip install -e .`) also
  breaks under a non-ASCII home directory, because the `.pth` file is written UTF-8 and read
  back as cp936. The Quick Start path (`pip install -r requirements.txt`) is unaffected.

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
