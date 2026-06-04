# Lexora

> A Verifiable AI System for Mapping Digital-Trade Regulations to the RDTII Framework.
>
> **No canonical span, no claim.**

Lexora is an open-source (Apache 2.0) pipeline that ingests national digital-trade
regulations across the Asia-Pacific and maps them onto the UN ESCAP **RDTII 2.1**
indicators, with minimum coverage of **Pillar 6 (Cross-Border Data Policies)** and
**Pillar 7 (Domestic Data Protection & Privacy)**.

Built for the [UN ESCAP × KMITL Global Hackathon on AI for Digital Trade Regulatory Analysis](https://unitescap.medium.com/global-hackathon-on-using-ai-for-digital-trade-regulatory-analysis-3c6213ddffa3) (2026).

## Why

Existing legal-AI tools fail on three fronts:

1. They bias toward EU/US common-law sources and miss APAC civil-law and hybrid systems.
2. They **paraphrase** rather than quote, producing citations that cannot be audited.
3. They are commercial closed-source tools that developing-country regulators cannot adopt.

Lexora is built around a single non-negotiable design rule: **every machine-emitted
claim is a verbatim quote bound to a specific document, page, and structural URI,
and is validated before it reaches a human.**

## Architecture — six stages

```
┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌────────┐   ┌─────────┐
│ collect  │ → │ extract  │ → │ structure  │ → │ classify │ → │  cite  │ → │ export  │
└──────────┘   └──────────┘   └────────────┘   └──────────┘   └────────┘   └─────────┘
   crawler       OCR/HTML/PDF   article tree    retrieval +     verbatim     JSON-LD,
   + hashes     + confidence    (no Akoma N.)   LLM verifier   quote check    CSV, UI
```

See [`docs/architecture.md`](docs/architecture.md) for the full design and
[`docs/anti_hallucination.md`](docs/anti_hallucination.md) for the verbatim contract.

## Status

**MVP under active development for Hackathon 2026.**

- [x] Repository scaffold + canonical data models
- [x] Jurisdiction profile schema + 3 Round 1 economy profiles (Singapore, Australia, Malaysia)
- [x] RDTII Pillar 6 + 7 indicator definitions (9 regulatory indicators, official codes P6-I1…P7-I5)
- [x] Citation validator (verbatim contract)
- [x] Slice 0 end-to-end pipeline (collect → … → cite → export) on a local PDF
- [x] Submission CSV matching the official OUTPUT_TEMPLATE schema
- [ ] Live portal crawling (mandatory for scoring)
- [ ] OCR pipeline + confidence triage
- [ ] Hybrid retrieval (BM25 + multilingual embeddings)
- [ ] LLM verifier (vLLM-served open weights)
- [ ] Review UI (side-by-side audit)
- [ ] Gold-set evaluation harness

Round 1 submission: 2026-07-20 · 20 shortlisted: 2026-07-31 · live e-pitch: 2026-08-03 · 5 finalists: 2026-08-05 · Bangkok finale: Oct 2026.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/Wendy-Hamlet/Lexora.git
cd Lexora
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest
lexora --help
```

## Project layout

```
Lexora/
├── src/lexora/             # Python package
│   ├── models/             # Pydantic data contracts (citation, clause, source, indicator)
│   ├── collect/            # Stage 1 — crawler, hashing, profile loader
│   ├── extract/            # Stage 2 — HTML / PDF text / OCR
│   ├── structure/          # Stage 3 — article/section/paragraph parser
│   ├── classify/           # Stage 4 — retrieval + constrained LLM verifier
│   ├── cite/               # Stage 5 — citation builder + verbatim validator
│   ├── export/             # Stage 6 — JSON-LD / CSV
│   ├── storage/            # SQLite/PostgreSQL + object store
│   └── api/                # FastAPI app
├── configs/
│   ├── jurisdictions/      # one YAML per economy (sg, au, my, …)
│   └── rdtii_indicators.yaml
├── docs/                   # architecture, memo, schemas
├── tests/
├── scripts/
└── data/                   # gitignored — raw + canonical document store
```

## Demo scope

Round 1 covers the **three mandatory economies** (Singapore, Australia, Malaysia)
and RDTII Pillars 6 and 7. Final-round economies (Thailand, China, India,
Indonesia, Russian Federation, Lao PDR, Mongolia, Timor-Leste) are added by
writing one YAML profile under `configs/jurisdictions/` — see
[`docs/jurisdiction_profile.md`](docs/jurisdiction_profile.md).

## License

Apache 2.0 — see [`LICENSE`](LICENSE). All required runtime dependencies are
permissively licensed.

## Team

Team **Verbatim Trade** — Bohan LIU, Chiheng JIN, Wenxiang SHI, Yiying TANG, Zijie Oscar WEI.
