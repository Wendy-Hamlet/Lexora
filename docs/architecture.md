# Architecture

Lexora is a six-stage pipeline. Each stage is a separate Python sub-package under
`src/lexora/` and can be developed, tested and replaced independently.

```
            ┌─────────────┐
   YAML ──▶│  1. collect │  crawler + SHA-256 + immutable raw store
            └──────┬──────┘
                   ▼
            ┌─────────────┐
            │  2. extract │  HTML DOM | PDF text-layer | OCR (page-level confidence)
            └──────┬──────┘
                   ▼
            ┌─────────────┐
            │ 3. structure│  article/section/paragraph paths
            └──────┬──────┘
                   ▼
            ┌─────────────┐
            │ 4. classify │  BM25 + multilingual embeddings → LLM verifier (or abstain)
            └──────┬──────┘
                   ▼
            ┌─────────────┐
            │   5. cite   │  build Citation from canonical span; validate quote equality
            └──────┬──────┘
                   ▼
            ┌─────────────┐
            │  6. export  │  JSON-LD + CSV + side-by-side audit UI
            └─────────────┘
```

## Design rules

These rules apply to every module. Violating them is a bug.

1. **No canonical span, no claim.** The LLM may select clause IDs and emit labels,
   but it never writes quote text. The orchestrator copies the quote from canonical
   storage using `char_start` / `char_end`. The validator confirms equality before
   release.
2. **Primary vs secondary sources.** Binding laws, regulations, gazettes and
   ratified treaties are *citable evidence*. Ministry guidelines and secondary
   datasets are *discovery context only* — never cited as binding evidence unless
   they reproduce and link to a primary instrument.
3. **Page-level OCR confidence.** Pages below threshold are tagged
   `UNVERIFIED_SCAN` and cannot back a citation until manually corrected.
4. **No automatic precedence between conflicting instruments.** The system shows
   all candidates side by side, optionally sorted by legal form and date as a
   review hint. A human reviewer decides which one governs.
5. **Reproducibility.** Every output is reproducible from pinned document
   snapshots (SHA-256), retrieval timestamps, model versions and prompt versions.

## Data flow

```
SourceProfile  ──▶  RawDocument  ──▶  Clause + CanonicalSpan
                                            │
                                            ▼
                                  EvidenceClaim  (LLM-emitted)
                                            │
                                            ▼  validator (verbatim check)
                                            ▼
                                       Citation
```

See [`citation_schema.md`](citation_schema.md) for the on-the-wire JSON shape and
[`anti_hallucination.md`](anti_hallucination.md) for the validator contract.

## Module ownership

| Stage     | Package                     | Primary surface                                  |
|-----------|-----------------------------|--------------------------------------------------|
| collect   | `lexora.collect`            | `crawl(profile) -> list[RawDocument]`            |
| extract   | `lexora.extract`            | `extract(raw) -> list[Page]`                     |
| structure | `lexora.structure`          | `parse(pages) -> list[Clause]`                   |
| classify  | `lexora.classify`           | `classify(clauses, indicator) -> EvidenceClaim`  |
| cite      | `lexora.cite`               | `build_citation(claim) -> Citation \| None`      |
| export    | `lexora.export`             | `to_jsonld / to_csv`                             |

The `lexora.api` package wraps the pipeline behind FastAPI for the audit UI.
The `lexora.storage` package provides DB and object-store helpers shared by all stages.
