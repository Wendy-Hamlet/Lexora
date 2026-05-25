# Citation Schema

Every Lexora output is a `Citation` object. A claim that cannot be turned into
a valid `Citation` is suppressed and surfaced to the human reviewer instead of
being released.

## JSON shape (on the wire)

```json
{
  "indicator_id": "6.1",
  "clause_id": "sg.pdpa.s26",
  "source_url": "https://sso.agc.gov.sg/Act/PDPA2012",
  "retrieval_timestamp": "2026-06-12T08:14:23Z",
  "document_hash": "sha256:7f3a...",
  "title": "Personal Data Protection Act 2012",
  "jurisdiction": "SG",
  "legal_form": "statute",
  "article_path": "Part VI > Section 26",
  "page_or_dom_anchor": "section-26",
  "char_start": 14820,
  "char_end": 15422,
  "quote": "An organisation shall not transfer any personal data to a country or territory outside Singapore except in accordance with requirements prescribed under this Act ...",
  "confidence": 0.92,
  "review_status": "VERIFIED"
}
```

## Field-by-field

| Field                  | Type     | Source                            | Notes |
|------------------------|----------|-----------------------------------|-------|
| `indicator_id`         | str      | LLM                               | Must exist in `configs/rdtii_indicators.yaml` |
| `clause_id`            | str      | LLM                               | Must exist in the canonical store |
| `source_url`           | URL      | RawDocument                       | Snapshot URL at retrieval time |
| `retrieval_timestamp` | ISO 8601 | RawDocument                       | UTC |
| `document_hash`       | str      | RawDocument                       | `sha256:<hex>` of raw bytes |
| `title`               | str      | RawDocument metadata              |       |
| `jurisdiction`        | str      | SourceProfile                     | ISO code |
| `legal_form`          | str      | Structure parser                  | `statute`, `regulation`, `gazette`, `treaty`, `guideline` |
| `article_path`        | str      | Structure parser                  | Human-readable structural path |
| `page_or_dom_anchor`  | str      | Extractor                         | Page number (PDF) or DOM selector (HTML) |
| `char_start` / `char_end` | int  | Extractor                         | Character offsets inside canonical text |
| `quote`               | str      | **Orchestrator (NOT LLM)**        | Copied from canonical span using offsets |
| `confidence`          | float    | LLM verifier                      | 0..1 |
| `review_status`       | enum     | Validator                         | See below |

## `review_status` values

| Value                            | Meaning |
|----------------------------------|---------|
| `VERIFIED`                       | All gates passed; safe to publish. |
| `LOW_OCR_CONFIDENCE`             | The span is on a page below the OCR citable threshold. Show to reviewer; do not auto-publish. |
| `CONFLICT_REVIEW`                | Multiple candidate clauses across primary instruments disagree. All are returned; the reviewer decides precedence. |
| `NO_PRIMARY_SOURCE_FOUND`        | The verifier only found secondary sources (guidelines, datasets) for this indicator. Not citable. |
| `HALLUCINATED_OR_UNSUPPORTED_MAPPING` | The quote does not equal the canonical span after normalization, OR the selected clause does not support the indicator. Suppressed; logged for evaluation. |

## Export formats

- **JSON-LD** (`lexora.export.jsonld_exporter`) — for programmatic consumers.
- **CSV** (`lexora.export.csv_exporter`) — flat one-row-per-citation for analysts.
- **Audit UI** (`lexora.api`) — side-by-side view: original PDF/HTML on the left,
  extracted text with highlighted span on the right.
