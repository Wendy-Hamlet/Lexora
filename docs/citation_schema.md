# Citation Schema

Every Lexora output is a `Citation` object. A claim that cannot be turned into
a valid `Citation` is suppressed and surfaced to the human reviewer instead of
being released.

The `Citation` field set is a **superset** of the official submission columns
(OUTPUT_TEMPLATE_31MAY.xlsx) plus the provenance fields needed for the verbatim
audit trail. The submission CSV projects it onto the exact 13-column template
(see "Export formats").

## JSON shape (on the wire)

```json
{
  "economy": "Singapore",
  "title": "Personal Data Protection Act 2012",
  "law_number": "Act 26 of 2012",
  "last_amended": "2021",
  "indicator_id": "P6-I4",
  "article_path": "S. 26(1)",
  "discovery_tag": "KNOWN",
  "page_or_dom_anchor": "section-26",
  "quote": "An organisation must not transfer any personal data to a country or territory outside Singapore except in accordance with requirements prescribed under this Act ...",
  "mapping_rationale": "S.26 permits transfer only subject to prescribed safeguards → conditional flow regime (P6-I4).",
  "source_url": "https://sso.agc.gov.sg/Act/PDPA2012",
  "confidence": 0.92,
  "notes": "",
  "clause_id": "sg.pdpa.s26",
  "retrieval_timestamp": "2026-06-12T08:14:23Z",
  "document_hash": "sha256:7f3a...",
  "jurisdiction": "SG",
  "legal_form": "statute",
  "coverage": "Horizontal",
  "char_start": 14820,
  "char_end": 15422,
  "review_status": "VERIFIED"
}
```

## Field-by-field

Official submission columns (required ones marked ✓):

| Field                  | CSV column          | Source                       | Notes |
|------------------------|---------------------|------------------------------|-------|
| `economy`           ✓  | Economy             | SourceProfile                | Official UN economy name |
| `title`             ✓  | Law Name            | RawDocument metadata         | Full official statute name + year |
| `law_number`           | Law Number / Ref    | metadata                     | e.g. `Act 709`, `No. 9 of 2018` |
| `last_amended`      ✓  | Last Amended        | metadata                     | Year; blank if not amended |
| `indicator_id`      ✓  | Indicator ID        | LLM (submission code)        | e.g. `P6-I4`; must exist in `configs/rdtii_indicators.yaml` |
| `article_path`      ✓  | Article / Section   | Structure parser             | e.g. `S. 26(1)` |
| `discovery_tag`     ✓  | Discovery Tag       | Orchestrator                 | `NEW` / `KNOWN` (NEW = 20 of 40 accuracy points) |
| `page_or_dom_anchor`   | Location Reference  | Extractor                    | PDF page or HTML anchor |
| `quote`             ✓  | Verbatim Snippet    | **Orchestrator (NOT LLM)**   | Copied from canonical span using offsets |
| `mapping_rationale`    | Mapping Rationale   | LLM verifier                 | ≤300 chars |
| `source_url`        ✓  | Source URL          | RawDocument                  | Direct URL on the official portal |
| `confidence`           | Confidence          | LLM verifier                 | 0..1 |
| `notes`                | Notes               | any stage                    | OCR/bilingual flags |

Provenance / audit fields (JSON + audit CSV, not in the submission CSV):
`clause_id`, `retrieval_timestamp`, `document_hash`, `jurisdiction`,
`legal_form`, `coverage` (`Horizontal`/`Sectoral`), `char_start`/`char_end`,
`review_status`.

## `review_status` values

| Value                            | Meaning |
|----------------------------------|---------|
| `VERIFIED`                       | All gates passed; safe to publish. |
| `LOW_OCR_CONFIDENCE`             | The span is on a page below the OCR citable threshold. Show to reviewer; do not auto-publish. |
| `CONFLICT_REVIEW`                | Multiple candidate clauses across primary instruments disagree. All are returned; the reviewer decides precedence. |
| `NO_PRIMARY_SOURCE_FOUND`        | The verifier only found secondary sources (guidelines, datasets) for this indicator. Not citable. |
| `HALLUCINATED_OR_UNSUPPORTED_MAPPING` | The quote does not equal the canonical span after normalization, OR the selected clause does not support the indicator. Suppressed; logged for evaluation. |

## Export formats

- **Submission CSV** (`lexora.export.csv_exporter.to_csv`) — the exact official
  13-column OUTPUT_TEMPLATE schema (column names and order must not change).
  Written as UTF-8 with BOM so it opens cleanly in Excel for policy judges.
- **Audit CSV** (`lexora.export.csv_exporter.to_audit_csv`) — all provenance
  columns for internal review.
- **JSON-LD** (`lexora.export.jsonld_exporter`) — richer metadata for the
  technical judge / programmatic consumers.
- **Audit UI** (`lexora.api`) — side-by-side view: original PDF/HTML on the left,
  extracted text with highlighted span on the right.
