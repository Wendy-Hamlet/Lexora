# Anti-Hallucination: the Verbatim Contract

> **No canonical span, no claim.**

This document specifies what the LLM is allowed and forbidden to do, and how
the validator gates every claim before it becomes a public Citation.

## Allowed actions

The LLM, acting as a verifier inside the retrieval workflow, may:

- **Select** one or more clause IDs from the retrieved candidate set.
- **Assign** one RDTII indicator ID to a selected clause.
- **Produce a label**: `match` | `no_match` | `uncertain`.
- **Emit a confidence score** in `[0, 1]`.
- **Provide a short rationale** built from predefined fields (indicator
  definition, legal form, effective date) and exact source spans copied by
  the orchestrator. No free-form legal prose.
- **Abstain** — return `uncertain` rather than guess.

## Forbidden actions

The LLM **must not**:

- Invent legal text or paraphrase a cited quotation.
- Cite URLs, articles, pages or dates that are not present in the retrieved
  candidate metadata.
- Resolve legal hierarchy or precedence between conflicting instruments.
  (Sorting candidates as a transparency hint is fine; deciding which one
  governs is a human-reviewer responsibility.)
- Output free-form legal advice.

## Claim shape

The LLM returns claims in this exact structure (validated by Pydantic):

```json
{
  "clause_id": "sg.pdpa.s26",
  "label": "match",
  "confidence": 0.91,
  "rationale": "s26 conditions transfer abroad on comparable protection"
}
```

The indicator under test is fixed by the call, not chosen by the model. Notice
there is **no `quote` field**: quote text is copied by the orchestrator from
canonical storage using `clause_id` + the stored span offsets. The LLM literally
cannot produce quote text, so it cannot hallucinate a quote — by construction, not
by validation.

## Validation gates

Before any claim becomes a `Citation`, the validator confirms:

1. `indicator_id` exists in `configs/rdtii_indicators.yaml`.
2. `clause_id` resolves in the canonical store.
3. `quote_span_id` resolves and the stored span's text passes through
   `normalize()` and equals the canonical text under the same normalization.
   *Fuzzy matching is used only to tolerate OCR line-break drift, never to
   accept paraphrases.*
4. The page's OCR confidence is at or above `OCR_CITABLE_THRESHOLD` (default
   `0.85`). Otherwise → `LOW_OCR_CONFIDENCE`.
5. The source document's SHA-256 hash matches the stored snapshot. If the
   source has drifted, the citation is held for re-collection.
6. The source is a **primary instrument**, or is a secondary source that
   itself quotes and links to a primary instrument. Otherwise →
   `NO_PRIMARY_SOURCE_FOUND`.

Only claims that pass every gate are released. The rest are returned to the
reviewer with the gate-failure reason as `review_status`.

## A failure case the validator catches

The model selects a real article ("Article 17") and claims it proves
*"data shall be stored locally."* The article actually only says
*"the controller must keep records of processing activities."*

The orchestrator copies the canonical span for Article 17 → the text does not
contain the phrase the model implied. The claim is tagged
`HALLUCINATED_OR_UNSUPPORTED_MAPPING`, suppressed, and the reviewer sees
"manual review required" — never a false RDTII mapping.

## Prior-art reference

This pattern — model returns chunk IDs, orchestrator copies stored text —
mirrors the design used by Anthropic's [Citations API](https://docs.anthropic.com/en/docs/build-with-claude/citations)
in production legal systems such as Thomson Reuters CoCounsel.
