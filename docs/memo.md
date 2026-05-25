# Technical Memo (submitted)

> Verbatim copy of the team's submitted Technical Memo for Hackathon 2026.
> Authoritative project specification is in [`architecture.md`](architecture.md)
> and [`anti_hallucination.md`](anti_hallucination.md), which may incorporate
> further refinements agreed after this Memo was written. Where the two
> conflict, the code follows the architecture docs; track open discrepancies
> in the repo issues.

---

## Technical Memo for RegTech Hackathon 2026

by Bohan LIU, Chiheng JIN, Wenxiang SHI, Yiying TANG, Zijie Oscar WEI
from Team **Verbatim Trade**, with Project **Lexora**.

A Verifiable AI System for Mapping Digital-Trade Regulations to the RDTII Framework.

### Tech-Problem & Approach

National regulators, exporters, and trade negotiators across the Asia-Pacific
need to compare digital-trade rules across jurisdictions, but the underlying
texts are scattered across dozens of portals, written in many languages, and
locked in formats that range from clean HTML to scanned PDFs. Existing AI
legal tools fall short on three fronts: (i) they bias toward EU/US common-law
sources and miss APAC civil-law and hybrid systems; (ii) they paraphrase
rather than quote, producing citations that cannot be audited; (iii) they are
commercial closed-source tools that developing-country regulators cannot adopt.

LEXORA is an end-to-end open-source pipeline (Apache 2.0) that automates the
mapping of national regulations to the RDTII 2.1 framework, with minimum
coverage of Pillar 6 (Cross-Border Data Policies) and Pillar 7 (Domestic Data
Protection & Privacy). The system is built around a single non-negotiable
design rule: every machine-emitted claim is a verbatim quote bound to a
specific document, page, and structural URI, validated before it reaches a
human.

### System Architecture

The pipeline has six stages — collect → extract → classify → explain → cite
→ export — and a jurisdiction-authority context that the classifier consults
for every mapping decision.

### Anti-Hallucination — the Lexora Contract

The LLM is constrained to four behaviours: quote, label, score, or abstain.
It is structurally prevented from generating legal text, citing material
outside the retrieval set, or returning free-form prose. Every emitted
snippet is checked against the canonical text before release; any failure
discards the claim and logs an HALLUCINATED_QUOTE event. A calibrated
confidence threshold gates low-evidence cases into manual review. Document
snapshots, model versions, and prompts are pinned per run, so any citation
can be replayed and contested by an auditor.

### Cross-Border Adaptation

A new jurisdiction is onboarded by writing one YAML profile (portals,
language, legal-system family, treaty-integration mode) — no code change.
BGE-M3 provides native multilingual retrieval across English, Chinese,
Japanese, Korean, Thai, Vietnamese, and other APAC languages, removing
fragile MT pipelines. The canonical citation schema means downstream
reasoning is format-agnostic. Self-hosting via vLLM lets
data-sovereignty-sensitive regulators run the full stack on local hardware.

### Accuracy, Transparency, Cost

**Accuracy.** Two-stage retrieval (dense + reranker) plus an authority-aware
re-rank improves top-1 indicator selection over single-stage RAG; the
canonical-span validator drives source-attribution accuracy toward zero
hallucinated quotes by construction.

**Transparency.** Every output is reproducible from pinned snapshots and is
presented in a side-by-side audit view.

**Cost.** All defaults are open-weight: BGE-M3 (free), Qwen-2.5/Llama-3
served via vLLM, MinerU/Docling for OCR. Optional commercial APIs are
pluggable but never required.

### Roadmap & Risks

**Risks & mitigations.** (i) OCR drift on poorly scanned gazettes —
confidence-gated triage + OCR_UNVERIFIED quarantine; (ii) legal-structure
ambiguity — fallback heading detection with manual-review flagging, never
silent merging; (iii) cross-source conflicts — candidates surfaced under a
CONFLICT tag for human review rather than a unilateral pick; (iv) scope creep
beyond Pillars 6 and 7 — other potential Pillars are gated behind the
indicator definition library and only enabled once Pillar 6 or 7 hit the eval
bar.

---

## Appendix — Memo vs. Code (known discrepancies)

The narrative above was finalized in a parallel track to the code. After the
supervisor's review and a subsequent tech-team consolidation, four design
points in the Memo were superseded. The repository implements the newer
design; the table below records the deltas for transparency.

| Topic                | Memo description                                                  | Code implementation                                                                                                  |
|----------------------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| Quote validation     | "byte level / UTF-8 sub-string match"                             | NFKC + whitespace normalization; tolerates OCR line-break drift, rejects paraphrases (`src/lexora/cite/validator.py`) |
| Document schema      | "canonical XML / Akoma Ntoso schema"                              | Smaller canonical citation schema; Akoma Ntoso kept as an optional future export only                                |
| Conflict resolution  | "jurisdiction-authority graph + recency rules"                    | No automatic precedence; system shows all candidates, human reviewer decides                                         |
| OCR                  | "dual-engine cross-check"                                         | Page-level confidence triage + `UNVERIFIED_SCAN` quarantine; re-OCR once on medium-confidence pages                  |

Additionally, the code introduces three points not stated in the Memo:

- **Primary vs. secondary source rule** — only `SourceType.primary` is citable;
  guidelines are discovery context unless they reproduce and link to a primary
  instrument.
- **Quote-by-span-ID orchestrator pattern** — the LLM emits IDs only; the
  orchestrator copies quote text from canonical storage. Hallucination is
  prevented by construction, not by post-hoc validation.
- **Evaluation methodology** — 50–100 gold clauses across the demo
  jurisdictions, with six metrics (retrieval recall@k, indicator precision,
  citation exact-match rate, OCR citable-page rate, abstention quality,
  conflict-detection accuracy). Harness skeleton in `scripts/eval_gold_set.py`.
