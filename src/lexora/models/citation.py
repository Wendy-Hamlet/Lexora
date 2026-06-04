"""Citation + claim models — the on-the-wire contract.

`EvidenceClaim` is what the LLM verifier returns. Notice it has NO quote text —
quote text is copied from canonical storage by the orchestrator, never written
by the model.

`Citation` is what reaches the audit UI / JSON-LD export, after the validator
has confirmed all gates.

See docs/citation_schema.md and docs/anti_hallucination.md.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class ClaimLabel(str, Enum):
    match = "match"
    no_match = "no_match"
    uncertain = "uncertain"


class DiscoveryTag(str, Enum):
    """Whether the provision was supplied in the sample kit or found by the tool.

    NEW provisions are worth 20 of the 40 Substantive-Accuracy points, so this
    field is required in the official output template.
    """

    new = "NEW"
    known = "KNOWN"


class Coverage(str, Enum):
    """RDTII coverage of a measure. Carried for gold-data matching / JSON; not a
    column in the submission CSV template."""

    horizontal = "Horizontal"
    sectoral = "Sectoral"


class ReviewStatus(str, Enum):
    verified = "VERIFIED"
    low_ocr_confidence = "LOW_OCR_CONFIDENCE"
    conflict_review = "CONFLICT_REVIEW"
    no_primary_source = "NO_PRIMARY_SOURCE_FOUND"
    hallucinated_or_unsupported = "HALLUCINATED_OR_UNSUPPORTED_MAPPING"


class EvidenceClaim(BaseModel):
    """The LLM verifier's output.

    The model selects IDs and a label; the orchestrator fills the quote from
    canonical storage. The model cannot produce free text by design.
    """

    indicator_id: str
    clause_id: str
    quote_span_id: str
    label: ClaimLabel
    confidence: float = Field(ge=0.0, le=1.0)


class Citation(BaseModel):
    """Final, validated citation released to the audit UI / exports.

    Field set is a superset of the official OUTPUT_TEMPLATE columns plus the
    provenance fields needed for the verbatim audit trail. The CSV exporter
    projects this onto the exact 13-column submission schema.
    """

    # --- official output columns (OUTPUT_TEMPLATE_31MAY.xlsx) ---
    economy: str = ""  # official UN economy name, e.g. "Singapore"
    title: str  # "Law Name" — full official statute name + year
    law_number: str = ""  # "Law Number / Ref" — e.g. "Act 709", "No. 9 of 2018"
    last_amended: str = ""  # "Last Amended" — year; blank if not amended
    indicator_id: str  # "Indicator ID" — submission code, e.g. "P6-I4"
    article_path: str  # "Article / Section" — e.g. "S. 26(1)"
    discovery_tag: DiscoveryTag = DiscoveryTag.known
    page_or_dom_anchor: str  # "Location Reference" — PDF page | HTML anchor
    quote: str  # "Verbatim Snippet" — copied from canonical span (never the LLM)
    mapping_rationale: str = ""  # max 300 chars
    source_url: HttpUrl  # direct URL to the law on the official portal
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""

    # --- provenance / audit (JSON export + UI, not in submission CSV) ---
    clause_id: str
    retrieval_timestamp: datetime
    document_hash: str  # sha256:<hex>
    jurisdiction: str
    legal_form: str  # statute | regulation | gazette | treaty | guideline
    coverage: Coverage | None = None
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    review_status: ReviewStatus = ReviewStatus.verified
