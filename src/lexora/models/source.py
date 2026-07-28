"""Source profile, portal spec, and raw document models."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class LegalSystem(str, Enum):
    civil = "civil"
    common = "common"
    hybrid = "hybrid"


class SourceType(str, Enum):
    """Citability tier.

    primary   = binding law, regulation, official gazette, ratified treaty.
                Quotes from here are citable evidence.
    secondary = guidelines, FAQs, commentary, third-party datasets.
                Discovery context only; never cited as binding unless reproducing
                a primary instrument.
    """

    primary = "primary"
    secondary = "secondary"


class FetchMethod(str, Enum):
    http = "http"
    sitemap = "sitemap"
    playwright = "playwright"
    api = "api"


class InstrumentStatus(str, Enum):
    """Lifecycle status of a legal instrument — drives the official enforced-only
    filter (exclude pending drafts + repealed measures from the inventory).

    ``unknown`` is the safe default: the enforced-only filter treats it as
    enforced, so a law is only ever dropped on a POSITIVE repealed/draft signal,
    never on the mere absence of an in-force confirmation. See
    :mod:`lexora.classify.lifecycle`.
    """

    in_force = "IN_FORCE"
    repealed = "REPEALED"  # repealed / ceased / spent / revoked — no longer enforced
    draft = "DRAFT"  # bill / exposure draft / not yet commenced — not yet enforced
    unknown = "UNKNOWN"


class PortalSpec(BaseModel):
    name: str
    url: HttpUrl
    source_type: SourceType
    fetch_method: FetchMethod = FetchMethod.http
    search_query: str | None = None
    # Optional template to turn a query into a portal search URL, e.g.
    #   "https://lom.agc.gov.my/search.php?keyword={query}"
    # `{query}` is URL-encoded before substitution. When absent, discovery
    # harvests candidate links from the portal landing page instead.
    search_url_template: str | None = None
    # Whether the portal supports FULL-TEXT search. True (SG SSO, MY Fess) means
    # indicator concept phrases reach sectoral statutes by their text. False (AU
    # OData, which only matches law NAMES) means concept phrases are useless — the
    # multi-instrument discoverer falls back to name-driven lookup of the
    # jurisdiction's known instruments instead of scraping the SPA browse list.
    full_text: bool = True
    # Whether the portal's search matches a BAG OF WORDS rather than the literal
    # string (SG SSO's `PhraseType=AllWords`). When true, punctuation carries no
    # meaning and a query that merely adds a ubiquitous word ("Act") asks the same
    # question — so query variants that differ only in those ways are redundant.
    word_tokenised_search: bool = False
    notes: str | None = None


class FetchPolicy(BaseModel):
    """How hard a jurisdiction's portal may be pushed.

    Document-level concurrency is worth several minutes a run — 93% of the per-document
    time is spent waiting on the network — but every portal here sits behind a
    rate-limiter that answers a burst with a refusal rather than an error. The safe
    setting is a property of the portal, not of the caller, so it lives beside the
    portal definition and defaults to fully serial for a profile that has not been
    measured. Raise it only with the acquisition tally (blocked / unrendered counts)
    in front of you: parallelism that turns answers into refusals is not a speed-up.
    """

    doc_workers: int = 1        # instruments processed concurrently
    serial_fetch: bool = False  # one same-host download in flight at a time
    min_interval: float = 0.0   # minimum seconds between same-host fetch starts


class SourceProfile(BaseModel):
    """A jurisdiction configuration loaded from configs/jurisdictions/<iso>.yaml."""

    jurisdiction: str
    iso_code: str = Field(min_length=2, max_length=3)
    primary_language: str
    additional_languages: list[str] = Field(default_factory=list)
    legal_system: LegalSystem
    ocr_languages: list[str] = Field(default_factory=list)
    keywords_by_indicator: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    # Canonical names of the primary instruments we expect to find in this
    # jurisdiction. Discovery fuzzy-matches candidate titles against these to
    # tag KNOWN hits and to flag instrument-like links that match nothing here
    # as NEW-evidence candidates.
    known_instruments: list[str] = Field(default_factory=list)
    # Map a portal-native instrument identifier (e.g. MY Act number "709", AU
    # "C2004A03712") to the canonical instrument name. Lets a strategy tag a hit
    # KNOWN by identity even when the result's title is a filename that defeats
    # fuzzy name matching (MY Fess document records).
    known_instrument_ids: dict[str, str] = Field(default_factory=dict)
    # Curated amendment registry (BACKSTOP only — Signal C in lexora.cite.amendments).
    # Maps a principal law's native number (or title) to its known amending acts, so a
    # citation drawn from a stale source is flagged even when the amending instrument
    # was not fetched. Only covers KNOWN laws; the corpus/portal signals generalize to
    # NEW laws. Shape: {"709": [{"by": "Act A1727", "year": 2024}, ...]}.
    amended_by: dict[str, list[dict]] = Field(default_factory=dict)
    # How hard this jurisdiction's portals tolerate being pushed. Serial by default.
    fetch_policy: FetchPolicy = Field(default_factory=FetchPolicy)
    portals: list[PortalSpec] = Field(default_factory=list)


class RawDocument(BaseModel):
    """An immutable record of a document fetched from a portal.

    The raw bytes are stored separately in the object store at `bytes_path`.
    """

    document_id: str
    source_url: HttpUrl
    retrieval_timestamp: datetime
    http_status: int
    sha256: str
    content_type: str
    bytes_path: str
    portal_name: str
    jurisdiction: str
    source_type: SourceType
    title: str | None = None
    # Structured metadata captured from the portal at fetch time (register API /
    # page label), when available. Carried so the citation layer can populate the
    # Law Number / Last Amended columns from the source itself, generalizing to
    # NEW laws. Empty -> the citation layer falls back to the LLM metadata extractor.
    law_number: str = ""
    last_amended: str = ""
    # Lifecycle status captured from the portal channel at fetch time (e.g. the AU
    # register's ``isInForce`` flag). Drives the enforced-only filter; ``unknown``
    # is treated as enforced. See :class:`InstrumentStatus`.
    status: InstrumentStatus = InstrumentStatus.unknown
