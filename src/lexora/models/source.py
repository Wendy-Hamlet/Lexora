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
    notes: str | None = None


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
