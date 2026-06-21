"""Secondary-source signal model (RDTII work-stream WS-S).

A :class:`SecondarySignal` is a POINTER harvested from a third-party tracker or
database (UNCTAD, OECD STRI, law-firm trackers). The ESCAP RDTII 2.1 Guide says
such secondary sources "should only serve to guide researchers to the primary
sources" — they are never scored and never quoted as evidence.

RED LINE: this is deliberately NOT a Citation. It carries no verbatim span, no
clause id, and no quote. It must never be converted into an ``EvidenceClaim`` /
``Citation`` or emitted in the submission CSV. Its only sanctioned uses are the
three in :mod:`lexora.collect.secondary.base` — discovery-recall seeding,
coverage cross-check, and provenance annotation — none of which produce evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Presence(str, Enum):
    """What the secondary source asserts about the economy having the law.

    ``yes``/``draft`` mean the source points to an (enacted/pending) instrument we
    should be able to find; ``no`` is an explicit absence (useful for the coverage
    cross-check); ``unknown`` is the safe default when the source is silent.
    """

    yes = "yes"
    no = "no"
    draft = "draft"
    unknown = "unknown"


class SecondarySignal(BaseModel):
    """A non-citable pointer from a secondary source to a primary instrument."""

    economy: str  # how the source labels the economy (ISO code or country name)
    indicator_id: str  # RDTII submission_id this signal informs, e.g. "P7-I1"
    source_name: str  # human-readable, e.g. "UNCTAD Data Protection and Privacy Worldwide"
    presence: Presence = Presence.unknown
    # The primary instrument the source points to — the SEED for discovery recall.
    # A name only; resolving it to a real, citable document is the primary
    # pipeline's job (the secondary source never supplies citable text).
    primary_law_name: str = ""
    primary_law_ref: str = ""  # act/law number if the source gives one
    source_url: str = ""  # the secondary-source page the signal was read from
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_snippet: str = ""  # short copy of the source row, for provenance/audit only


__all__ = ["Presence", "SecondarySignal"]
