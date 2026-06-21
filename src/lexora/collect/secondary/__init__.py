"""Secondary-source layer (WS-S): RDTII third-party trackers as a discovery aid.

Re-exports the S-0 infrastructure. Concrete source adapters (UNCTAD, OECD STRI,
law-firm trackers) are added in later phases and self-register via
:func:`register_source`.
"""
from __future__ import annotations

from lexora.collect.secondary.base import (
    SECONDARY_SOURCES,
    CoverageGap,
    SecondaryAdapter,
    SecondarySourceSpec,
    adapter_for,
    applicable_sources,
    cached_json,
    coverage_gaps,
    load_secondary_sources,
    register_source,
    to_discovery_seeds,
    to_provenance_note,
)
from lexora.models.secondary import Presence, SecondarySignal

__all__ = [
    "SECONDARY_SOURCES",
    "CoverageGap",
    "SecondaryAdapter",
    "SecondarySourceSpec",
    "adapter_for",
    "applicable_sources",
    "cached_json",
    "coverage_gaps",
    "load_secondary_sources",
    "register_source",
    "to_discovery_seeds",
    "to_provenance_note",
    "Presence",
    "SecondarySignal",
]
