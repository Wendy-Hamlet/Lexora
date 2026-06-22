"""Secondary-source layer (WS-S): RDTII third-party trackers as a discovery aid.

Re-exports the S-0 infrastructure. Concrete source adapters (UNCTAD, OECD STRI,
law-firm trackers) are added in later phases and self-register via
:func:`register_source`.
"""
from __future__ import annotations

# Import adapter modules for their import-time self-registration into
# SECONDARY_SOURCES (S-1 UNCTAD; S-3 STRI snapshot; S-4 DLA Piper + ICLG;
# WS-S extra free sources: Linklaters + World Map of Encryption).
from lexora.collect.secondary import (  # noqa: E402,F401
    dlapiper,
    iclg,
    linklaters,
    stri,
    unctad,
    worldmap,
)
from lexora.collect.secondary.base import (
    SECONDARY_SOURCES,
    CoverageGap,
    SecondaryAdapter,
    SecondarySourceSpec,
    adapter_for,
    applicable_sources,
    cached_json,
    coverage_gaps,
    gather_signals,
    indicator_gaps,
    load_secondary_sources,
    provenance_notes_by_indicator,
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
    "gather_signals",
    "indicator_gaps",
    "load_secondary_sources",
    "provenance_notes_by_indicator",
    "register_source",
    "to_discovery_seeds",
    "to_provenance_note",
    "Presence",
    "SecondarySignal",
]
