"""Secondary-source infrastructure (WS-S, S-0).

This is the cross-jurisdiction, indicator-keyed layer for the RDTII-defined
secondary sources (UNCTAD, OECD STRI, law-firm trackers). It is deliberately
SEPARATE from :data:`lexora.collect.strategies.PORTAL_CONNECTORS` (which are
per-host, per-jurisdiction regulator-guidance connectors producing CITABLE
subsidiary instruments): a secondary source produces a non-citable
:class:`~lexora.models.secondary.SecondarySignal` pointer, not evidence.

S-0 ships the machinery only — the registry, an adapter protocol, a disk cache
helper, the config loader, and the THREE (and only three) sanctioned ways a
signal is allowed to leave this layer. Concrete adapters (UNCTAD, OECD, …) land
in later phases (S-1, S-3, S-4) and self-register via :func:`register_source`.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

from lexora.models.indicator import RDTIIIndicator
from lexora.models.secondary import Presence, SecondarySignal


@dataclass(frozen=True)
class SecondarySourceSpec:
    """Declarative description of one secondary source (from config).

    ``indicators`` are the RDTII submission_ids this source can inform (per the
    guide's per-indicator "useful secondary sources" notes). ``tier`` orders the
    build effort: 1 = easiest/free (UNCTAD), 3 = downloadable index (STRI),
    4 = paywalled law-firm tracker.
    """

    key: str
    name: str
    url: str = ""
    pillars: tuple[int, ...] = ()
    indicators: tuple[str, ...] = ()
    tier: int = 1
    notes: str = ""


@runtime_checkable
class SecondaryAdapter(Protocol):
    """A callable that harvests signals for one economy from one source.

    Adapters take the economy (ISO code) and the in-scope indicators and return a
    list of :class:`SecondarySignal`. They MUST NOT return citable instruments;
    the return type is the structural guarantee that they can't inject evidence.
    """

    key: str

    def __call__(
        self, economy: str, indicators: Sequence[RDTIIIndicator], **kwargs
    ) -> list[SecondarySignal]: ...


# Source key -> adapter. Empty in S-0; adapters self-register in later phases.
SECONDARY_SOURCES: dict[str, SecondaryAdapter] = {}


def register_source(key: str) -> Callable[[SecondaryAdapter], SecondaryAdapter]:
    """Decorator: register an adapter under ``key`` (its config key)."""

    def _register(adapter: SecondaryAdapter) -> SecondaryAdapter:
        SECONDARY_SOURCES[key] = adapter
        return adapter

    return _register


def adapter_for(key: str) -> SecondaryAdapter | None:
    """Return the registered adapter for a source key, or None if not built yet."""
    return SECONDARY_SOURCES.get(key)


# --- config -----------------------------------------------------------------

# .../Lexora/src/lexora/collect/secondary/base.py -> parents[4] == repo root (Lexora/)
_DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "configs" / "secondary_sources.yaml"


def load_secondary_sources(path: Path | None = None) -> list[SecondarySourceSpec]:
    """Load the secondary-source catalogue from ``configs/secondary_sources.yaml``."""
    path = Path(path) if path is not None else _DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    specs: list[SecondarySourceSpec] = []
    for s in data.get("sources", []) or []:
        specs.append(
            SecondarySourceSpec(
                key=str(s["key"]),
                name=str(s["name"]),
                url=str(s.get("url", "") or ""),
                pillars=tuple(int(p) for p in s.get("pillars", []) or []),
                indicators=tuple(str(i) for i in s.get("indicators", []) or []),
                tier=int(s.get("tier", 1)),
                notes=str(s.get("notes", "") or ""),
            )
        )
    return specs


def applicable_sources(
    indicator_id: str, specs: Sequence[SecondarySourceSpec]
) -> list[SecondarySourceSpec]:
    """The secondary sources the guide attaches to ``indicator_id`` (submission_id)."""
    return [s for s in specs if indicator_id in s.indicators]


# --- disk cache (secondary datasets are stable; cache like au_act_catalogue) --


def cached_json(cache_path: str | os.PathLike, ttl_hours: float, producer: Callable[[], object]):
    """Return cached JSON when fresh, else call ``producer()``, cache, and return.

    Generalises the :func:`lexora.collect.strategies.au_act_catalogue` pattern.
    A failed/empty producer result is returned but NOT cached, so a transient
    fetch failure never poisons the cache. Cache read/write errors degrade to a
    live call rather than raising.
    """
    cache_path = os.fspath(cache_path)
    if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path)) < ttl_hours * 3600:
        try:
            with open(cache_path, encoding="utf-8") as fh:
                cached = json.load(fh)
            if cached:
                return cached
        except Exception:
            pass

    result = producer()
    if result:
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False)
        except Exception:
            pass
    return result


# --- the THREE sanctioned exits (this is where the red line is enforced) -----
#
# A SecondarySignal may influence the system ONLY through these helpers. Each is
# deliberately LOSSY — it returns plain strings, never an instrument or a
# Citation — so there is no code path from a secondary source to scored evidence.


def to_discovery_seeds(signals: Iterable[SecondarySignal]) -> list[str]:
    """USE 1 (discovery-recall booster): the primary-law NAME strings to seed into
    discovery, for signals that assert the law exists (yes/draft). Deduped,
    order-preserving. Returns ``list[str]`` — never instruments, never citations.
    """
    seeds: list[str] = []
    seen: set[str] = set()
    for s in signals:
        if s.presence in (Presence.yes, Presence.draft):
            name = s.primary_law_name.strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                seeds.append(name)
    return seeds


def to_provenance_note(signal: SecondarySignal) -> str:
    """USE 3 (provenance annotation): the short Notes/raw_context string. Tagged
    so a reader knows it is CORROBORATION, not evidence. Returns ``str``."""
    note = f"corroborated by secondary source: {signal.source_name}"
    if signal.source_url:
        note += f" ({signal.source_url})"
    return note


@dataclass
class CoverageGap:
    """USE 2 (coverage cross-check) row: one indicator/economy where a secondary
    source asserts a law exists but our discovery found nothing to back it."""

    economy: str
    indicator_id: str
    source_name: str
    expected_law_name: str = ""


def coverage_gaps(
    signals: Iterable[SecondarySignal], discovered_names: Iterable[str]
) -> list[CoverageGap]:
    """USE 2: signals asserting a law exists (yes) whose pointed-to law name does
    not fuzzily appear among ``discovered_names`` -> a recall gap to report. Pure
    + offline; the fuzzy match lives in S-2 when wired to real discovery output.
    Here we use a conservative case-insensitive containment check."""
    found = [d.lower() for d in discovered_names]
    gaps: list[CoverageGap] = []
    for s in signals:
        if s.presence is not Presence.yes:
            continue
        name = s.primary_law_name.strip().lower()
        if name and not any(name in d or d in name for d in found):
            gaps.append(
                CoverageGap(
                    economy=s.economy,
                    indicator_id=s.indicator_id,
                    source_name=s.source_name,
                    expected_law_name=s.primary_law_name.strip(),
                )
            )
    return gaps


__all__ = [
    "SecondarySourceSpec",
    "SecondaryAdapter",
    "SECONDARY_SOURCES",
    "register_source",
    "adapter_for",
    "load_secondary_sources",
    "applicable_sources",
    "cached_json",
    "to_discovery_seeds",
    "to_provenance_note",
    "CoverageGap",
    "coverage_gaps",
]
