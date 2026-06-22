"""UNCTAD secondary-source adapters (WS-S, S-1).

All three UNCTAD trackers named by the RDTII guide — Global Cyberlaw Tracker, Data
Protection & Privacy Legislation Worldwide, Cybercrime Legislation Worldwide — are
driven by ONE published dataset, ``CyberlawData.js``: an ISO2 -> 5-int array map
where the columns are::

    [Electronic Transactions, Consumer Protection,
     Privacy and Data Protection, Cybercrime, Indirect Taxation]

and the code (verified against the file's own ``statistics`` block) is
``0 = no data, 1 = legislation, 2 = draft legislation, 3 = no legislation``.

We fetch+cache that one file and expose three adapters reading the relevant
column(s). UNCTAD reports PRESENCE only (no instrument name), so these signals
feed the coverage cross-check (USE 2) and provenance (USE 3); they do not seed
discovery (USE 1), which needs a law name. ``to_discovery_seeds`` already drops
the blank ``primary_law_name`` these emit, so that contract holds automatically.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence

import httpx

from lexora.collect.secondary.base import cached_json, register_source
from lexora.models.indicator import RDTIIIndicator
from lexora.models.secondary import Presence, SecondarySignal

_DATA_URL = "https://unctad.org/sites/default/files/data-file/CyberlawData.js"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36"

# Column index -> human label (for the raw_snippet audit string).
_COL_PRIVACY = 2
_COL_CYBERCRIME = 3
_COL_LABEL = {_COL_PRIVACY: "Privacy and Data Protection", _COL_CYBERCRIME: "Cybercrime"}

# Verified against CyberlawData.js `statistics`: 0/1/2/3 -> unknown/yes/draft/no.
_CODE_TO_PRESENCE = {
    0: Presence.unknown,
    1: Presence.yes,
    2: Presence.draft,
    3: Presence.no,
}

# countries:{"AF":[..],...} — values are flat int arrays, so no nested braces.
_COUNTRIES_RE = re.compile(r"countries:\s*(\{[^{}]*\})")


def _fetch_cyberlaw_data(
    *,
    data: dict | None = None,
    client: httpx.Client | None = None,
    timeout: float = 30.0,
    cache_path: str | None = None,
    ttl_hours: float = 24.0,
) -> dict[str, list[int]]:
    """Return ``{ISO2: [5 ints]}`` from UNCTAD's CyberlawData.js (disk-cached 24h).

    ``data`` short-circuits the fetch (offline tests pass a parsed dict). A failed
    fetch/parse returns ``{}`` (never cached), so a transient outage degrades to
    no signals rather than crashing a run."""
    if data is not None:
        return data
    cache_path = cache_path or os.path.join("outputs", "cache", "unctad_cyberlaw.json")

    def _produce() -> dict[str, list[int]]:
        owns = client is None
        cl = client or httpx.Client(
            follow_redirects=True, timeout=timeout, headers={"User-Agent": _UA}
        )
        try:
            resp = cl.get(_DATA_URL)
            if resp.status_code != 200:
                return {}
            return parse_cyberlaw_js(resp.text)
        except Exception:
            return {}
        finally:
            if owns:
                cl.close()

    return cached_json(cache_path, ttl_hours, _produce)


def parse_cyberlaw_js(text: str) -> dict[str, list[int]]:
    """Extract the ``countries`` ISO2->ints map from the CyberlawData.js source."""
    m = _COUNTRIES_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except Exception:
        return {}


def _emit(
    economy: str,
    col_to_indicators: dict[int, tuple[str, ...]],
    source_name: str,
    indicators: Sequence[RDTIIIndicator],
    **fetch_kwargs,
) -> list[SecondarySignal]:
    """Build signals for one economy: for each tracker column, map its presence
    code to each candidate indicator that is in-scope for this call."""
    iso = economy.upper()
    row = _fetch_cyberlaw_data(**fetch_kwargs).get(iso)
    if row is None:
        return []
    wanted = {i.submission_id for i in indicators}
    out: list[SecondarySignal] = []
    for col, ind_ids in col_to_indicators.items():
        if col >= len(row):
            continue
        presence = _CODE_TO_PRESENCE.get(row[col], Presence.unknown)
        for ind_id in ind_ids:
            if wanted and ind_id not in wanted:
                continue
            out.append(
                SecondarySignal(
                    economy=iso,
                    indicator_id=ind_id,
                    source_name=source_name,
                    presence=presence,
                    source_url=_DATA_URL,
                    raw_snippet=f"UNCTAD {_COL_LABEL[col]}: {presence.value}",
                )
            )
    return out


@register_source("unctad_data_protection")
def unctad_data_protection(
    economy: str, indicators: Sequence[RDTIIIndicator], **kw
) -> list[SecondarySignal]:
    """Privacy & Data Protection column -> comprehensive data-protection framework
    (P7-I1) and the DPIA/DPO regime that lives inside it (P7-I4)."""
    return _emit(
        economy,
        {_COL_PRIVACY: ("P7-I1", "P7-I4")},
        "UNCTAD Data Protection and Privacy Legislation Worldwide",
        indicators,
        **kw,
    )


@register_source("unctad_cyberlaw")
def unctad_cyberlaw(
    economy: str, indicators: Sequence[RDTIIIndicator], **kw
) -> list[SecondarySignal]:
    """Global Cyberlaw Tracker: privacy column -> P7-I1, cybercrime column -> P7-I2."""
    return _emit(
        economy,
        {_COL_PRIVACY: ("P7-I1",), _COL_CYBERCRIME: ("P7-I2",)},
        "UNCTAD Global Cyberlaw Tracker",
        indicators,
        **kw,
    )


@register_source("unctad_cybercrime")
def unctad_cybercrime(
    economy: str, indicators: Sequence[RDTIIIndicator], **kw
) -> list[SecondarySignal]:
    """Cybercrime column -> dedicated cybersecurity framework (P7-I2).

    The guide names UNCTAD Cybercrime Legislation Worldwide specifically under P7-I2.
    Cybercrime statutes often carry retention (P7-I3) and government-access (P7-I5)
    provisions, but the tracker reports presence only — presence alone is not
    evidence of those, so we no longer claim P7-I3/P7-I5 (reconciled 2026-06-22)."""
    return _emit(
        economy,
        {_COL_CYBERCRIME: ("P7-I2",)},
        "UNCTAD Cybercrime Legislation Worldwide",
        indicators,
        **kw,
    )


__all__ = [
    "unctad_data_protection",
    "unctad_cyberlaw",
    "unctad_cybercrime",
    "parse_cyberlaw_js",
]
