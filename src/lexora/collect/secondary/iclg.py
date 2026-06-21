"""ICLG 'Data Protection Laws and Regulations' adapter (WS-S, S-4 second source).

ICLG publishes a free, server-rendered per-jurisdiction chapter naming the
data-protection statutes. It complements DLA Piper: for SG it corroborates the
principal Act, and for AU it additionally surfaces the STATE/TERRITORY privacy
acts (Information Privacy Act 2014 (ACT), PPIP Act 1998 (NSW), …) that DLA's
federal-focused summary omits — real extra USE-1 seeds.

Coverage is partial: ICLG has no Malaysia data-protection chapter (404), so MY
degrades to no signals (DLA Piper / UNCTAD still cover it). Reuses the DLA Piper
LLM-first / regex-fallback :func:`extract_laws` (same accuracy A/B applies).
"""
from __future__ import annotations

import os
import re
from collections.abc import Sequence

import httpx

from lexora.collect.secondary.base import cached_json, register_source
from lexora.collect.secondary.dlapiper import extract_laws
from lexora.models.indicator import RDTIIIndicator
from lexora.models.secondary import Presence, SecondarySignal

_URL = "https://iclg.com/practice-areas/data-protection-laws-and-regulations/{slug}"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36"
_SOURCE = "ICLG Data Protection Laws and Regulations"

# ICLG chapter slugs are country names. Curated for the in-scope + probed economies;
# an unmapped ISO is skipped (don't guess a slug that would 404). Extend as needed.
_ISO_TO_SLUG = {
    "SG": "singapore", "AU": "australia", "MY": "malaysia", "TH": "thailand",
    "ID": "indonesia", "PH": "philippines", "IN": "india", "JP": "japan", "CN": "china",
}

# A paragraph that actually names a statute (year-bearing), to skip ICLG's generic
# "covers common issues ... in 27 jurisdictions" preface.
_STATUTE = re.compile(r"(Act,?\s+\d{4}|Act of \d{4}|Law No\.?\s*\d+|Ordinance\s+\d{4})")


def iclg_law_text(html: str) -> str:
    """Pull the statute-naming prose from an ICLG chapter page (skipping preface)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    parts: list[str] = []
    for tag in soup.find_all(["p", "li"]):
        t = tag.get_text(" ", strip=True)
        if len(t) > 60 and _STATUTE.search(t) and re.search(
            r"(data protection|privacy|personal data)", t, re.I
        ):
            parts.append(t)
        if len(" ".join(parts)) > 1400:
            break
    return re.sub(r"\s+", " ", " ".join(parts))[:1500]


def _fetch_law_text(
    economy: str, *, client: httpx.Client | None = None, timeout: float = 30.0,
    cache_path: str | None = None, ttl_hours: float = 24.0,
) -> str:
    """Fetch + 24h-cache ICLG chapter text for one economy, or '' if uncovered."""
    iso = economy.upper()
    slug = _ISO_TO_SLUG.get(iso)
    if not slug:
        return ""
    cache_path = cache_path or os.path.join("outputs", "cache", f"iclg_{iso.lower()}.json")

    def _produce() -> dict:
        owns = client is None
        cl = client or httpx.Client(
            follow_redirects=True, timeout=timeout, headers={"User-Agent": _UA}
        )
        try:
            resp = cl.get(_URL.format(slug=slug))
            if resp.status_code != 200:  # e.g. MY -> 404
                return {}
            return {"text": iclg_law_text(resp.text)}
        except Exception:
            return {}
        finally:
            if owns:
                cl.close()

    blob = cached_json(cache_path, ttl_hours, _produce)
    return (blob or {}).get("text", "")


@register_source("iclg")
def iclg_data_protection(
    economy: str,
    indicators: Sequence[RDTIIIndicator],
    *,
    data: str | None = None,
    use_llm: bool = True,
    llm_client=None,
    **fetch_kwargs,
) -> list[SecondarySignal]:
    """Emit a signal per (data-protection law name x in-scope indicator), like the
    DLA Piper adapter. ``data`` short-circuits the fetch with the chapter TEXT."""
    text = data if data is not None else _fetch_law_text(economy, **fetch_kwargs)
    if not text:
        return []
    laws = extract_laws(text, use_llm=use_llm, client=llm_client)
    if not laws:
        return []
    iso = economy.upper()
    slug = _ISO_TO_SLUG.get(iso, "")
    wanted = {i.submission_id for i in indicators}
    targets = [i for i in ("P7-I1", "P7-I4") if not wanted or i in wanted]
    out: list[SecondarySignal] = []
    for law in laws:
        for ind_id in targets:
            out.append(
                SecondarySignal(
                    economy=iso,
                    indicator_id=ind_id,
                    source_name=_SOURCE,
                    presence=Presence.yes,
                    primary_law_name=law,
                    source_url=_URL.format(slug=slug) if slug else "",
                    raw_snippet=f"ICLG names: {law}",
                )
            )
    return out


__all__ = ["iclg_data_protection", "iclg_law_text"]
