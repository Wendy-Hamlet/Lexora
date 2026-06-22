"""Linklaters 'Data Protected' adapter (WS-S) — per-jurisdiction data-protection laws.

The RDTII guide lists Linklaters Data Protected under both Pillar 6 and Pillar 7.
Each jurisdiction has a server-rendered page at
``/en/insights/data-protected/data-protected---<slug>`` whose "National Legislation"
section names the principal data-protection statute(s). Like DLA Piper / ICLG it
yields law NAMES, so it can feed USE 1 (discovery seeding) for P7-I1/P7-I4.

DATA-QUALITY GUARD: Linklaters mis-serves some jurisdiction bodies — e.g. the
``---malaysia`` page renders the EU GDPR / Maltese act in its body (its <title>
correctly says "Data Protected: Malaysia", but the legislation prose is a foreign
default). So extraction is cross-checked against the EXPECTED country: an
LLM extractor returns [] when the text is about another country/region, and the
deterministic fallback drops laws bearing a foreign demonym / "(EU)" marker. The
result for such a page is no signal (accuracy over recall).
"""
from __future__ import annotations

import os
import re
from collections.abc import Sequence

import httpx

from lexora.collect.secondary.base import cached_json, register_source
from lexora.collect.secondary.dlapiper import extract_laws_regex
from lexora.models.indicator import RDTIIIndicator
from lexora.models.secondary import Presence, SecondarySignal

_URL = "https://www.linklaters.com/en/insights/data-protected/data-protected---{slug}"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36"
_SOURCE = "Linklaters Data Protected"

# Linklaters slugs are country names; curated for in-scope + probed economies.
_ISO_TO_SLUG = {
    "SG": "singapore", "AU": "australia", "MY": "malaysia", "TH": "thailand",
    "ID": "indonesia", "PH": "philippines", "IN": "india", "JP": "japan",
    "CN": "china", "HK": "hong-kong", "GB": "united-kingdom",
}

# Markers that betray a mis-served / foreign body (the Malaysia->EU/Malta bug).
_FOREIGN_MARKERS = re.compile(
    r"\(EU\)|European|\bGDPR\b|Maltese|British|United Kingdom|Maltese", re.I
)

_LLM_SYSTEM = (
    "You are given the 'national legislation' text from a data-protection country "
    "profile that is SUPPOSED to be about {country}. Extract {country}'s principal "
    "personal-data-protection / privacy statute(s), with their year. Output rules:\n"
    "- Copy each title VERBATIM from the text. Use your knowledge ONLY to verify the "
    "jurisdiction; do not invent titles not present in the text.\n"
    "- If the text is actually about a DIFFERENT country, or only about the EU GDPR "
    "(a known mis-served page), return [] — do not guess.\n"
    "- Exclude foreign laws and sectoral laws mentioned only in passing.\n"
    'Respond as JSON: {"laws": ["...", "..."]}.'
)


def linklaters_law_text(html: str) -> str:
    """Pull the 'National Legislation' / 'General data protection laws' prose."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    parts: list[str] = []
    started = False
    for tag in soup.find_all(["h2", "h3", "h4", "h5", "strong", "p", "li"]):
        t = tag.get_text(" ", strip=True)
        if not t:
            continue
        if re.search(r"General data protection laws|National Legislation", t, re.I):
            started = True
            continue
        if started:
            if re.search(r"\b(Act|Ordinance|Law|Regulation)\b", t) and len(t) > 12:
                parts.append(t)
            if len(" ".join(parts)) > 700:
                break
    return re.sub(r"\s+", " ", " ".join(parts))[:900]


def _drop_foreign(names: list[str]) -> list[str]:
    """Deterministic guard: drop laws bearing a foreign-jurisdiction marker."""
    return [n for n in names if not _FOREIGN_MARKERS.search(n)]


def extract_laws_for_country(
    text: str, country: str, *, use_llm: bool = True, client=None
) -> list[str]:
    """LLM cross-check against the expected country (returns [] on mismatch), with a
    regex + foreign-marker-drop fallback when the LLM backend is unavailable."""
    if use_llm:
        laws = _extract_llm(text, country, client)
        if laws is not None:
            return laws
    # Fallback: if the body leads with the EU GDPR it's a mis-serve -> trust nothing.
    if re.search(r"^\W*The General Data Protection Regulation \(EU\)", text):
        return []
    return _drop_foreign(extract_laws_regex(text))


def _extract_llm(text: str, country: str, client=None) -> list[str] | None:
    if client is None:
        from lexora.classify.llm_client import LlmClient, is_available

        if not is_available():
            return None
        try:
            client = LlmClient(timeout=60)
        except Exception:
            return None
    try:
        data = client.chat(_LLM_SYSTEM.format(country=country), f"TEXT:\n{text}", json_schema={})
    except Exception:
        return None
    laws = data.get("laws", []) if isinstance(data, dict) else []
    if not isinstance(laws, list):
        return None
    out, seen = [], set()
    for x in laws:
        n = str(x).strip()
        if len(n) >= 8 and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def _fetch_law_text(
    economy: str, *, client: httpx.Client | None = None, timeout: float = 40.0,
    cache_path: str | None = None, ttl_hours: float = 24.0,
) -> str:
    """Fetch + 24h-cache the Linklaters chapter text for one economy, or ''."""
    iso = economy.upper()
    slug = _ISO_TO_SLUG.get(iso)
    if not slug:
        return ""
    cache_path = cache_path or os.path.join("outputs", "cache", f"linklaters_{iso.lower()}.json")

    def _produce() -> dict:
        owns = client is None
        cl = client or httpx.Client(
            follow_redirects=True, timeout=timeout, headers={"User-Agent": _UA}
        )
        try:
            resp = cl.get(_URL.format(slug=slug))
            if resp.status_code != 200:
                return {}
            return {"text": linklaters_law_text(resp.text)}
        except Exception:
            return {}
        finally:
            if owns:
                cl.close()

    blob = cached_json(cache_path, ttl_hours, _produce)
    return (blob or {}).get("text", "")


@register_source("linklaters")
def linklaters_data_protection(
    economy: str,
    indicators: Sequence[RDTIIIndicator],
    *,
    data: str | None = None,
    use_llm: bool = True,
    llm_client=None,
    **fetch_kwargs,
) -> list[SecondarySignal]:
    """Emit a signal per (data-protection law name x in-scope indicator), like the
    DLA Piper / ICLG adapters, but jurisdiction-cross-checked. ``data`` short-circuits
    the fetch with the chapter TEXT (offline tests)."""
    iso = economy.upper()
    country = _ISO_TO_SLUG.get(iso, "").replace("-", " ").title() or iso
    text = data if data is not None else _fetch_law_text(economy, **fetch_kwargs)
    if not text:
        return []
    laws = extract_laws_for_country(text, country, use_llm=use_llm, client=llm_client)
    if not laws:
        return []
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
                    raw_snippet=f"Linklaters names: {law}",
                )
            )
    return out


__all__ = [
    "linklaters_data_protection",
    "linklaters_law_text",
    "extract_laws_for_country",
]
