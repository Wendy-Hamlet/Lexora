"""DLA Piper 'Data Protection Laws of the World' adapter (WS-S, S-1's law-name tier).

DLA Piper server-renders, per country, the principal data-protection statute(s) by
name + year in the "Data protection laws in <country>" section of
``?t=law&c=<ISO2>``. Unlike UNCTAD (presence only), this yields law NAMES — so it
is the first source that activates USE 1 (discovery-recall seeding) via
``to_discovery_seeds``, on top of USE 2 / USE 3.

Law-name extraction: an LLM extractor is the default (an A/B on SG/AU/MY/TH/ID
showed it is more accurate and, critically, GENERALISES to non-"Act" naming —
e.g. Indonesia's "Law No. 27 of 2022", which a keyword-anchored regex misses
entirely — while staying cleaner on AU/MY). A deterministic regex is the fallback
when the LLM backend is unavailable (offline / no endpoint), mirroring the
project's optional-LLM contract.
"""
from __future__ import annotations

import os
import re
from collections.abc import Sequence

import httpx

from lexora.collect.secondary.base import cached_json, register_source
from lexora.models.indicator import RDTIIIndicator
from lexora.models.secondary import Presence, SecondarySignal

_URL = "https://www.dlapiperdataprotection.com/index.html?t=law&c={iso2}"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36"
_SOURCE = "DLA Piper Data Protection Laws of the World"

# Capitalized run (allowing connectives + a "(Amendment)"-style paren) ending in a
# statute keyword and optional year. The deterministic FALLBACK only; it cannot
# catch non-"Act" naming (that is exactly why the LLM is the default).
_NAME = r"(?:[A-Z][A-Za-z]+|\((?:Amendment|Cth|No[^)]*|[A-Z][a-z]+)[^)]*\))"
_CON = r"(?:and|of|the|on|for|&|Other)"
_LAW_RE = re.compile(
    rf"({_NAME}(?:\s+(?:{_NAME}|{_CON}))*\s+(?:Act|Ordinance)"
    rf"(?:\s+(?:of\s+)?\d{{4}})?(?:\s*\(No\.?\s*\d+\s+of\s+\d{{4}}\))?)"
)
_JUNK = {"act", "the act", "amendment act", "privacy act amendment act"}

_LLM_SYSTEM = (
    "You extract the PRINCIPAL personal-data-protection / privacy statutes named in "
    "the text. Rules: copy each name VERBATIM from the text including its year; "
    "include amendment acts if named; EXCLUDE sectoral or unrelated laws mentioned "
    "only in passing, guidelines, principles, and laws of OTHER jurisdictions cited "
    "for comparison; use ONLY the provided text, no outside knowledge; if none, "
    'return []. Respond as JSON: {"laws": ["...", "..."]}.'
)


def _clean(names: list[str]) -> list[str]:
    """Strip prose lead-ins, drop junk/dupes (case-insensitive), keep order."""
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        n = re.sub(r"^(the|a|via|under|and)\s+", "", (n or "").strip(), flags=re.I).strip()
        low = n.lower()
        if len(n) < 8 or low in _JUNK or low in seen:
            continue
        seen.add(low)
        out.append(n)
    return out


def extract_laws_regex(text: str) -> list[str]:
    """Deterministic fallback: capitalized-run + Act/Ordinance + year."""
    return _clean(_LAW_RE.findall(text))


def extract_laws_llm(text: str, client=None) -> list[str] | None:
    """LLM extraction (default). Returns a cleaned list, or None when the backend
    is unavailable / errors (caller then falls back to the regex)."""
    if client is None:
        from lexora.classify.llm_client import LlmClient, is_available

        if not is_available():
            return None
        try:
            client = LlmClient(timeout=60)
        except Exception:
            return None
    try:
        data = client.chat(_LLM_SYSTEM, f"TEXT:\n{text}", json_schema={})
    except Exception:
        return None
    laws = data.get("laws", []) if isinstance(data, dict) else []
    if not isinstance(laws, list):
        return None
    return _clean([str(x) for x in laws])


def extract_laws(text: str, *, use_llm: bool = True, client=None) -> list[str]:
    """LLM first (most accurate + generalises), deterministic regex as fallback."""
    if use_llm:
        laws = extract_laws_llm(text, client)
        if laws is not None:
            return laws
    return extract_laws_regex(text)


def law_section_text(html: str) -> str:
    """Pull the 'Data protection laws in <country>' prose from a DLA Piper page.

    The page is a tab SPA; the law narrative is the lead paragraphs that name the
    statute. We take the first few substantial <p> that mention a statute/privacy
    term — robust across countries without depending on the panel id."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    ps = []
    for p in soup.find_all("p"):
        t = p.get_text(" ", strip=True)
        if len(t) > 120 and re.search(r"\b(Act|Ordinance|Regulation|Law|privacy|data protection)\b", t, re.I):
            ps.append(t)
        if len(ps) >= 3:
            break
    return re.sub(r"\s+", " ", " ".join(ps))[:1600]


def _fetch_law_text(
    economy: str, *, client: httpx.Client | None = None, timeout: float = 30.0,
    cache_path: str | None = None, ttl_hours: float = 24.0,
) -> str:
    """Fetch + 24h-cache the law-section text for one economy (ISO2)."""
    iso = economy.upper()
    cache_path = cache_path or os.path.join("outputs", "cache", f"dla_piper_{iso.lower()}.json")

    def _produce() -> dict:
        owns = client is None
        cl = client or httpx.Client(
            follow_redirects=True, timeout=timeout, headers={"User-Agent": _UA}
        )
        try:
            resp = cl.get(_URL.format(iso2=iso))
            if resp.status_code != 200:
                return {}
            return {"text": law_section_text(resp.text)}
        except Exception:
            return {}
        finally:
            if owns:
                cl.close()

    blob = cached_json(cache_path, ttl_hours, _produce)
    return (blob or {}).get("text", "")


@register_source("dla_piper")
def dla_piper_data_protection(
    economy: str,
    indicators: Sequence[RDTIIIndicator],
    *,
    data: str | None = None,
    use_llm: bool = True,
    llm_client=None,
    **fetch_kwargs,
) -> list[SecondarySignal]:
    """Emit a signal per (data-protection law name x in-scope indicator).

    ``data`` short-circuits the fetch with the law-section TEXT (offline tests).
    Each named statute becomes a presence=yes signal carrying ``primary_law_name``
    — the seed that finally drives USE 1. Covers P7-I1 (comprehensive framework)
    and P7-I4 (the DPO/DPIA regime that lives inside that same law)."""
    text = data if data is not None else _fetch_law_text(economy, **fetch_kwargs)
    if not text:
        return []
    laws = extract_laws(text, use_llm=use_llm, client=llm_client)
    if not laws:
        return []
    iso = economy.upper()
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
                    source_url=_URL.format(iso2=iso),
                    raw_snippet=f"DLA Piper names: {law}",
                )
            )
    return out


__all__ = [
    "dla_piper_data_protection",
    "extract_laws",
    "extract_laws_regex",
    "extract_laws_llm",
    "law_section_text",
]
