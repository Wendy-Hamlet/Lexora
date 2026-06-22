"""World Map of Encryption Laws and Policies adapter (WS-S) — Global Partners Digital.

The RDTII guide names this map under P7-I5 (government access to personal data):
it documents, per country, the laws governing encryption and a government's lawful
access to encrypted data / decryption keys. Unlike the STRI indices it NAMES laws,
so — like DLA Piper / ICLG — it can feed USE 1 (discovery-recall seeding) on top of
USE 2 / USE 3.

The whole map is ONE server-rendered page (no JS framework): each country is an
``<h4 class="h2">Country</h4>`` block followed by ``<h5>`` sub-sections (general
right to encryption, licensing, import/export controls, other restrictions, …) and
prose. We fetch the page once (24h cache), split it into per-country sections, and
extract the named laws for the requested economy. A country-specific LLM prompt
(encryption / lawful-access framing) is the default extractor with the shared
deterministic regex as the offline fallback.
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

_URL = "https://www.gp-digital.org/world-map-of-encryption/"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36"
_SOURCE = "World Map of Encryption Laws and Policies"

# ISO2 -> the country name as it appears in the page's <h4 class="h2"> heading.
# Curated for the in-scope + common economies; an unmapped ISO is skipped, and a
# name that doesn't match any heading simply yields nothing (never wrong data).
_ISO_TO_COUNTRY = {
    "SG": "Singapore", "AU": "Australia", "MY": "Malaysia", "TH": "Thailand",
    "ID": "Indonesia", "PH": "Philippines", "IN": "India", "JP": "Japan",
    "CN": "China", "GB": "United Kingdom", "US": "United States",
}

# Encryption / lawful-access framing — distinct from the data-protection prompt.
_LLM_SYSTEM = (
    "You extract the NAMED laws, regulations or ordinances of the country that is "
    "the SUBJECT of the text that govern ENCRYPTION or a GOVERNMENT's lawful access "
    "to encrypted data, decryption keys or communications. Output rules:\n"
    "- Copy each title VERBATIM from the text, INCLUDING its year; each item must be "
    "a COMPLETE statute title, never a truncated fragment.\n"
    "- Return ONLY laws of the subject country, never another country's law.\n"
    "- Use ONLY the provided text; no outside knowledge. If none, return [].\n"
    'Respond as JSON: {"laws": ["...", "..."]}.'
)


def parse_country_sections(html: str) -> dict[str, str]:
    """Split the single map page into ``{country_name_lower: prose}``.

    Each country starts at an ``<h4 class="h2">`` heading and runs until the next
    one; we concatenate the text of the elements in between (the <h5> sub-headings
    and their paragraphs) into one prose blob per country."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    heads = soup.find_all("h4", class_="h2")
    out: dict[str, str] = {}
    for h in heads:
        country = h.get_text(" ", strip=True)
        if not country:
            continue
        parts: list[str] = []
        for sib in h.next_elements:
            # stop when we reach the next country heading
            if getattr(sib, "name", None) == "h4" and "h2" in (sib.get("class") or []):
                break
            if getattr(sib, "name", None) in ("p", "li", "h5"):
                t = sib.get_text(" ", strip=True)
                if t:
                    parts.append(t)
            if len(" ".join(parts)) > 2000:
                break
        out[country.lower()] = re.sub(r"\s+", " ", " ".join(parts))[:2000]
    return out


def country_section_text(html: str, country: str) -> str:
    """The encryption-law prose for one country, or '' if it isn't on the map."""
    return parse_country_sections(html).get(country.strip().lower(), "")


def extract_encryption_laws(text: str, *, use_llm: bool = True, client=None) -> list[str]:
    """Encryption / lawful-access law names: LLM (country-framed) first, regex fallback."""
    if use_llm:
        laws = _extract_llm(text, client)
        if laws is not None:
            return laws
    return extract_laws_regex(text)


def _extract_llm(text: str, client=None) -> list[str] | None:
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
    # reuse the shared cleaner via the regex module's _clean indirectly: dedupe here.
    out, seen = [], set()
    for x in laws:
        n = str(x).strip()
        if len(n) >= 8 and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def _fetch_section_text(
    economy: str, *, client: httpx.Client | None = None, timeout: float = 40.0,
    cache_path: str | None = None, ttl_hours: float = 24.0,
) -> str:
    """Fetch + 24h-cache the whole map (parsed per-country) and return one economy's
    section. The cache is shared across economies (one page serves all countries)."""
    iso = economy.upper()
    country = _ISO_TO_COUNTRY.get(iso)
    if not country:
        return ""
    cache_path = cache_path or os.path.join("outputs", "cache", "world_map_encryption.json")

    def _produce() -> dict:
        owns = client is None
        cl = client or httpx.Client(
            follow_redirects=True, timeout=timeout, headers={"User-Agent": _UA}
        )
        try:
            resp = cl.get(_URL)
            if resp.status_code != 200:
                return {}
            return parse_country_sections(resp.text)
        except Exception:
            return {}
        finally:
            if owns:
                cl.close()

    sections = cached_json(cache_path, ttl_hours, _produce) or {}
    return sections.get(country.lower(), "")


@register_source("world_map_encryption")
def world_map_encryption(
    economy: str,
    indicators: Sequence[RDTIIIndicator],
    *,
    data: str | None = None,
    use_llm: bool = True,
    llm_client=None,
    **fetch_kwargs,
) -> list[SecondarySignal]:
    """Emit a P7-I5 signal per named encryption / lawful-access law.

    ``data`` short-circuits the fetch with the country's section TEXT (offline
    tests). Each named law becomes a presence=yes signal carrying
    ``primary_law_name`` — a USE-1 seed for the government-access indicator."""
    text = data if data is not None else _fetch_section_text(economy, **fetch_kwargs)
    if not text:
        return []
    laws = extract_encryption_laws(text, use_llm=use_llm, client=llm_client)
    if not laws:
        return []
    iso = economy.upper()
    wanted = {i.submission_id for i in indicators}
    if wanted and "P7-I5" not in wanted:
        return []
    out: list[SecondarySignal] = []
    for law in laws:
        out.append(
            SecondarySignal(
                economy=iso,
                indicator_id="P7-I5",
                source_name=_SOURCE,
                presence=Presence.yes,
                primary_law_name=law,
                source_url=_URL,
                raw_snippet=f"World Map of Encryption names: {law}",
            )
        )
    return out


__all__ = [
    "world_map_encryption",
    "parse_country_sections",
    "country_section_text",
    "extract_encryption_laws",
]
