"""Per-portal discovery strategies.

The generic harvester (``discovery.harvest_candidates``) handles any
server-rendered or browser-rendered results page. Some portals expose a
structured API that is far more reliable than scraping their SPA — Australia's
Federal Register of Legislation is the prime case: its search UI is a JavaScript
SPA whose harvested links are a browse-all list, but it is backed by a clean
OData API.

A strategy is a callable with the same return type as the harvester
(``list[DiscoveryResult]``). :func:`strategy_for` matches a portal to a strategy
by host; :func:`discovery.discover` uses the strategy when one matches and falls
back to generic harvesting when the strategy yields nothing.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from urllib.parse import quote, urlencode, urlparse

import httpx

from lexora.collect.discovery import (
    TAG_KNOWN,
    TAG_NEW,
    DiscoveryResult,
    _fuzzy_known,
)
from lexora.models.source import PortalSpec

Strategy = Callable[..., list[DiscoveryResult]]

_AU_API = "https://api.prod.legislation.gov.au/v1/titles"
_AU_DOC = "https://www.legislation.gov.au/{id}/{point}"

_MY_API = "https://lom.agc.gov.my/fess-proxy.php"


def _odata_escape(value: str) -> str:
    """Escape a string literal for an OData filter (single quote -> doubled)."""
    return value.replace("'", "''")


def _first_href(snippet: str | None) -> str | None:
    """Extract the first href URL from an HTML anchor snippet (MY download cell)."""
    if not snippet:
        return None
    m = re.search(r'href="([^"]+)"', snippet)
    return m.group(1) if m else None


def _tag_for(fuzzy: float, known: list[str]) -> str | None:
    """KNOWN / NEW tag from a fuzzy-to-known score (shared by API strategies)."""
    if not known:
        return None
    if fuzzy >= 0.80:
        return TAG_KNOWN
    if fuzzy < 0.55:
        return TAG_NEW
    return None


def _resolve_tag(
    fuzzy: float, matched: str | None, ident: str, known: list[str],
    ids: dict[str, str] | None,
) -> tuple[str | None, str | None]:
    """Tag + matched-name, preferring an identity hit (e.g. Act number / doc id)
    over fuzzy name matching — so filename-titled records still resolve KNOWN."""
    id_name = (ids or {}).get(ident)
    if id_name:
        return TAG_KNOWN, id_name
    tag = _tag_for(fuzzy, known)
    return tag, (matched if tag == TAG_KNOWN else None)


def au_legislation_api(
    portal: PortalSpec,
    *,
    query: str | None,
    limit: int = 10,
    min_score: float = 0.1,
    timeout: float = 30.0,
    known_instruments: list[str] | None = None,
    known_instrument_ids: dict[str, str] | None = None,
    client: httpx.Client | None = None,
) -> list[DiscoveryResult]:
    """Discover AU instruments via the Federal Register OData API.

    Queries ``contains(tolower(name), '<query>')``, then ranks client-side
    favouring principal, in-force titles and fuzzy match to the query / known
    names. Returns canonical ``/latest`` (or ``/asmade``) document URLs.

    Honest limitation (verified 2026-06-04): the AU OData API has NO usable
    full-text search — ``$search`` is a no-op (returns all 131k titles in id
    order regardless of term) and the SSR search page ignores the query. So this
    is *name-driven*: it resolves a query that contains (part of) the instrument
    name. Indicator-concept queries that share no tokens with the law name (e.g.
    "cross-border disclosure" -> Privacy Act 1988) cannot be discovered here;
    that bridge needs a semantic/LLM crosswalk and is out of scope.
    """
    query = query or portal.search_query or ""
    if not query:
        return []
    known = known_instruments or []

    flt = f"contains(tolower(name),'{_odata_escape(query.lower())}')"
    url = (
        f"{_AU_API}?%24filter={quote(flt, safe='(),')}"
        f"&%24top=25&%24select=id,name,collection,isPrincipal,isInForce,status"
    )

    owns = client is None
    client = client or httpx.Client(
        follow_redirects=True, timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    try:
        resp = client.get(url)
        if resp.status_code != 200:
            return []
        values = resp.json().get("value", [])
    except Exception:
        return []
    finally:
        if owns:
            client.close()

    results: list[DiscoveryResult] = []
    for v in values:
        name = v.get("name", "")
        q_sim = _fuzzy_known(name, [query])[0] if query else 0.0
        fuzzy, matched = _fuzzy_known(name, known) if known else (0.0, None)
        relevance = max(q_sim, fuzzy)
        score = min(1.0, 0.8 * relevance + 0.1 * bool(v.get("isPrincipal")) + 0.1 * bool(v.get("isInForce")))
        if score < min_score:
            continue
        tag, matched_name = _resolve_tag(fuzzy, matched, v["id"], known, known_instrument_ids)
        point = "latest" if v.get("isInForce") else "asmade"
        results.append(
            DiscoveryResult(
                url=_AU_DOC.format(id=v["id"], point=point),
                title=name,
                source_type=portal.source_type,
                score=score,
                via="api",
                is_pdf_link=False,
                discovery_tag=tag,
                matched_instrument=matched_name,
                n_variants=1,
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


def _my_act_id(doc: dict, os_url: str, title: str) -> str:
    """Instrument identity for a MY Fess doc.

    Fess returns several records per Act — a catalog record (``act-detail.php?act=709``,
    clean title) and document records (PDFs, filename titles like ``Act 709 ori.pdf``).
    They must collapse onto one instrument keyed by the Act number so a full-text
    (indicator) query and a name query agree on the same instrument.
    """
    # 1) catalog URL `act=709` is the most reliable.
    m = re.search(r"act=([0-9]+|A\d+)", os_url, re.I)
    if m:
        return m.group(1).upper()
    # 2) "Act 709" in the filename/title — but NOT a 4-digit year (the name's year,
    #    e.g. "...Act 2010"). Document records' metadata holds the gazette P.U.
    #    number, not the Act number, so we trust the "Act <n>" token instead.
    for mm in re.finditer(r"\bact[\s_]+([0-9]{1,4}|A\d+)\b", f"{title} {os_url}", re.I):
        val = mm.group(1)
        if re.fullmatch(r"(19|20)\d{2}", val):
            continue
        return val.upper()
    # 3) a bare numeric metadata field, if any.
    for k in ("actNo", "nombor_akta"):
        v = str(doc.get(k) or "").strip()
        if re.fullmatch(r"[0-9]{1,4}|A\d+", v, re.I):
            return v.upper()
    return os_url


def _my_is_clean_title(title: str) -> bool:
    """True if `title` looks like an Act name rather than a filename."""
    if not title or re.search(r"\.(pdf|indd|docx?|tiff?)\b", title, re.I):
        return False
    return bool(re.search(r"\bact\b", title, re.I) and re.search(r"\d{4}", title))


def my_legislation_api(
    portal: PortalSpec,
    *,
    query: str | None,
    limit: int = 10,
    min_score: float = 0.1,
    timeout: float = 30.0,
    known_instruments: list[str] | None = None,
    known_instrument_ids: dict[str, str] | None = None,
    client: httpx.Client | None = None,
) -> list[DiscoveryResult]:
    """Discover MY instruments via the AGC Fess/Solr proxy (``fess-proxy.php``).

    The lom.agc.gov.my search box is a DataTables widget backed by a Fess/Solr
    JSON proxy that does real full-text search, so this works for both name and
    indicator-phrased queries. Each Act surfaces as several docs (a catalog record
    ``act-detail.php?act=709`` with a clean title, plus PDF document records with
    filename titles); they are collapsed by Act number. Ranking combines Solr's
    own relevance order, fuzzy match to the query / known names, and a principal
    ("IBU") preference. The Act PDF is captured as ``fulltext_url``.
    """
    query = query or portal.search_query or ""
    if not query:
        return []
    known = known_instruments or []

    params = {
        "q": query, "fq": "", "start": "0", "rows": "25",
        "sort": "", "lookup": "all", "kategori": "all", "draw": "1",
    }
    url = f"{_MY_API}?{urlencode(params)}"

    owns = client is None
    client = client or httpx.Client(
        follow_redirects=True, timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    try:
        resp = client.get(url)
        if resp.status_code != 200:
            return []
        docs = resp.json().get("response", {}).get("docs", [])
    except Exception:
        return []
    finally:
        if owns:
            client.close()

    span = max(1, len(docs) - 1)
    groups: dict[str, dict] = {}
    for idx, d in enumerate(docs):
        os_url = d.get("_os_url")
        if not os_url:
            continue
        title = (
            d.get("titleBI") or d.get("attr_titlebi")
            or d.get("titleBM") or d.get("attr_titlebm") or ""
        )
        act_id = _my_act_id(d, os_url, title)
        is_pdf = os_url.lower().endswith(".pdf")
        pdf_url = os_url if is_pdf else _first_href(d.get("DOC2DOWNLOADBI") or d.get("DOC2DOWNLOADBM"))
        g = groups.setdefault(
            act_id,
            {"solr_rel": 0.0, "principal": False, "clean_title": "", "any_title": "",
             "url": "", "fulltext": None, "count": 0},
        )
        g["count"] += 1
        g["solr_rel"] = max(g["solr_rel"], 1.0 - idx / span)
        g["principal"] = g["principal"] or d.get("legislationStatus") == "IBU" or "PRINCIPAL" in os_url
        g["any_title"] = g["any_title"] or title
        if _my_is_clean_title(title) and len(title) > len(g["clean_title"]):
            g["clean_title"] = title
        # canonical link: always prefer the act-detail page; else first seen wins.
        if "act-detail" in os_url or not g["url"]:
            g["url"] = os_url
        if pdf_url and not g["fulltext"]:
            g["fulltext"] = pdf_url

    results: list[DiscoveryResult] = []
    for act_id, g in groups.items():
        title = g["clean_title"] or g["any_title"]
        q_sim = _fuzzy_known(title, [query])[0] if query else 0.0
        fuzzy, matched = _fuzzy_known(title, known) if known else (0.0, None)
        # Solr relevance + lexical match + principal preference, normalized to [0,1].
        score = min(1.0, 0.4 * g["solr_rel"] + 0.4 * max(q_sim, fuzzy) + 0.2 * g["principal"])
        if score < min_score:
            continue
        # Act-number identity tags KNOWN even when the title is just a filename.
        tag, matched_name = _resolve_tag(fuzzy, matched, act_id, known, known_instrument_ids)
        results.append(
            DiscoveryResult(
                url=g["url"], title=title, source_type=portal.source_type, score=score,
                via="api", is_pdf_link=g["url"].lower().endswith(".pdf"),
                discovery_tag=tag, matched_instrument=matched_name,
                fulltext_url=g["fulltext"], n_variants=g["count"],
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


# host substring -> strategy
STRATEGIES: dict[str, Strategy] = {
    "legislation.gov.au": au_legislation_api,
    "lom.agc.gov.my": my_legislation_api,
}


def strategy_for(portal: PortalSpec) -> Strategy | None:
    """Return the registered strategy for a portal's host, or None."""
    host = urlparse(str(portal.url)).netloc.lower()
    for needle, strat in STRATEGIES.items():
        if needle in host:
            return strat
    return None


__all__ = [
    "Strategy",
    "au_legislation_api",
    "my_legislation_api",
    "STRATEGIES",
    "strategy_for",
]
