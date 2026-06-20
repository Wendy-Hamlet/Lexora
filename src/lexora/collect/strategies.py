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

import json
import os
import re
import time
from collections.abc import Callable
from urllib.parse import quote, urlencode, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from lexora.collect.discovery import (
    TAG_KNOWN,
    TAG_NEW,
    DiscoveryResult,
    _fuzzy_known,
)
from lexora.models.source import PortalSpec, SourceType

Strategy = Callable[..., list[DiscoveryResult]]

_AU_API = "https://api.prod.legislation.gov.au/v1/titles"
_AU_DOC = "https://www.legislation.gov.au/{id}/{point}"
_AU_PAGE = 100  # OData max $top (200+ -> HTTP 400)

_MY_API = "https://lom.agc.gov.my/fess-proxy.php"


def _odata_escape(value: str) -> str:
    """Escape a string literal for an OData filter (single quote -> doubled)."""
    return value.replace("'", "''")


def _get_with_retry(
    client: httpx.Client, url: str, *, retries: int = 2, backoff: float = 1.0
) -> httpx.Response:
    """GET ``url`` retrying on transport errors with exponential backoff.

    MY's Fess proxy (lom.agc.gov.my) intermittently drops the TLS connection
    mid-handshake/read, and the AU OData catalogue paging (~48 sequential GETs)
    is similarly exposed to a transient blip. ``httpx.TransportError`` covers the
    connect/read/SSL family; only it is retried (an HTTP 4xx/5xx is a real
    response and returned as-is). The final error propagates after the last
    attempt, where the caller turns it into a graceful empty result."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return client.get(url)
        except httpx.TransportError as exc:
            last = exc
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    raise last or httpx.TransportError(f"failed to fetch {url}")


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
        f"&%24top=25&%24select=id,name,collection,isPrincipal,isInForce,status,number,year"
    )

    owns = client is None
    client = client or httpx.Client(
        follow_redirects=True, timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    try:
        resp = _get_with_retry(client, url)
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
        # Drop subsidiary instruments (Determinations / Notices / Guidelines, which
        # come back as `LegislativeInstrument` / `Gazette`): a name query for
        # "Privacy Act 1988" otherwise returns 20+ of its determinations, all
        # non-statute and mostly not-in-force, which crowd real statutes out of the
        # budget. The statute targets (incl. amendment Acts) are all collection 'Act'.
        if v.get("collection") != "Act":
            continue
        name = v.get("name", "")
        q_sim = _fuzzy_known(name, [query])[0] if query else 0.0
        fuzzy, matched = _fuzzy_known(name, known) if known else (0.0, None)
        relevance = max(q_sim, fuzzy)
        score = min(1.0, 0.8 * relevance + 0.1 * bool(v.get("isPrincipal")) + 0.1 * bool(v.get("isInForce")))
        if score < min_score:
            continue
        tag, matched_name = _resolve_tag(fuzzy, matched, v["id"], known, known_instrument_ids)
        point = "latest" if v.get("isInForce") else "asmade"
        # Official law number straight from the register: "No. 119 of 1988". The
        # register `id` (C2004A03712) is its compilation/series ref — kept as the
        # fallback ref when the act number is absent (e.g. some instruments).
        num, yr = v.get("number"), v.get("year")
        law_number = f"No. {num} of {yr}" if num and yr else (v.get("id") or "")
        # Authoritative lifecycle from the register itself (drives enforced-only):
        # the `isInForce` flag is machine-readable; `status` is its label fallback.
        from lexora.classify.lifecycle import detect_status

        status = detect_status(
            in_force=bool(v.get("isInForce")) if v.get("isInForce") is not None else None,
            portal_status=str(v.get("status") or ""),
        ).value
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
                law_number=law_number,
                status=status,
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


def au_act_catalogue(
    *,
    client: httpx.Client | None = None,
    timeout: float = 30.0,
    cache_path: str | None = None,
    ttl_hours: float = 24.0,
    max_pages: int = 60,
) -> list[dict]:
    """Fetch the in-force AU Act catalogue (``[{id, name, isPrincipal}]``).

    The OData ``$search`` is a no-op, so concept queries can't be sent to the
    server (see :func:`au_legislation_api`). The semantic crosswalk instead pulls
    the whole in-force Act list once and matches it locally. ``$top`` caps at 100,
    so this pages with ``$skip`` (~48 requests for the ~4.7k in-force Acts) and
    caches the result to disk for ``ttl_hours`` to keep repeat runs cheap.
    """
    cache_path = cache_path or os.path.join("outputs", "cache", "au_act_catalogue.json")
    if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path)) < ttl_hours * 3600:
        try:
            with open(cache_path, encoding="utf-8") as fh:
                cached = json.load(fh)
            if cached:
                return cached
        except Exception:
            pass

    flt = quote("collection eq 'Act' and isInForce eq true", safe="(),")
    owns = client is None
    client = client or httpx.Client(
        follow_redirects=True, timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    catalogue: list[dict] = []
    try:
        for page in range(max_pages):
            url = (
                f"{_AU_API}?%24filter={flt}&%24top={_AU_PAGE}&%24skip={page * _AU_PAGE}"
                f"&%24select=id,name,isPrincipal"
            )
            resp = _get_with_retry(client, url)
            if resp.status_code != 200:
                break
            values = resp.json().get("value", [])
            if not values:
                break
            catalogue.extend(
                {"id": v["id"], "name": v.get("name", ""), "isPrincipal": bool(v.get("isPrincipal"))}
                for v in values if v.get("id") and v.get("name")
            )
            if len(values) < _AU_PAGE:
                break
    except Exception:
        return catalogue
    finally:
        if owns:
            client.close()

    if catalogue:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(catalogue, fh, ensure_ascii=False)
        except Exception:
            pass
    return catalogue


def au_semantic_crosswalk(
    portal: PortalSpec,
    indicators: list,
    *,
    embedder,
    top_k: int = 6,
    min_sim: float = 0.50,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
    known_instruments: list[str] | None = None,
    known_instrument_ids: dict[str, str] | None = None,
    catalogue: list[dict] | None = None,
) -> list[DiscoveryResult]:
    """Bridge RDTII concepts to AU statute titles by dense similarity.

    AU's name-only portal can never surface a statute that isn't already in the
    known list (a concept query shares no tokens with the law name). This closes
    that gap: embed every in-force Act title once, embed each indicator's concept
    text, and surface the ``top_k`` titles above ``min_sim`` as candidates,
    attributed to the indicator whose concept found them. Hits that fuzzy-match a
    known instrument are tagged KNOWN; the rest are NEW — which is how AU gets any
    NEW discovery at all.
    """
    from lexora.semantic.embedder import cosine_topk

    catalogue = catalogue if catalogue is not None else au_act_catalogue(
        client=client, timeout=timeout
    )
    if not catalogue or not indicators:
        return []

    titles = [c["name"] for c in catalogue]
    title_vecs = embedder.encode(titles)
    concept_texts = [
        " ".join([ind.name, ind.description, *ind.query_phrases()]) for ind in indicators
    ]
    concept_vecs = embedder.encode(concept_texts)
    known = known_instruments or []

    agg: dict[str, DiscoveryResult] = {}
    for ind, qvec in zip(indicators, concept_vecs, strict=True):
        for doc_idx, sim in cosine_topk(qvec, title_vecs, top_k):
            if sim < min_sim:
                continue
            entry = catalogue[doc_idx]
            name = entry["name"]
            fuzzy, matched = _fuzzy_known(name, known) if known else (0.0, None)
            tag, matched_name = _resolve_tag(
                fuzzy, matched, entry["id"], known, known_instrument_ids
            )
            tag = tag or TAG_NEW  # a concept-surfaced title we don't already know
            url = _AU_DOC.format(id=entry["id"], point="latest")
            cur = agg.get(entry["id"])
            if cur is None:
                agg[entry["id"]] = DiscoveryResult(
                    url=url, title=name, source_type=portal.source_type,
                    score=float(sim), via="api", is_pdf_link=False,
                    discovery_tag=tag, matched_instrument=matched_name,
                    indicator_hits=[ind.submission_id],
                )
            else:
                cur.score = max(cur.score, float(sim))
                if ind.submission_id not in cur.indicator_hits:
                    cur.indicator_hits.append(ind.submission_id)
                if tag == TAG_KNOWN:
                    cur.discovery_tag = TAG_KNOWN
                    cur.matched_instrument = cur.matched_instrument or matched_name
    for res in agg.values():
        res.indicator_hits.sort()
    return sorted(agg.values(), key=lambda r: r.score, reverse=True)


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
        resp = _get_with_retry(client, url)
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


# --- Regulator-portal connectors (secondary sources: guidance / codes / notices) ---
# Unlike the query-driven STRATEGIES above (which search a statute portal for one
# query), a connector harvests a regulator portal's *guidance corpus* once: these
# documents (advisory guidelines, codes of practice, licence conditions) are the
# subsidiary-instrument gold rows that never appear on the statute portals, and
# discovering them is pure NEW-evidence. A connector takes the full indicator list
# (not a single query) and returns secondary DiscoveryResults.

PortalConnector = Callable[..., list[DiscoveryResult]]

# Anchor texts that are navigation, not documents.
_GUIDANCE_SKIP = {
    "", "skip to content", "skip to main content", "next", "previous",
    "back to top", "read more", "home", "organisations", "individuals",
    "government agencies", "health service providers", "more guidance",
}


def _collect_guidance(
    html: str,
    base_url: str,
    *,
    include: Callable[[str], bool],
    known: list[str],
    via: str = "http",
    clean_title: Callable[[str], str] | None = None,
    hubs: tuple[str, ...] = (),
    out: dict[str, DiscoveryResult] | None = None,
) -> dict[str, DiscoveryResult]:
    """Harvest guidance/document links from a regulator hub page into ``out``.

    Shared by every connector: pick anchors whose URL ``include(url)`` accepts
    (and that aren't a hub or a ``?`` facet link), clean the title, dedup by URL
    keeping the longest title, and tag KNOWN only when the title fuzzy-matches a
    known instrument (guidance is otherwise NEW). All results are *secondary*.
    """
    soup = BeautifulSoup(html, "lxml")
    hubset = {h.rstrip("/") for h in hubs}
    out = out if out is not None else {}
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0].strip()
        if not href or "?" in href:
            continue
        url = urljoin(base_url, href)
        if url.rstrip("/") in hubset or not include(url):
            continue
        raw = " ".join(a.get_text(" ", strip=True).split())
        title = clean_title(raw) if clean_title else raw
        if len(title) < 8 or title.lower() in _GUIDANCE_SKIP:
            continue
        fuzzy, matched = _fuzzy_known(title, known) if known else (0.0, None)
        tag = TAG_KNOWN if fuzzy >= 0.80 else TAG_NEW
        cur = out.get(url)
        if cur is None or len(title) > len(cur.title):
            out[url] = DiscoveryResult(
                url=url, title=title, source_type=SourceType.secondary,
                score=1.0, via=via, is_pdf_link=url.lower().endswith(".pdf"),
                discovery_tag=tag, matched_instrument=(matched if tag == TAG_KNOWN else None),
            )
    return out


# --- SG PDPC (browser-rendered) ---
# PDPC publishes guidance as JS-rendered hub pages that list per-document detail
# pages; the detail-page titles carry the instrument name (+ a date / category
# prefix we strip). These two hubs are the canonical lists.
_PDPC_HUBS = (
    "https://www.pdpc.gov.sg/organisations/regulations-decisions/regulatory-guidance",
    "https://www.pdpc.gov.sg/organisations/resources/guidance-by-topic",
)
_PDPC_DETAIL = re.compile(r"/(?:regulatory-guidance|guidance-by-topic|resources)/[a-z0-9]", re.I)
# Leading "<Category> <DD Mon YYYY>" noise on a harvested guidance title.
_PDPC_TITLE_PREFIX = re.compile(
    r"^(?:Advisory Guidelines|Practical Guidance|Publications?|Templates?|"
    r"Training Courses|Tools?|Guides?)?\s*\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}\s+",
)


def _clean_guidance_title(text: str) -> str:
    """Strip the leading category + date prefix a PDPC hub prepends to a title."""
    cleaned = _PDPC_TITLE_PREFIX.sub("", " ".join(text.split())).strip()
    return cleaned or " ".join(text.split())


def pdpc_guidance(
    portal: PortalSpec,
    indicators: list,
    *,
    browser_session=None,
    limit: int = 40,
    timeout: float = 45.0,
    known_instruments: list[str] | None = None,
    known_instrument_ids: dict[str, str] | None = None,
) -> list[DiscoveryResult]:
    """Harvest the PDPC (Singapore) guidance corpus from its hub pages.

    Renders each hub with a headless browser (the list is JS-built), harvests the
    per-guideline detail links and their titles, strips the category/date prefix,
    and returns them as *secondary* candidates. Almost all are NEW (guidance is not
    in the statute known-list); a title that fuzzy-matches a known instrument is
    tagged KNOWN. ``indicator_hits`` is left empty — these are discovery-only
    context, and per-indicator mapping of secondary sources is a later step.
    """
    from lexora.collect.browser import is_available

    if browser_session is None and not is_available():
        return []
    known = known_instruments or []

    def _render(url: str) -> str:
        if browser_session is not None:
            return browser_session.render(url, timeout=timeout).html
        from lexora.collect.browser import render

        return render(url, timeout=timeout).html

    agg: dict[str, DiscoveryResult] = {}
    for hub in _PDPC_HUBS:
        try:
            html = _render(hub)
        except Exception:
            continue
        _collect_guidance(
            html, hub, include=lambda u: bool(_PDPC_DETAIL.search(u)), known=known,
            via="browser", clean_title=_clean_guidance_title, hubs=_PDPC_HUBS, out=agg,
        )
    return list(agg.values())[:limit]


# --- MY PDP / JPDP (server-rendered HTTP) ---
# pdp.gov.my serves static HTML. The "Code of Practice" list is exposed in the
# sidebar of each code page (not the bare hub), so we seed with the hub *and* a
# known code page and harvest the sibling `code-of-practice` links from both.
_MY_PDP_SEEDS = (
    "https://www.pdp.gov.my/ppdpv1/en/akta/code-of-practice/",
    "https://www.pdp.gov.my/ppdpv1/en/akta/"
    "personal-data-protection-code-of-practice-for-banking-sector-and-financial-institutions/",
)


def _my_is_code(url: str) -> bool:
    """A sector code-of-practice page (not the bare `/code-of-practice/` hub)."""
    u = url.lower().rstrip("/")
    return "code-of-practice" in u and not u.endswith("code-of-practice")


def my_pdp_guidance(
    portal: PortalSpec,
    indicators: list,
    *,
    browser_session=None,
    limit: int = 40,
    timeout: float = 45.0,
    known_instruments: list[str] | None = None,
    known_instrument_ids: dict[str, str] | None = None,
) -> list[DiscoveryResult]:
    """Harvest the Malaysian PDP sectoral Codes of Practice (banking, communications,
    healthcare, utilities). Server-rendered, so this uses plain HTTP."""
    known = known_instruments or []
    agg: dict[str, DiscoveryResult] = {}
    with httpx.Client(
        follow_redirects=True, timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36"},
    ) as client:
        for seed in _MY_PDP_SEEDS:
            try:
                html = client.get(seed).text
            except Exception:
                continue
            # Exclude only the bare `/code-of-practice/` listing hub — the banking
            # seed is itself a real code page (a gold target) and must stay.
            _collect_guidance(
                html, seed, include=_my_is_code, known=known,
                hubs=(_MY_PDP_SEEDS[0],), out=agg,
            )
    return list(agg.values())[:limit]


# --- AU OAIC (server-rendered HTTP) ---
_OAIC_HUBS = (
    "https://www.oaic.gov.au/privacy/privacy-guidance-for-organisations-and-government-agencies",
    "https://www.oaic.gov.au/privacy/australian-privacy-principles/australian-privacy-principles-guidelines",
)
_OAIC_INCLUDE = re.compile(
    r"oaic\.gov\.au/privacy/.*(privacy-impact|guidance|guidelines|data-breach|handling-personal)",
    re.I,
)


def oaic_guidance(
    portal: PortalSpec,
    indicators: list,
    *,
    browser_session=None,
    limit: int = 40,
    timeout: float = 45.0,
    known_instruments: list[str] | None = None,
    known_instrument_ids: dict[str, str] | None = None,
) -> list[DiscoveryResult]:
    """Harvest OAIC (Australia) privacy guidance — APP guidelines, Privacy Impact
    Assessment guidance, data-breach guidance. Server-rendered, so plain HTTP."""
    known = known_instruments or []
    agg: dict[str, DiscoveryResult] = {}
    with httpx.Client(
        follow_redirects=True, timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36"},
    ) as client:
        for hub in _OAIC_HUBS:
            try:
                html = client.get(hub).text
            except Exception:
                continue
            _collect_guidance(
                html, hub, include=lambda u: bool(_OAIC_INCLUDE.search(u)), known=known,
                hubs=_OAIC_HUBS, out=agg,
            )
    return list(agg.values())[:limit]


# host substring -> strategy
STRATEGIES: dict[str, Strategy] = {
    "legislation.gov.au": au_legislation_api,
    "lom.agc.gov.my": my_legislation_api,
}

# host substring -> regulator-portal guidance connector
PORTAL_CONNECTORS: dict[str, PortalConnector] = {
    "pdpc.gov.sg": pdpc_guidance,
    "pdp.gov.my": my_pdp_guidance,
    "oaic.gov.au": oaic_guidance,
}


def connector_for(portal: PortalSpec) -> PortalConnector | None:
    """Return the registered guidance connector for a portal's host, or None."""
    host = urlparse(str(portal.url)).netloc.lower()
    for needle, conn in PORTAL_CONNECTORS.items():
        if needle in host:
            return conn
    return None

# host substring -> concept (semantic) crosswalk, used by name-only portals to
# discover statutes that share no tokens with the indicator's concept phrasing.
CONCEPT_STRATEGIES: dict[str, Callable[..., list[DiscoveryResult]]] = {
    "legislation.gov.au": au_semantic_crosswalk,
}


def strategy_for(portal: PortalSpec) -> Strategy | None:
    """Return the registered strategy for a portal's host, or None."""
    host = urlparse(str(portal.url)).netloc.lower()
    for needle, strat in STRATEGIES.items():
        if needle in host:
            return strat
    return None


def concept_strategy_for(portal: PortalSpec) -> Callable[..., list[DiscoveryResult]] | None:
    """Return the registered semantic-crosswalk strategy for a portal, or None."""
    host = urlparse(str(portal.url)).netloc.lower()
    for needle, strat in CONCEPT_STRATEGIES.items():
        if needle in host:
            return strat
    return None


# --- Per-portal full-text (PDF) resolvers (two-stage discovery, stage 2) ---
# Some portals serve the full-text PDF from a URL that isn't a plain `.pdf` link
# on the page, so the generic `.pdf`-harvest in discovery.resolve_fulltext can't
# find it. These resolvers know each portal's PDF convention.

_AU_PDF_PATH = re.compile(r"/[^\"'\s]+/text/\w+/pdf")


def sg_resolve_fulltext(result, *, timeout: float = 30.0) -> str | None:
    """SG SSO serves the whole-Act PDF at ``<act-url>?ViewType=Pdf`` (fetchable
    with a browser UA even though the HTML landing 403s bots)."""
    base = result.url.split("?")[0].rstrip("/")
    return f"{base}?ViewType=Pdf" if "/Act/" in base else None


def au_resolve_fulltext(result, *, timeout: float = 30.0) -> str | None:
    """AU FRL serves the latest-compilation PDF at a dated path
    ``/{id}/{date}/{date}/text/original/pdf`` that is embedded in the rendered
    downloads page (the date is the latest compilation, so APP-era amendments are
    included — unlike the as-made PDF)."""
    from urllib.parse import urljoin

    from lexora.collect.browser import render

    downloads = result.url.rstrip("/")
    if not downloads.endswith("/downloads"):
        downloads += "/downloads"
    try:
        html = render(downloads, timeout=timeout).html
    except Exception:
        return None
    m = _AU_PDF_PATH.search(html)
    return urljoin("https://www.legislation.gov.au/", m.group(0)) if m else None


RESOLVERS: dict[str, Callable[..., str | None]] = {
    "sso.agc.gov.sg": sg_resolve_fulltext,
    "legislation.gov.au": au_resolve_fulltext,
}


def resolver_for(url: str) -> Callable[..., str | None] | None:
    """Return the registered full-text resolver for a URL's host, or None."""
    host = urlparse(url).netloc.lower()
    for needle, fn in RESOLVERS.items():
        if needle in host:
            return fn
    return None


__all__ = [
    "Strategy",
    "au_legislation_api",
    "my_legislation_api",
    "au_act_catalogue",
    "au_semantic_crosswalk",
    "pdpc_guidance",
    "my_pdp_guidance",
    "oaic_guidance",
    "PORTAL_CONNECTORS",
    "connector_for",
    "sg_resolve_fulltext",
    "au_resolve_fulltext",
    "STRATEGIES",
    "CONCEPT_STRATEGIES",
    "RESOLVERS",
    "strategy_for",
    "concept_strategy_for",
    "resolver_for",
]
