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
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from urllib.parse import quote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from lexora.collect.discovery import (
    TAG_KNOWN,
    TAG_NEW,
    DiscoveryResult,
    _fuzzy_known,
)
from lexora.models.source import PortalSpec, SourceType

_LOG = logging.getLogger(__name__)

Strategy = Callable[..., list[DiscoveryResult]]

_AU_API = "https://api.prod.legislation.gov.au/v1/titles"
_AU_DOC = "https://www.legislation.gov.au/{id}/{point}"
_AU_PAGE = 100  # OData max $top (200+ -> HTTP 400)

_MY_API = "https://lom.agc.gov.my/fess-proxy.php"


def _odata_escape(value: str) -> str:
    """Escape a string literal for an OData filter (single quote -> doubled)."""
    return value.replace("'", "''")


def _get_with_retry(
    client: httpx.Client, url: str, *, retries: int = 3, backoff: float = 1.0
) -> httpx.Response:
    """GET ``url`` retrying on transport errors AND 5xx, with exponential backoff.

    MY's Fess proxy (lom.agc.gov.my) intermittently drops the TLS connection
    mid-handshake/read, and the AU OData catalogue paging (~48 sequential GETs) is
    similarly exposed to a transient blip. ``httpx.TransportError`` covers the
    connect/read/SSL family.

    5xx is retried too, and this version of the docstring exists because the previous
    one asserted the opposite -- "an HTTP 4xx/5xx is a real response and returned
    as-is". For this endpoint that is false. Measured 2026-08-01: **fourteen identical
    requests to fess-proxy.php returned seven 500s and seven 200s.** The failure is a
    coin flip, and `discover_my_fess` turns a non-200 into an empty result, so half of
    Malaysia's forty discovery queries were silently finding nothing -- which is most
    of why Malaysia's working set was thirteen instruments and never contained the
    Personal Data Protection Act. Four attempts take that 50% down to about 6%.

    4xx is still returned as-is: a 404 is an answer, and retrying it only costs time.
    A replay MISS is dressed as a 504 by the record/replay layer; it carries
    ``x-lexora-cache: miss`` and is taken at its word rather than re-asked three times.
    """
    return _request_with_retry(client, url, retries=retries, backoff=backoff)


_MY_CLIENT: httpx.Client | None = None
_MY_CLIENT_LOCK = threading.Lock()


def _my_client(timeout: float) -> httpx.Client:
    """One pooled client for the whole Malaysian sweep, because the 500s are per-connection.

    Measured 2026-08-01 against `fess-proxy.php`, the same POST repeated:

        a fresh httpx.Client each time   ->  6 x 500, 4 x 200
        one client reused               -> 10 x 200

    The failure is the first request on a new connection, not the query and not the
    server's health. Every discovery query used to build its own client, so Malaysia
    paid that coin flip forty times a run. Retries only paper over it.

    Built lazily and never at import: `http_cache.install()` patches
    `httpx.Client.__init__`, so a client constructed at import time would be created
    before the patch and quietly bypass the record/replay layer.
    """
    global _MY_CLIENT
    with _MY_CLIENT_LOCK:
        if _MY_CLIENT is None:
            _MY_CLIENT = httpx.Client(
                follow_redirects=True, timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json",
                         "Referer": "https://lom.agc.gov.my/",
                         "X-Requested-With": "XMLHttpRequest"},
            )
        return _MY_CLIENT


def _request_with_retry(
    client: httpx.Client, url: str, *, data: dict | None = None,
    retries: int = 3, backoff: float = 1.0,
) -> httpx.Response:
    """``_get_with_retry`` with an optional form body, which makes it a POST.

    Malaysia's proxy needs the POST; see ``my_legislation_api``.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = (client.post(url, data=data) if data is not None
                        else client.get(url))
            if (response.status_code < 500
                    or response.headers.get("x-lexora-cache") == "miss"
                    or attempt == retries):
                return response
        except httpx.TransportError as exc:
            last = exc
            if attempt == retries:
                break
        time.sleep(backoff * (attempt + 1))
    raise last or httpx.TransportError(f"failed to fetch {url}")


def _first_href(snippet: str | None, base: str = _MY_API) -> str | None:
    """Extract the first href URL from an HTML anchor snippet (MY download cell).

    Resolved against the portal, because the cell holds a PAGE-RELATIVE link:
    `downloadPDF.php?cs=1&token=...`. Handed to the fetcher unjoined it raised
    UnsupportedProtocol, and since one document's exception takes the whole economy
    down, three such links cost the entire Malaysian run. Joined, they are real Acts
    -- the first one is an 18 MB `application/pdf`.
    """
    if not snippet:
        return None
    m = re.search(r'href="([^"]+)"', snippet)
    return urljoin(base, m.group(1)) if m else None


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
    ids: dict[str, str] | None, title: str | None = None,
) -> tuple[str | None, str | None]:
    """Tag + matched-name, preferring an identity hit (e.g. Act number / doc id)
    over fuzzy name matching — so filename-titled records still resolve KNOWN.

    Pass ``title`` to have identity re-scored year-strictly. The callers' ``fuzzy`` is a
    RELEVANCE score and is deliberately not year-aware; identity is a different question,
    and ``token_set_ratio`` answers it wrongly for a longer title that contains a shorter
    one (see ``discovery._years_conflict``)."""
    id_name = (ids or {}).get(ident)
    if id_name:
        return TAG_KNOWN, id_name
    if title is not None and known:
        fuzzy, matched = _fuzzy_known(title, known, year_strict=True)
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
        tag, matched_name = _resolve_tag(fuzzy, matched, v["id"], known, known_instrument_ids, title=name)
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


_FRL_ID_RE = re.compile(r"/([CF]\d{4}[A-Z]\d{5})\b")


def frl_id_from_url(url: str) -> str:
    """The Federal Register title id (e.g. ``C2004A03712``) embedded in an AU
    legislation URL, or ``""`` when the URL is not an FRL document link."""
    m = _FRL_ID_RE.search(url or "")
    return m.group(1) if m else ""


def au_amendment_acts(
    title_id: str,
    *,
    limit: int = 40,
    timeout: float = 30.0,
    source_type: SourceType = SourceType.primary,
    known_instruments: list[str] | None = None,
    known_instrument_ids: dict[str, str] | None = None,
    client: httpx.Client | None = None,
) -> list[DiscoveryResult]:
    """Every Act that AMENDS the principal ``title_id``, from the FRL versions graph.

    The title-derived ``amendment_search_queries`` guess (``"<principal core>
    Amendment"``) cannot surface Australia's THEME-named omnibus amendment Acts —
    "Surveillance Legislation Amendment (Identify and Disrupt) Act 2021" amends the
    Crimes Act and the Surveillance Devices Act but shares no title core with either,
    so a name search never finds it. The authoritative source is each principal's
    own compilation history: ``GET /titles('<id>')?$expand=versions`` returns every
    version, and each version's ``reasons`` records the affecting Act under
    ``affectedByTitle`` (id + name + affected provisions + year). This reverse-lookup
    returns those amending Acts as canonical ``/latest`` document candidates, newest
    first (an amendment that post-dates the principal's compilation is the one that
    can make it stale), de-duplicated by title id.

    GOTCHA: the populated field is ``affectedByTitle`` — ``amendedByTitle`` is always
    null. Only ``affect == "Amend"`` reasons are kept (Repeal/Commence are not
    amendments). The ``Affect`` / ``_AffectsSearch`` EntitySets are not directly
    queryable (404); ``$expand=versions`` with inline ``reasons`` is the only route."""
    if not title_id:
        return []
    url = (
        f"{_AU_API}('{title_id}')?%24expand=versions"
        f"&%24select=id"
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
        versions = resp.json().get("versions", [])
    except Exception:  # noqa: BLE001 — a failed lookup never breaks the run
        return []
    finally:
        if owns:
            client.close()

    # title id -> (name, year, number) for distinct amending Acts (keep first = newest,
    # since versions come oldest-first we overwrite, then sort by year desc below).
    by_id: dict[str, tuple[str, int | None, int | None]] = {}
    for v in versions:
        for reason in v.get("reasons") or []:
            if (reason.get("affect") or "") != "Amend":
                continue
            amb = reason.get("affectedByTitle")
            if not amb or not amb.get("titleId"):
                continue
            by_id[amb["titleId"]] = (
                amb.get("name") or "", amb.get("year"), amb.get("number")
            )

    results: list[DiscoveryResult] = []
    known = known_instruments or []
    for tid, (name, year, number) in by_id.items():
        law_number = f"No. {number} of {year}" if number and year else tid
        # Tagged like every other route, against the gold inventory. Hardcoding NEW here
        # claimed a discovery for an instrument already on the official list.
        tag, matched = _resolve_tag(0.0, None, tid, known, known_instrument_ids, title=name)
        results.append(
            DiscoveryResult(
                url=_AU_DOC.format(id=tid, point="latest"),
                title=name,
                source_type=source_type,
                score=1.0,  # authoritative register relationship
                via="api",
                is_pdf_link=False,
                discovery_tag=tag or TAG_NEW,
                matched_instrument=matched,
                n_variants=1,
                law_number=law_number,
            )
        )
    # Newest first: the amendments most likely to post-date a compilation come first,
    # so a per-principal `limit` keeps the staleness-relevant ones.
    results.sort(key=lambda r: _year_of_title(r.title), reverse=True)
    return results[:limit]


def _year_of_title(title: str) -> int:
    """Trailing 4-digit year in a title ("... Act 2021" -> 2021), else 0 (sorts last)."""
    m = re.search(r"\b(19|20)\d{2}\b", title or "")
    return int(m.group(0)) if m else 0


def _act_title_core(act_title: str) -> str:
    """The naming stem of an Act title — everything before " Act [year]"
    ("Telecommunications Act 1997" -> "Telecommunications"). Empty when the title
    has no "Act" stem (so the caller skips the lookup)."""
    stem = re.split(r"\bAct\b", act_title or "", maxsplit=1)[0]
    return re.sub(r"\s+", " ", stem).strip()


def _authorised_by(
    reg_id: str, act_id: str, client: httpx.Client, timeout: float
) -> bool:
    """True iff legislative instrument ``reg_id`` is authorised (made under) the Act
    ``act_id``, per the Federal Register's authorisation edge. The instrument's
    ``authorisedBy`` is a collection of ``Affect`` whose ``affectingTitleId`` is the
    enabling Act — this turns the name-pattern guess into a confirmed parent/child
    link (drops a same-named instrument made under a different Act)."""
    url = f"{_AU_API}('{reg_id}')?%24expand=authorisedBy&%24select=id"
    try:
        resp = _get_with_retry(client, url, retries=1)
        if resp.status_code != 200:
            return False
        for affect in resp.json().get("authorisedBy", []) or []:
            if affect.get("affectingTitleId") == act_id:
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


def au_child_regulations(
    act_id: str,
    act_title: str,
    *,
    limit: int = 20,
    timeout: float = 30.0,
    source_type: SourceType = SourceType.primary,
    known_instruments: list[str] | None = None,
    known_instrument_ids: dict[str, str] | None = None,
    client: httpx.Client | None = None,
) -> list[DiscoveryResult]:
    """The principal REGULATIONS made under the Act ``act_id`` — delegated legislation
    that the principal-Act brute enumeration skips (it admits only ``collection ==
    'Act'``, dropping the ``LegislativeInstrument`` collection where regulations live).

    AU exposes no Act -> instruments navigation, and the OData ``Affect``/
    ``_AffectsSearch`` sets are not directly queryable. The reliable channel is the
    naming convention: a principal regulation is named "<Act stem> Regulations <year>"
    (Telecommunications Act 1997 -> Telecommunications Regulations 2021). So query the
    register by that name stem, keep in-force PRINCIPAL items in the Regulations
    subcollection, then CONFIRM each via its ``authorisedBy`` edge that it is actually
    made under this Act (precision guard against a same-stemmed instrument under a
    different Act). De-duplicated by id, newest first."""
    core = _act_title_core(act_title)
    if not act_id or len(core) < 3:
        return []
    flt = f"contains(tolower(name),'{_odata_escape(core.lower())} regulations')"
    url = (
        f"{_AU_API}?%24filter={quote(flt, safe='(),')}"
        f"&%24top=50&%24select=id,name,collection,subCollection,isPrincipal,isInForce,number,year"
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
        results: list[DiscoveryResult] = []
        seen: set[str] = set()
        for v in values:
            if (
                v.get("subCollection") != "Regulations"
                or not v.get("isPrincipal")
                or not v.get("isInForce")
            ):
                continue
            rid = v.get("id")
            if not rid or rid in seen:
                continue
            if not _authorised_by(rid, act_id, client, timeout):
                continue
            seen.add(rid)
            num, yr = v.get("number"), v.get("year")
            from lexora.classify.lifecycle import detect_status

            # Tagged against the gold inventory like every other route. This one used to
            # hardcode NEW, so it claimed a discovery for regulations already on the
            # official list -- including Telecommunications Regulations 2021, which is both
            # in the AU profile's known_instruments AND the example in this function's own
            # motivation. NEW is 20 of the 40 accuracy points; a false NEW is not a rounding
            # error, it is a claim we cannot support.
            name = v.get("name", "")
            known = known_instruments or []
            tag, matched = _resolve_tag(0.0, None, rid, known, known_instrument_ids, title=name)
            results.append(
                DiscoveryResult(
                    url=_AU_DOC.format(id=rid, point="latest"),
                    title=name,
                    source_type=source_type,
                    score=1.0,  # authoritative authorisation edge
                    via="api",
                    is_pdf_link=False,
                    discovery_tag=tag or TAG_NEW,
                    matched_instrument=matched,
                    n_variants=1,
                    law_number=f"No. {num} of {yr}" if num and yr else rid,
                    status=detect_status(in_force=True, portal_status="").value,
                )
            )
    except Exception:  # noqa: BLE001 — a failed lookup never breaks the run
        return []
    finally:
        if owns:
            client.close()
    results.sort(key=lambda r: _year_of_title(r.title), reverse=True)
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
    # Whether paging ran to its natural end. Only a COMPLETE catalogue may be cached: the
    # loop below also stops on a non-200, and that stop is indistinguishable from the
    # normal one once you are looking at the list. A single transient 503 on page 30 of 48
    # therefore used to write ~3,000 of the ~4,700 in-force Acts into a 24-hour cache, and
    # every run in that window enumerated the truncated list and reported success. A short
    # catalogue is not an error anyone would notice; it is just fewer laws found.
    complete = False
    try:
        for page in range(max_pages):
            url = (
                f"{_AU_API}?%24filter={flt}&%24top={_AU_PAGE}&%24skip={page * _AU_PAGE}"
                f"&%24select=id,name,isPrincipal"
            )
            resp = _get_with_retry(client, url)
            if resp.status_code != 200:
                _LOG.warning(
                    "AU catalogue paging stopped at page %d on HTTP %d — %d entries so far, "
                    "not cached", page, resp.status_code, len(catalogue),
                )
                break
            values = resp.json().get("value", [])
            if not values:
                complete = True
                break
            catalogue.extend(
                {"id": v["id"], "name": v.get("name", ""), "isPrincipal": bool(v.get("isPrincipal"))}
                for v in values if v.get("id") and v.get("name")
            )
            if len(values) < _AU_PAGE:
                complete = True
                break
        else:
            _LOG.warning(
                "AU catalogue hit max_pages=%d (%d entries) — the register may have grown "
                "past what this fetches; not cached", max_pages, len(catalogue),
            )
    except Exception:
        return catalogue
    finally:
        if owns:
            client.close()

    if catalogue and complete:
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

    # POST, not GET, and the difference is the whole Malaysian corpus.
    #
    # On GET the proxy accepts the request and IGNORES the query: `numFound` comes back
    # as 10000 -- the entire corpus -- and the first rows are simply the lowest act
    # numbers, so every one of the forty discovery queries returned the same handful
    # (MINISTERIAL FUNCTIONS ACT 1969, FINANCE COMPANIES ACT 1969, ...). It answers 200,
    # so nothing ever looked wrong. Measured 2026-08-01, same term:
    #
    #     GET  q="personal data"  -> numFound 10000, first: MINISTERIAL FUNCTIONS ACT 1969
    #     POST q="personal data"  -> numFound   821, first: PERSONAL DATA PROTECTION ACT 2010
    #
    # POST also honours `rows` (GET capped the page at 10). Every flagship the gold
    # inventory names -- Computer Crimes Act 1997, Security Offences (Special Measures)
    # Act 2012, Communications and Multimedia Act 1998 -- is the top hit for its own
    # concept query over POST, and was unreachable over GET.
    params = {
        "q": query, "fq": "", "start": "0", "rows": "25",
        "sort": "", "lookup": "all", "kategori": "all", "draw": "1",
    }

    owns = False  # the pooled client outlives this call by design; see _my_client
    client = client or _my_client(timeout)
    try:
        resp = _request_with_retry(client, _MY_API, data=params)
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
        fuzzy, matched = _fuzzy_known(title, known, year_strict=True) if known else (0.0, None)
        # Solr relevance + lexical match + principal preference, normalized to [0,1].
        score = min(1.0, 0.4 * g["solr_rel"] + 0.4 * max(q_sim, fuzzy) + 0.2 * g["principal"])
        if score < min_score:
            continue
        # Act-number identity tags KNOWN even when the title is just a filename.
        tag, matched_name = _resolve_tag(fuzzy, matched, act_id, known, known_instrument_ids, title=title)
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
        fuzzy, matched = _fuzzy_known(title, known, year_strict=True) if known else (0.0, None)
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
# The DPIA guide is a standalone PDF under /Other-Guides — not on either hub the
# harvest crawls — so the sibling-link walk never reaches it. It is a single stable
# PDF; add it explicitly. resolve_fulltext serves the PDF directly for mapping.
_PDPC_EXTRA = (
    ("https://www.pdpc.gov.sg/-/media/Files/PDPC/PDF-Files/Other-Guides/DPIA/"
     "Guide-to-Data-Protection-Impact-Assessments-14-Sep-2021.pdf",
     "Guide to Data Protection Impact Assessments"),
)
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
    # Guides outside the crawled hubs (the DPIA guide PDF) — add explicitly.
    for url, title in _PDPC_EXTRA:
        if url not in agg:
            fuzzy, matched = _fuzzy_known(title, known, year_strict=True) if known else (0.0, None)
            agg[url] = DiscoveryResult(
                url=url, title=title, source_type=SourceType.secondary,
                score=1.0, via="http", is_pdf_link=url.lower().endswith(".pdf"),
                discovery_tag=TAG_KNOWN if fuzzy >= 0.80 else TAG_NEW,
                matched_instrument=matched if fuzzy >= 0.80 else None,
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
# The PDP Standard 2015 sits OUTSIDE the code-of-practice sidebar (a separate
# instrument), so the sibling-link harvest never reaches it. It is a single stable
# page; add it explicitly. resolve_fulltext's generic .pdf-harvest pulls its
# LatestStandard.pdf for mapping.
_MY_PDP_EXTRA = (
    ("https://www.pdp.gov.my/ppdpv1/en/akta/personal-data-protection-standard-2015/",
     "Personal Data Protection Standard 2015"),
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
        # Soft-law instruments outside the code-of-practice sidebar (the Standard).
        for url, title in _MY_PDP_EXTRA:
            if url not in agg:
                fuzzy, matched = _fuzzy_known(title, known, year_strict=True) if known else (0.0, None)
                agg[url] = DiscoveryResult(
                    url=url, title=title, source_type=SourceType.secondary,
                    score=1.0, via="http", is_pdf_link=False,
                    discovery_tag=TAG_KNOWN if fuzzy >= 0.80 else TAG_NEW,
                    matched_instrument=matched if fuzzy >= 0.80 else None,
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


# --- SG IMDA telecom licence conditions (static PDFs, no crawlable hub) ---
# The telecom soft-law referenced under P7-I3 / P7-I5 lives as standalone official
# PDFs with no list page the harvest can walk, so the gold instruments are seeded
# explicitly. Each is a stable PDF that resolve_fulltext serves directly for mapping.
_IMDA_INSTRUMENTS = (
    ("https://www.imda.gov.sg/~/media/imda/files/inner/pcdg/consultations/"
     "20040921_propoiptelephony/tcsforiptelephony240605.pdf",
     "Specific Terms and Conditions for IP Telephony Services"),
    ("https://www.imda.gov.sg/regulations-and-licences/licensing/"
     "list-of-telecommunication-and-postal-service-licensees/-/media/Imda/Files/"
     "Regulation-Licensing-and-Consultations/Licensing/Licensees/FBO/SingTelLtd.pdf",
     "Licence to Provide Facilities-Based Operations (Singapore Telecommunications Limited)"),
)


def imda_guidance(
    portal: PortalSpec,
    indicators: list,
    *,
    browser_session=None,
    limit: int = 40,
    timeout: float = 45.0,
    known_instruments: list[str] | None = None,
    known_instrument_ids: dict[str, str] | None = None,
) -> list[DiscoveryResult]:
    """Seed the IMDA (Singapore) telecom licence-condition instruments. IMDA exposes
    no crawlable guidance list, so the gold soft-law (IP Telephony T&Cs, the SingTel
    facilities-based-operations licence) is added explicitly as secondary PDFs."""
    known = known_instruments or []
    out: list[DiscoveryResult] = []
    for url, title in _IMDA_INSTRUMENTS:
        fuzzy, matched = _fuzzy_known(title, known, year_strict=True) if known else (0.0, None)
        out.append(DiscoveryResult(
            url=url, title=title, source_type=SourceType.secondary,
            score=1.0, via="http", is_pdf_link=True,
            discovery_tag=TAG_KNOWN if fuzzy >= 0.80 else TAG_NEW,
            matched_instrument=matched if fuzzy >= 0.80 else None,
        ))
    return out[:limit]


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
    "imda.gov.sg": imda_guidance,
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
# The FRL EPUB carries the full operative prose as one static XHTML file at a
# dated path. The ``/{point}/text`` SPA shell only renders the table of contents,
# but it embeds every section's deep-link into that EPUB, so the EPUB URL can be
# harvested from the shell HTML with a plain GET — no browser, no PDF.
_AU_EPUB_PATH = re.compile(
    r"https://www\.legislation\.gov\.au/[^\"'\s]+/epub/OEBPS/document_1/document_1\.html"
)


def sg_resolve_fulltext(result, *, timeout: float = 30.0) -> str | None:
    """SG SSO serves the whole-instrument PDF at ``<url>?ViewType=Pdf`` (fetchable
    with a browser UA even though the HTML landing 403s bots). Covers both principal
    Acts (``/Act/``) and amendment instruments in the Acts Supplement
    (``/Acts-Supp/``, surfaced by the SG amendment reverse-lookup)."""
    base = result.url.split("?")[0].rstrip("/")
    return f"{base}?ViewType=Pdf" if ("/Act/" in base or "/Acts-Supp/" in base) else None


# A consolidated SSO Act annotates each amending Act inline as "Act N of YYYY" (in
# the section endnotes), the SG analogue of AU FRL's structured affects graph.
_SG_AMEND_CITE = re.compile(r"\bAct\s+(\d+)\s+of\s+((?:19|20)\d{2})\b")


def sg_amendment_acts(principal_text: str, *, limit: int = 6) -> list[DiscoveryResult]:
    """SG SSO reverse-lookup: SSO has no FRL-style affects API, but a consolidated
    Act's text annotates each amending Act inline as "Act N of YYYY". Construct each
    one's Acts Supplement URL (``/Acts-Supp/{N}-{YYYY}/`` — the form the official
    inventory itself uses) so it can be fetched and classified. Newest first; the
    caller keeps only those that actually classify as AMENDMENT_DELTA, which drops the
    principal's own "Act N of YYYY" (its original enactment) and any cross-reference."""
    seen: set[tuple[str, str]] = set()
    cites: list[tuple[int, str, str]] = []
    for m in _SG_AMEND_CITE.finditer(principal_text or ""):
        n, y = m.group(1), m.group(2)
        if (n, y) in seen:
            continue
        seen.add((n, y))
        cites.append((int(y), n, y))
    cites.sort(reverse=True)  # newest amendments first
    out: list[DiscoveryResult] = []
    for _, n, y in cites[:limit]:
        out.append(DiscoveryResult(
            url=f"https://sso.agc.gov.sg/Acts-Supp/{n}-{y}/",
            title=f"Act {n} of {y}", source_type=SourceType.primary,
            score=1.0, via="sso-history", is_pdf_link=False,
        ))
    return out


def au_resolve_fulltext(result, *, timeout: float = 30.0) -> str | None:
    """Resolve an AU FRL instrument page to a fetchable full-text source.

    Primary route is the **EPUB HTML** (``/{id}/{point}/{date}/{date}/text/
    original/epub/OEBPS/document_1/document_1.html``): a single static XHTML file
    holding the whole compilation's operative prose. Its URL is embedded in the
    ``/{point}/text`` SPA shell (each TOC entry deep-links into it), so a plain
    GET on the shell yields it — NO browser. This is cheaper and richer than the
    PDF path: the structure parser extracts hundreds-to-thousands of verbatim
    clauses from the EPUB (incl. Schedule-1 Australian Privacy Principles), where
    the shell alone yields zero (it is just a table of contents). The date in the
    EPUB path is the latest compilation, so APP-era amendments are included.

    Falls back to the legacy browser-rendered downloads page → dated PDF
    (``/text/original/pdf``) when the shell has no EPUB link (e.g. image-only
    as-made instruments).
    """
    from urllib.parse import urljoin

    shell = result.url.rstrip("/")
    if not shell.endswith("/text"):
        shell += "/text"
    try:
        with httpx.Client(
            follow_redirects=True, timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as client:
            html = client.get(shell).text
        m = _AU_EPUB_PATH.search(html)
        if m:
            return m.group(0)
    except Exception:
        pass

    # Fallback: legacy browser-rendered downloads page → dated PDF.
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
    "imda_guidance",
    "sg_amendment_acts",
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
