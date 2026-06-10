"""Discovery layer (Zone 1, stage 1): search -> candidate instrument URLs.

Slice 1 fetched a known URL. The mandatory crawl, however, must *discover* which
instrument to fetch. This module answers a single question:

    given a portal and a query, which URLs on that portal are likely to be the
    primary legal instrument we want?

Discovery is uniform across portals: fetch a results page (over HTTP, or via a
headless browser for JS / anti-bot portals such as Singapore SSO which returns
403 to plain clients), then harvest and rank the anchor links on that page with
:func:`harvest_candidates` — a pure function that needs no network and is fully
unit-tested. The ranking favours links that (a) overlap the query terms, (b)
point at a legal instrument (``.pdf``, paths containing ``act`` / ``legislation``
/ a year), and drops site chrome (login, contact, social, sitemap).

The fetched-vs-rendered split is driven by ``PortalSpec.fetch_method`` /
``search_url_template`` so the same code path serves MY (server-rendered HTML),
AU (SPA) and SG (403 -> browser).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import quote, urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from lexora.collect.crawler import DEFAULT_UA
from lexora.models.source import FetchMethod, PortalSpec, SourceType

# Tokens that mark a link as legal-instrument-like (boost) or site-chrome (drop).
_INSTRUMENT_MARKERS = (
    "act", "legislation", "statute", "law", "/cap", "bill", "regulation",
    "ordinance", "gazette", "decree", "code",
)
_CHROME_MARKERS = (
    "login", "signin", "sign-in", "register", "logout", "contact", "about",
    "sitemap", "privacy-policy", "terms", "feedback", "subscribe", "rss",
    "facebook.com", "twitter.com", "x.com", "linkedin.com", "youtube.com",
    "instagram.com", "javascript:", "mailto:", "tel:",
    # Portal navigation / empty-results-page chrome (esp. SG SSO): a search that
    # matches nothing renders the default browse shell, whose links are these —
    # not legal instruments. Dropping them stops an empty query polluting results.
    "/browse/", "/help/", "/search/advanced", "/search/content",
    "frequently-accessed", "acts-supp", "act-rev", "/sso-guide",
    "my collections", "revised editions", "acts supplement",
)
_RESULT_CONTAINER = re.compile(r"result|item|card|search|title|listing|row", re.I)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Fuzzy title-vs-known-name thresholds for the Discovery Tag.
_KNOWN_THRESHOLD = 0.80   # >= -> KNOWN (matches a listed instrument)
_NEW_CEILING = 0.55       # instrument-like + relevant but < this -> NEW candidate

# Tags (kept as plain strings; the citation layer maps them to DiscoveryTag).
TAG_KNOWN = "KNOWN"
TAG_NEW = "NEW"


@dataclass
class DiscoveryResult:
    """One candidate instrument URL surfaced from a portal.

    `score` is normalized to [0, 1]. `n_variants` is how many raw links collapsed
    into this instrument (a relevance signal of its own — an Act that matched many
    provisions surfaces many deep-links). `discovery_tag` is KNOWN / NEW / None and
    `matched_instrument` names the known instrument a KNOWN hit matched.
    """

    url: str
    title: str
    source_type: SourceType
    score: float
    via: str  # "http" | "browser" | "api"
    is_pdf_link: bool
    discovery_tag: str | None = None
    matched_instrument: str | None = None
    n_variants: int = 1
    # A direct full-text source (usually a PDF) for this instrument, when the
    # portal exposes one — the entry point for two-stage discovery (stage 2 feeds
    # the proven PDF pipeline). None means "fetch `url` and locate it".
    fulltext_url: str | None = None
    # Submission ids of the indicators whose concept query surfaced this
    # instrument (set by `discover_for_indicators`). It ties an instrument back to
    # the indicators it is relevant to, so the mapper only scores a sectoral law
    # against the indicator that found it — not blindly against all of them.
    indicator_hits: list[str] = field(default_factory=list)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text)}


def _normalize_url(url: str) -> str:
    """Collapse duplicate slashes in the path (some SPAs emit `//Act/...`)."""
    parts = urlparse(url)
    path = re.sub(r"/{2,}", "/", parts.path)
    return urlunparse(parts._replace(path=path))


def _canonical_key(url: str) -> str:
    """Instrument-level identity: scheme+host+path, ignoring query and fragment.

    Collapses the many deep-links a portal emits for one Act (e.g. SG SSO
    `/Act/PDPA2012?ProvIds=P12-` vs `?ProvIds=pr5-`) onto a single instrument.
    """
    p = urlparse(url)
    path = re.sub(r"/{2,}", "/", p.path).rstrip("/")
    return f"{p.scheme}://{p.netloc}{path}".lower()


_PROVISION_MARKER = re.compile(r"provids|prov=|/pr\d|#pr", re.I)


def _pick_representative(urls: set[str]) -> str:
    """Among variant URLs for one instrument, prefer the Act-level link: penalize
    provision deep-links (e.g. `?ProvIds=...`) first, then fewest params, then
    shortest."""
    return min(
        urls,
        key=lambda u: (bool(_PROVISION_MARKER.search(u)), u.count("?") + u.count("&"), len(u)),
    )


def _is_chrome(href_l: str, text_l: str) -> bool:
    return any(m in href_l or m in text_l for m in _CHROME_MARKERS)


def _context_text(a) -> str:
    """Text of the link's nearest result container (or preceding heading).

    Recovers the instrument title when a portal renders it outside the anchor
    (e.g. AU result cards put the name in a heading and the `<a>` says 'View')."""
    container = a.find_parent(["li", "tr", "article"])
    if container is None:
        container = a.find_parent("div", class_=_RESULT_CONTAINER)
    if container is not None:
        return " ".join(container.get_text(" ", strip=True).split())[:240]
    heading = a.find_previous(["h1", "h2", "h3", "h4"])
    if heading is not None:
        return " ".join(heading.get_text(" ", strip=True).split())[:120]
    return ""


def _fuzzy_known(text: str, known: list[str]) -> tuple[float, str | None]:
    """Best fuzzy match of `text` against the known instrument names, in [0,1]."""
    best, name = 0.0, None
    tl = text.lower()
    for inst in known:
        s = fuzz.token_set_ratio(inst.lower(), tl) / 100.0
        if s > best:
            best, name = s, inst
    return best, name


def _score_link(
    query_tokens: set[str],
    known: list[str],
    href: str,
    anchor_text: str,
    context_text: str,
) -> tuple[float, bool, str | None, str | None, str]:
    """Score one link in [0, 1]. Returns (score, is_pdf, tag, matched, title)."""
    href_l = href.lower()
    is_pdf = href_l.endswith(".pdf") or ".pdf?" in href_l
    title = anchor_text if len(anchor_text) >= 12 else (context_text or anchor_text)

    haystack = _tokens(anchor_text) | _tokens(context_text) | _tokens(urlparse(href).path)
    q_overlap = (len(query_tokens & haystack) / len(query_tokens)) if query_tokens else 0.0

    fuzzy, matched = _fuzzy_known(title or context_text, known) if known else (0.0, None)
    relevance = max(q_overlap, fuzzy)

    title_l = (title or "").lower()
    instrument_like = any(m in href_l or m in title_l for m in _INSTRUMENT_MARKERS)
    bonus = 0.0
    if is_pdf:
        bonus += 0.15
    if instrument_like:
        bonus += 0.10
    if _YEAR.search(href_l) or _YEAR.search(title_l):
        bonus += 0.05
    score = min(1.0, 0.75 * relevance + bonus)

    tag: str | None = None
    if known:
        if fuzzy >= _KNOWN_THRESHOLD:
            tag = TAG_KNOWN
        elif instrument_like and q_overlap >= 0.5 and fuzzy < _NEW_CEILING:
            tag = TAG_NEW
    if tag != TAG_KNOWN:
        matched = None  # matched_instrument is only meaningful for a KNOWN hit
    return score, is_pdf, tag, matched, title


def harvest_candidates(
    html: str | bytes,
    *,
    base_url: str,
    query: str | None,
    source_type: SourceType,
    via: str = "http",
    limit: int = 10,
    min_score: float = 0.1,
    known_instruments: list[str] | None = None,
) -> list[DiscoveryResult]:
    """Harvest and rank candidate instrument links from a results/landing page.

    Pure function (no network). For each anchor it scores relevance from the
    anchor text, its surrounding result-container text and the URL path; collapses
    the many deep-links of one instrument onto a single canonical entry; and tags
    each instrument KNOWN / NEW against `known_instruments`. Returns the top
    instruments ranked by normalized score, then by how many variants matched.
    """
    soup = BeautifulSoup(html, "lxml")
    query_tokens = _tokens(query) if query else set()
    known = known_instruments or []

    agg: dict[str, dict] = {}
    for a in soup.find_all("a", href=True):
        raw = a["href"].strip()
        if not raw or raw.startswith("#"):
            continue
        anchor_text = " ".join(a.get_text(" ", strip=True).split())
        if _is_chrome(raw.lower(), anchor_text.lower()):
            continue
        url = _normalize_url(urljoin(base_url, raw))
        if urlparse(url).scheme not in ("http", "https"):
            continue

        context_text = _context_text(a)
        score, is_pdf, tag, matched, title = _score_link(
            query_tokens, known, url, anchor_text, context_text
        )
        if score < min_score:
            continue

        key = _canonical_key(url)
        rec = agg.get(key)
        if rec is None:
            agg[key] = {
                "score": score, "urls": {url}, "title": title, "is_pdf": is_pdf,
                "tag": tag, "matched": matched,
            }
        else:
            rec["urls"].add(url)
            rec["score"] = max(rec["score"], score)
            rec["is_pdf"] = rec["is_pdf"] or is_pdf
            if tag == TAG_KNOWN:
                rec["tag"] = TAG_KNOWN
                rec["matched"] = rec["matched"] or matched
            elif tag == TAG_NEW and rec["tag"] is None:
                rec["tag"] = TAG_NEW
            if title and len(title) > len(rec["title"]):
                rec["title"] = title

    results = [
        DiscoveryResult(
            url=_pick_representative(rec["urls"]),
            title=rec["title"],
            source_type=source_type,
            score=rec["score"],
            via=via,
            is_pdf_link=rec["is_pdf"],
            discovery_tag=rec["tag"],
            matched_instrument=rec["matched"],
            n_variants=len(rec["urls"]),
        )
        for rec in agg.values()
    ]
    results.sort(key=lambda r: (r.score, r.n_variants), reverse=True)
    return results[:limit]


def _search_url(portal: PortalSpec, query: str | None) -> str:
    """Build the page URL to harvest: a templated search URL when the portal
    provides one, otherwise the portal landing page."""
    if portal.search_url_template and query:
        # `quote` (space -> %20) works in both a path segment (AU /search/text/...)
        # and a query string; `quote_plus` (space -> +) is only valid in the latter.
        return portal.search_url_template.replace("{query}", quote(query, safe=""))
    return str(portal.url)


def _fetch_page(
    url: str,
    *,
    force_browser: bool,
    client: httpx.Client | None,
    timeout: float,
    user_agent: str,
    browser_session=None,
) -> tuple[str | bytes, str]:
    """Fetch a page's HTML, escalating to a headless browser on anti-bot 403/429
    (or when ``force_browser``). Returns ``(html, via)`` where via is http|browser.

    ``browser_session`` (a :class:`browser.BrowserSession`) renders through one
    persistent Chromium instead of relaunching per call — used by
    :func:`discover_for_indicators` so a burst of SG SSO queries stays stable."""
    use_browser = force_browser
    html: str | bytes = ""
    if not use_browser:
        owns = client is None
        client = client or httpx.Client(
            follow_redirects=True, timeout=timeout, headers={"User-Agent": user_agent}
        )
        try:
            resp = client.get(url)
            if resp.status_code in (403, 429):
                use_browser = True  # anti-bot — escalate to a real browser
            else:
                html = resp.content
        finally:
            if owns:
                client.close()

    if use_browser:
        # Render with the browser's own realistic UA — forwarding the polite HTTP
        # bot UA here just re-triggers the anti-bot 403 we are escalating past.
        if browser_session is not None:
            html = browser_session.render(url, timeout=timeout).html
        else:
            from lexora.collect.browser import render

            html = render(url, timeout=timeout).html
        return html, "browser"
    return html, "http"


def resolve_fulltext(
    result: DiscoveryResult,
    *,
    query: str | None = None,
    client: httpx.Client | None = None,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_UA,
    force_browser: bool = False,
) -> str | None:
    """Two-stage discovery, stage 2: turn an instrument page into a fetchable
    full-text source.

    Returns, in priority order: an explicit ``fulltext_url`` the strategy already
    captured (MY); the result URL itself if it is already a PDF; otherwise the
    best ``.pdf`` link found on the instrument page (SG SSO 'Download', AU
    'downloads'). ``None`` if no full-text source can be located — the caller can
    still fall back to extracting the instrument page's HTML.
    """
    if result.fulltext_url:
        return result.fulltext_url
    if result.is_pdf_link:
        return result.url

    # A per-portal resolver knows the portal's PDF convention (SG ?ViewType=Pdf,
    # AU dated /text/original/pdf) when it isn't a plain .pdf link on the page.
    from lexora.collect.strategies import resolver_for

    resolver = resolver_for(result.url)
    if resolver is not None:
        resolved = resolver(result, timeout=timeout)
        if resolved:
            return resolved

    html, _ = _fetch_page(
        result.url, force_browser=force_browser, client=client,
        timeout=timeout, user_agent=user_agent,
    )
    if not html:
        return None
    pdfs = [
        c for c in harvest_candidates(
            html, base_url=result.url, query=query or result.title,
            source_type=result.source_type, min_score=0.0, limit=25,
        )
        if c.is_pdf_link
    ]
    return pdfs[0].url if pdfs else None


def discover(
    portal: PortalSpec,
    *,
    query: str | None = None,
    client: httpx.Client | None = None,
    limit: int = 10,
    min_score: float = 0.1,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_UA,
    force_browser: bool = False,
    known_instruments: list[str] | None = None,
    known_instrument_ids: dict[str, str] | None = None,
    browser_session=None,
) -> list[DiscoveryResult]:
    """Discover candidate instrument URLs on a portal for a query.

    The query defaults to ``portal.search_query``. The page is fetched over HTTP
    unless the portal declares ``fetch_method: playwright`` (or ``force_browser``
    is set, or HTTP returns 403/429), in which case it is rendered with a
    headless browser. ``known_instruments`` (typically ``profile.known_instruments``)
    drives KNOWN / NEW tagging. Returns ranked :class:`DiscoveryResult`.
    """
    query = query or portal.search_query

    # A registered per-portal strategy (e.g. AU's OData API) takes precedence;
    # fall back to generic harvesting if it yields nothing.
    from lexora.collect.strategies import strategy_for

    strat = strategy_for(portal)
    if strat is not None:
        hits = strat(
            portal, query=query, limit=limit, min_score=min_score,
            timeout=timeout, known_instruments=known_instruments,
            known_instrument_ids=known_instrument_ids,
        )
        if hits:
            return hits

    page_url = _search_url(portal, query)
    use_browser = force_browser or portal.fetch_method is FetchMethod.playwright
    html, via = _fetch_page(
        page_url, force_browser=use_browser, client=client,
        timeout=timeout, user_agent=user_agent, browser_session=browser_session,
    )

    if not html:
        return []
    return harvest_candidates(
        html,
        base_url=page_url,
        query=query,
        source_type=portal.source_type,
        via=via,
        limit=limit,
        min_score=min_score,
        known_instruments=known_instruments,
    )


def _merge_into(agg: dict[str, DiscoveryResult], r: DiscoveryResult) -> None:
    """Fold a hit into the instrument-keyed accumulator, keeping the best-scoring
    representative but never losing a KNOWN tag or a captured full-text URL."""
    key = _canonical_key(r.url)
    cur = agg.get(key)
    if cur is None:
        agg[key] = r
        return
    winner, loser = (r, cur) if r.score > cur.score else (cur, r)
    # Preserve the stronger provenance signals across the merge.
    if winner.discovery_tag != TAG_KNOWN and loser.discovery_tag == TAG_KNOWN:
        winner.discovery_tag = TAG_KNOWN
        winner.matched_instrument = winner.matched_instrument or loser.matched_instrument
    winner.fulltext_url = winner.fulltext_url or loser.fulltext_url
    winner.n_variants = max(winner.n_variants, loser.n_variants)
    agg[key] = winner


def discover_for_indicators(
    portal: PortalSpec,
    indicators: list,
    *,
    client: httpx.Client | None = None,
    per_indicator_limit: int = 8,
    max_queries_per_indicator: int = 3,
    budget: int = 15,
    min_score: float = 0.1,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_UA,
    force_browser: bool = False,
    known_instruments: list[str] | None = None,
    known_instrument_ids: dict[str, str] | None = None,
) -> list[DiscoveryResult]:
    """Discover a *working set* of instruments for a list of indicators.

    For each indicator we run discovery on its concept phrases
    (``indicator.query_phrases``), union all hits, collapse them to one entry per
    instrument (``_canonical_key``), keep the best-scoring representative, and
    return the top ``budget`` instruments. This is what lets one ``map`` run reach
    the *family* of laws an indicator touches (flagship + sectoral statutes) on a
    full-text portal, instead of only the single top hit. Per-indicator mapping
    happens later — this stage only assembles candidates.

    Each surviving instrument records, in ``indicator_hits``, the submission ids
    of the indicators whose phrase surfaced it — so the mapper can score it only
    against the indicators it is actually relevant to.

    Note: browser-rendered portals (SG SSO) pay one page fetch per phrase, so
    ``max_queries_per_indicator`` caps the phrase count, and a phrase shared by
    several indicators is fetched once and attributed to all of them.
    """
    # Build the query -> attributed-indicators map. On a full-text portal we use
    # each indicator's concept phrases (and credit the surfacing indicators). On a
    # name-only portal (AU OData) concept phrases return nothing useful, so we
    # query the jurisdiction's known instrument NAMES instead and leave
    # attribution empty (the mapper then scores each against all indicators).
    phrase_indicators: dict[str, set[str]] = {}
    if portal.full_text:
        for ind in indicators:
            for phrase in ind.query_phrases(limit=max_queries_per_indicator):
                phrase_indicators.setdefault(phrase, set()).add(ind.submission_id)
        # Also query the known instrument NAMES directly. A full-text engine that
        # ranks on ALL words (SG SSO PhraseType=AllWords) under-recalls long
        # concept phrases, so the flagship/known laws can be missed entirely; a
        # name query reliably surfaces and KNOWN-tags them (attributed to no
        # single indicator -> scored against all, like the name-only path).
        for name in known_instruments or []:
            phrase_indicators.setdefault(name, set())
    else:
        for name in known_instruments or []:
            phrase_indicators.setdefault(name, set())

    # Browser portals (SG SSO) render every query; hold ONE Chromium session open
    # for the whole sweep so a burst of queries doesn't relaunch the browser each
    # time (slow, and trips anti-bot rate limits into serving the homepage).
    needs_browser = force_browser or portal.fetch_method is FetchMethod.playwright
    session_cm = None
    if needs_browser:
        # Use the browser's own realistic Chrome UA, NOT the polite HTTP bot UA —
        # anti-bot portals (SG SSO) serve 403/an empty shell to "Lexora/0.1".
        from lexora.collect.browser import DEFAULT_UA as BROWSER_UA
        from lexora.collect.browser import BrowserSession, is_available

        if is_available():
            session_cm = BrowserSession(user_agent=BROWSER_UA, timeout=timeout)

    agg: dict[str, DiscoveryResult] = {}
    indicators_by_key: dict[str, set[str]] = {}
    session = session_cm.__enter__() if session_cm is not None else None
    try:
        for phrase, ind_ids in phrase_indicators.items():
            for r in discover(
                portal, query=phrase, client=client, limit=per_indicator_limit,
                min_score=min_score, timeout=timeout, user_agent=user_agent,
                force_browser=force_browser, known_instruments=known_instruments,
                known_instrument_ids=known_instrument_ids, browser_session=session,
            ):
                _merge_into(agg, r)
                indicators_by_key.setdefault(_canonical_key(r.url), set()).update(ind_ids)
    finally:
        if session_cm is not None:
            session_cm.__exit__(None, None, None)

    for key, res in agg.items():
        res.indicator_hits = sorted(indicators_by_key.get(key, set()))

    ranked = sorted(agg.values(), key=lambda r: (r.score, r.n_variants), reverse=True)
    # Keep KNOWN instruments ahead of the budget cut: a flagship/known law must
    # not be squeezed out by a flood of lower-value NEW candidates (this is what
    # dropped MY's Cyber Security / Computer Crimes Acts before).
    known = [r for r in ranked if r.discovery_tag == TAG_KNOWN]
    rest = [r for r in ranked if r.discovery_tag != TAG_KNOWN]
    return (known + rest)[:budget]


__all__ = [
    "DiscoveryResult",
    "harvest_candidates",
    "discover",
    "discover_for_indicators",
    "resolve_fulltext",
]
