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

import collections
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import quote, urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from lexora.collect import http_cache
from lexora.collect.crawler import DEFAULT_UA
from lexora.models.source import FetchMethod, PortalSpec, SourceType

_LOG = logging.getLogger(__name__)

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
# Anchor texts that are a portal's own furniture, matched WHOLE and never as a
# substring. Malaysia's LOM portal put five of these into a 13-slot working set on
# 2026-08-01 -- "Ordinance", "Search", "Translated", "Top Hit (Weekly)", "See All..."
# were fetched, parsed and counted as instruments, crowding out the statutes.
#
# Whole-string equality is the point. "search" as a substring would throw away
# `Criminal Procedure Code` results about search and seizure, and "ordinance" would
# throw away every real Ordinance. A label is furniture only when it is the ENTIRE
# title; the same word inside a real title is signal.
_NAV_LABELS = frozenset({
    "search", "advanced search", "see all", "see all...", "see more", "more",
    "ordinance", "ordinances", "translated", "translation", "top hit", "top hits",
    "top hit (weekly)", "top hit (monthly)", "home", "back", "next", "previous",
    "all", "view all", "browse", "download", "print", "help",
})


def _is_nav_label(text: str) -> bool:
    """True when the anchor text IS a navigation label, not a title containing one."""
    return " ".join((text or "").split()).strip().lower().rstrip(".") in {
        lab.rstrip(".") for lab in _NAV_LABELS
    }


_RESULT_CONTAINER = re.compile(r"result|item|card|search|title|listing|row", re.I)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_YEARISH_FULL = re.compile(r"(?:19|20)\d{2}")  # a bare 4-digit year (for fullmatch)
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
    # Structured metadata read from the portal's own channel (register API field /
    # page label), when available — the generalizable source for the submission's
    # Law Number and Last Amended columns. Empty when the connector cannot supply
    # it; the citation layer then falls back to the curated anchor / LLM extractor.
    law_number: str = ""
    last_amended: str = ""
    # Lifecycle status from the portal channel / title marker (in_force / repealed
    # / draft / unknown), for the official enforced-only filter. Stored as the
    # InstrumentStatus *value* string; unknown by default. See classify/lifecycle.py.
    status: str = "UNKNOWN"


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


_QUERY_ECHO: dict[str, dict[tuple, set[str]]] = {}
_QUERY_ECHO_WARNED: set[tuple[str, tuple]] = set()
_QUERY_ECHO_LOCK = threading.Lock()
# How many DISTINCT queries must share one result set before we call it a dead search.
# Two is a coincidence a real portal produces (see the false positive in the docstring);
# three in one sweep is not.
_QUERY_ECHO_THRESHOLD = 3


def _warn_if_query_ignored(portal_key: str, query: str, ids: list) -> None:
    """Shout when a search endpoint stops searching.

    The 20 July submission mapped 150 Malaysian rows -- Personal Data Protection Act
    2010, Cyber Security Act 2024, Communications and Multimedia Act 1998 -- using
    exactly the GET this adapter still used on 1 August, when the same code found none
    of them. The code did not regress. The portal changed under it: `q` used to filter
    and now does not, and the response is still 200 with rows in it, so nothing looked
    wrong at any layer. Unit tests could not catch it either -- they mock the portal,
    which means they encode what we BELIEVE about it and stay green while the real one
    drifts.

    So assert something only a working search satisfies, on live traffic, for free:
    DIFFERENT queries must not keep coming back with an identical result set. One line
    in the log the first run after a portal changes is worth more than any number of
    green tests about a server we are not talking to.

    It counts distinct queries per result set rather than comparing each query with the
    one before it, because both blind spots of the pairwise version showed up on the
    same 2026-08-01 run:

    * **Alternating failure was invisible.** The Malaysian proxy answered roughly one
      query in three; a real answer between two dead ones made the consecutive pair
      differ, so the check stayed quiet through a half-blind sweep.
    * **A genuinely empty query looked like a dead search.** Australia tripped it on two
      policy-document titles that match no legislation at all -- the no-result fallback
      harvests the same page furniture both times, which is correct behaviour, not a
      broken endpoint.

    Counting fixes the first (a dead endpoint accumulates the same fingerprint all sweep
    long, however the hits are interleaved) and the threshold suppresses the second.
    Warns once per portal and result set: this belongs in the log once, not forty times.
    """
    fingerprint = tuple(ids)
    if not fingerprint:
        return
    with _QUERY_ECHO_LOCK:
        queries = _QUERY_ECHO.setdefault(portal_key, {}).setdefault(fingerprint, set())
        queries.add(query)
        if (len(queries) < _QUERY_ECHO_THRESHOLD
                or (portal_key, fingerprint) in _QUERY_ECHO_WARNED):
            return
        _QUERY_ECHO_WARNED.add((portal_key, fingerprint))
        sample = sorted(queries)[:_QUERY_ECHO_THRESHOLD]
    _LOG.error(
        "%s returned an IDENTICAL result set for %d different queries (%s) -- the "
        "endpoint is ignoring the query. Discovery is running blind: every phrase will "
        "map the same handful of instruments.",
        portal_key, len(queries), ", ".join(repr(q[:50]) for q in sample),
    )


def _is_chrome(href_l: str, text_l: str) -> bool:
    if _is_nav_label(text_l):
        return True
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


# Portal boilerplate that follows the instrument NAME in a result title (esp. SG
# SSO: "<Act name> Current version as at <date> <provision-number> <snippet>…").
# The snippet often quotes OTHER statutes (an Act whose text cross-references the
# PDPA), so KNOWN/NEW identity must be matched against the name slot only — not
# the body — or `token_set_ratio` reads the cross-reference as a self-match.
_TITLE_BOILERPLATE = re.compile(r"\s+(?:Current version\b|Repealed\b|Reprint\b).*", re.I)


def _name_part(title: str) -> str:
    """The instrument-name slot of a result title, stripped of portal boilerplate
    and capped so a trailing provision snippet can't spoof a KNOWN match."""
    cut = _TITLE_BOILERPLATE.split(title, maxsplit=1)[0]
    return cut[:90].strip() or title[:90]


# A legislative title's year is part of its identity: "... Act 1997" and "... Act 2023" are
# different instruments however similar their words.
_TITLE_YEAR_RX = re.compile(r"\b(?:1[6-9]\d{2}|20\d{2})\b")


def _years_conflict(a: str, b: str) -> bool:
    """True when both titles carry years and share none.

    ``token_set_ratio`` scores the INTERSECTION of the token sets, so it returns a perfect
    1.0 whenever the known name's tokens are a SUBSET of the candidate's — and Australia's
    theme-named omnibus Acts are exactly that shape. "Telecommunications Legislation
    Amendment (Information Disclosure, National Interest and Other Measures) Act 2023"
    scored a full match against "Telecommunications Legislation Amendment Act 1997", 26
    years apart, and would have been tagged KNOWN: a genuine discovery silently written off
    against the gold inventory, on the metric worth 20 of the 40 accuracy points.

    Set intersection, not equality, so a compiled title keeps matching its principal —
    "Privacy Act 1988 (Compilation No. 89, 2022)" still matches "Privacy Act 1988".
    """
    ya = set(_TITLE_YEAR_RX.findall(a))
    yb = set(_TITLE_YEAR_RX.findall(b))
    return bool(ya and yb and not (ya & yb))


def _fuzzy_known(
    text: str, known: list[str], *, year_strict: bool = False
) -> tuple[float, str | None]:
    """Best fuzzy match of `text` against the known instrument names, in [0,1].

    ``year_strict`` is for IDENTITY questions (is this the same instrument?) and must not
    be used for RELEVANCE. The two are different questions asked of the same function: a
    query for "privacy act 1988" should absolutely surface "Privacy Amendment Act 1990",
    and turning the guard on globally scored that candidate 0 and dropped it from the
    results entirely — a worse failure than the mis-tag the guard exists to prevent.
    """
    best, name = 0.0, None
    tl = text.lower()
    for inst in known:
        if year_strict and _years_conflict(inst, text):
            continue
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

    # Identity (KNOWN/NEW) matches the NAME slot only; relevance may use the body. The two
    # ask different questions of the same score, so only identity gets the year guard.
    name_slot = _name_part(title or context_text)
    fuzzy, _ = _fuzzy_known(name_slot, known) if known else (0.0, None)
    id_fuzzy, matched = (
        _fuzzy_known(name_slot, known, year_strict=True) if known else (0.0, None)
    )
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
        if id_fuzzy >= _KNOWN_THRESHOLD:
            tag = TAG_KNOWN
        elif instrument_like and q_overlap >= 0.5 and id_fuzzy < _NEW_CEILING:
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

    from lexora.classify.lifecycle import detect_status

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
            # Title-level lifecycle stamp (e.g. SG SSO "... (Repealed)"). Portal
            # connectors with a structured flag (AU) set this more authoritatively.
            status=detect_status(title=rec["title"]).value,
        )
        for rec in agg.values()
    ]
    results.sort(key=lambda r: (r.score, r.n_variants), reverse=True)
    return results[:limit]


# A real SG SSO results/landing page links to specific instrument pages
# (`/Act/PDPA2012`, `/Act/...`); the browse *shell* that the portal serves when it
# rate-limits a burst of headless navigations carries only browse nav
# (`/Browse/`, `/Acts-Supp/`, `/Act-Rev/`) and no `/Act/<slug>` link. `/Act/` is
# anchored to a slug char so `/Acts-Supp/` and `/Act-Rev/` don't match.
_SG_INSTRUMENT_LINK = re.compile(r'href="[^"]*/Act/[A-Za-z0-9]', re.I)


def sg_results_present(html: str | bytes) -> bool:
    """True if rendered SG SSO HTML carries at least one instrument link.

    A weaker signal than it looks: a blocked or unrendered page has no instrument
    link either. Use :func:`classify_page` to tell those apart; this only answers
    "did this page list any instruments".
    """
    text = html.decode("utf-8", "ignore") if isinstance(html, bytes) else html
    return bool(_SG_INSTRUMENT_LINK.search(text))


class PageOutcome(str, Enum):
    """What actually came back when we asked a portal a question.

    The distinction the pipeline used to be missing. "No instrument link on the page"
    was read as "this economy has no such law", when measurement showed it almost
    always meant the portal had refused to answer.
    """

    results = "RESULTS"        # a real page, listing instruments
    empty = "EMPTY"            # a real page, genuinely nothing matched
    blocked = "BLOCKED"        # a WAF/CDN refusal — the portal declined to answer
    unrendered = "UNRENDERED"  # navigation returned no document at all


# CDN/WAF refusal pages. Kept generic rather than SG-specific: CloudFront fronts
# several APAC government portals, and Round 2's economies will meet the same walls.
_BLOCK_SIGNATURES = re.compile(
    r"(?:ERROR:\s*The request could not be satisfied"
    r"|<h[12]>\s*40[36]\s*ERROR"
    r"|Request blocked"
    r"|Access Denied"
    r"|Attention Required!\s*\|\s*Cloudflare"
    r"|Checking your browser before accessing)",
    re.I,
)
# A document whose body carries no text and no markup of substance. Playwright
# returns exactly `<html><head></head><body></body></html>` when a navigation
# completes but the page never populated.
_BODY_CONTENT = re.compile(r"<body[^>]*>(.*?)</body>", re.I | re.S)


def classify_page(html: str | bytes, *, has_results: bool) -> PageOutcome:
    """Tell a real answer from a refusal.

    ``has_results`` is the portal-specific "did this page list instruments" verdict
    (e.g. :func:`sg_results_present`); everything else here is portal-agnostic.

    Measured on the 2026-07-27 Singapore run: of 67 query renders, 16 came back as a
    923-byte CloudFront "403 ERROR / Request blocked" page and 20 as a 39-byte empty
    document — and **not one** was a genuine empty result set. All 36 were recorded as
    "0 hits", i.e. as evidence that Singapore has no such law. Three were named known
    instruments (Computer Misuse Act, Criminal Procedure Code, Banking Act 1970).
    """
    text = html.decode("utf-8", "ignore") if isinstance(html, bytes) else (html or "")
    if not text.strip():
        return PageOutcome.unrendered
    if _BLOCK_SIGNATURES.search(text[:4096]):
        return PageOutcome.blocked
    body = _BODY_CONTENT.search(text)
    if body is not None and not body.group(1).strip():
        return PageOutcome.unrendered
    return PageOutcome.results if has_results else PageOutcome.empty


def page_is_usable(html: str | bytes) -> bool:
    """Retry predicate: re-render a refusal, accept a genuine answer.

    The old predicate retried whenever no instrument link was present, which spent
    three backoff renders (~20 s) on every empty answer *and* on every block — and
    then accepted the block anyway. Retrying a refusal is right; retrying an answer
    is not.
    """
    return classify_page(html, has_results=sg_results_present(html)) not in (
        PageOutcome.blocked, PageOutcome.unrendered,
    )


@dataclass
class AcquisitionLog:
    """Per-run tally of what the portals actually did with our questions.

    This is the metric that has to exist before any concurrency change can be
    honest: raising parallelism against a per-IP limiter buys wall-clock by turning
    answers into refusals, and without this counter that trade is invisible.
    """

    outcomes: collections.Counter = field(default_factory=collections.Counter)
    blocked_queries: list[str] = field(default_factory=list)

    def record(self, outcome: PageOutcome, query: str | None) -> None:
        with _ACQ_LOCK:
            self.outcomes[outcome.value] += 1
            if outcome in (PageOutcome.blocked, PageOutcome.unrendered) and query:
                self.blocked_queries.append(query)

    @property
    def refused(self) -> int:
        return self.outcomes[PageOutcome.blocked.value] + \
            self.outcomes[PageOutcome.unrendered.value]

    @property
    def asked(self) -> int:
        return sum(self.outcomes.values())

    def summary(self) -> str:
        if not self.asked:
            return "no portal queries"
        return (f"{self.asked} portal quer(ies): "
                f"{self.outcomes[PageOutcome.results.value]} answered, "
                f"{self.outcomes[PageOutcome.empty.value]} empty, "
                f"{self.outcomes[PageOutcome.blocked.value]} blocked, "
                f"{self.outcomes[PageOutcome.unrendered.value]} unrendered")


_ACQ_LOCK = threading.Lock()
_ACQUISITION = AcquisitionLog()


def acquisition_log() -> AcquisitionLog:
    """The process-wide acquisition tally."""
    return _ACQUISITION


# Answered search pages, keyed by the URL actually requested. Bounded because a run
# holds ~70 pages of 60-380 KB and there is no reason to keep the whole sweep alive
# once the working set is built; the pages are re-fetchable, this only removes
# same-run repeats.
_PAGE_MEMO: collections.OrderedDict = collections.OrderedDict()
_PAGE_MEMO_MAX = 256
_PAGE_MEMO_LOCK = threading.Lock()


def _page_memo_get(key: tuple[str, str]) -> tuple[str | bytes, str] | None:
    with _PAGE_MEMO_LOCK:
        hit = _PAGE_MEMO.get(key)
        if hit is not None:
            _PAGE_MEMO.move_to_end(key)
        return hit


def _page_memo_put(key: tuple[str, str], html: str | bytes, via: str) -> None:
    with _PAGE_MEMO_LOCK:
        _PAGE_MEMO[key] = (html, via)
        _PAGE_MEMO.move_to_end(key)
        while len(_PAGE_MEMO) > _PAGE_MEMO_MAX:
            _PAGE_MEMO.popitem(last=False)


def reset_page_memo() -> None:
    """Drop memoised pages (one run per memo — a later run must re-ask)."""
    with _PAGE_MEMO_LOCK:
        _PAGE_MEMO.clear()


def reset_acquisition_log() -> AcquisitionLog:
    """Start a fresh tally (one per economy run)."""
    global _ACQUISITION
    with _ACQ_LOCK:
        _ACQUISITION = AcquisitionLog()
    return _ACQUISITION


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
    is_valid=None,
    render_retries: int = 0,
    render_wait_until: str = "networkidle",
) -> tuple[str | bytes, str]:
    """Fetch a page's HTML, escalating to a headless browser on anti-bot 403/429
    (or when ``force_browser``). Returns ``(html, via)`` where via is http|browser.

    ``browser_session`` (a :class:`browser.BrowserSession`) renders through one
    persistent Chromium instead of relaunching per call — used by
    :func:`discover_for_indicators` so a burst of SG SSO queries stays stable.
    ``is_valid``/``render_retries`` are forwarded to the session render so an
    anti-bot empty shell is re-rendered with backoff (see :func:`sg_results_present`)."""
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
            html = browser_session.render(
                url, timeout=timeout, retries=render_retries, is_valid=is_valid,
                wait_until=render_wait_until,
            ).html
        else:
            from lexora.collect.browser import render

            html = render(url, timeout=timeout, wait_until=render_wait_until).html
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
    is_valid=None,
    render_retries: int = 0,
    render_wait_until: str = "networkidle",
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
            _warn_if_query_ignored(str(portal.url), query, [h.url for h in hits])
            return hits

    page_url = _search_url(portal, query)
    use_browser = force_browser or portal.fetch_method is FetchMethod.playwright

    # Ask each distinct question once per run. The sweep, the amendment pass, the
    # child-regulation pass and the regulator pass all reach `discover` independently
    # and re-issue queries the others have already asked; on a browser portal that is
    # a 3-10 s render for a page we are still holding. Only *answered* pages are
    # memoised — a refusal must be re-askable, that is the whole point of the late
    # retry. Harvesting still runs per call with the caller's own limit/min_score, so
    # a memo hit is byte-for-byte the result a re-fetch would have produced.
    memo_key = (str(portal.url), page_url)
    cached = _page_memo_get(memo_key)
    if cached is not None:
        html, via = cached
    else:
        html, via = _fetch_page(
            page_url, force_browser=use_browser, client=client,
            timeout=timeout, user_agent=user_agent, browser_session=browser_session,
            is_valid=is_valid, render_retries=render_retries,
            render_wait_until=render_wait_until,
        )

    hits = harvest_candidates(
        html,
        base_url=page_url,
        query=query,
        source_type=portal.source_type,
        via=via,
        limit=limit,
        min_score=min_score,
        known_instruments=known_instruments,
    ) if html else []

    # The same check as on the strategy path above, and this is the branch that
    # actually needed it: when Malaysia's API answered 500 the adapter returned
    # nothing, discovery fell through to HTML harvesting, and the harvest produced
    # the SAME handful of portal links for every phrase in the sweep. A guard that
    # lives inside one adapter is a guard the failure walks around.
    _warn_if_query_ignored(str(portal.url), query, [h.url for h in hits])

    # Record what the portal did, here rather than in the sweep, because the
    # follow-on amendment / child-regulation / regulator passes call `discover`
    # directly and account for the majority of a run's queries (measured 2026-07-27:
    # 68 of 92 renders). A refusal counted only in the main sweep is a refusal not
    # counted at all.
    outcome = classify_page(html, has_results=bool(hits))
    if cached is None:
        acquisition_log().record(outcome, query)
        if outcome not in (PageOutcome.blocked, PageOutcome.unrendered):
            _page_memo_put(memo_key, html, via)
        else:
            _LOG.warning("  portal REFUSED (%s): %s",
                         outcome.value, (query or page_url)[:70])
    return hits


# An Act number in a URL/title/filename: "act=709", "Act 709", "ACT 854.pdf",
# "Act A1727". Used to collapse the many PDF variants one Act surfaces under
# (e.g. "Act 709 ori.pdf", "ACT 709-REPRINT 2023.pdf", "act-detail.php?act=709").
_ACT_NO = re.compile(r"act[=_\s-]*([a-z]?\d{1,4})\b", re.I)
# Singapore (and any jurisdiction that restarts numbering each year) cites an Act as
# "Act 19 of 2021". There, the bare number is NOT an identity: measured on the round-1
# submission, "act:19" collapsed Act 19 of 2021, of 2023 AND of 2024 onto one key, and
# "act:6" collapsed Act 6 of 2024 with Act 6 of 2025. `_merge_into` keeps one
# representative per key, so the others leave the working set before they are ever
# fetched -- a law dropped where nothing downstream can notice. Malaysia's "Act 709"
# carries no "of <year>" and is unaffected.
_ACT_NO_OF_YEAR = re.compile(
    r"\bact[=_\s-]*([a-z]?\d{1,4})\s+of\s+((?:19|20)\d{2})\b", re.I
)


def _identity_key(r: DiscoveryResult) -> str:
    """Instrument identity for cross-query dedup. Prefer a portal-native Act
    number (collapses an Act's many full-text PDF variants onto one entry so they
    don't each consume a budget slot); fall back to the URL canonical key. A
    4-digit year (e.g. 'Act 2010') is not an Act number and is ignored, and a
    year-scoped number ("Act 19 of 2021") keeps its year — see `_ACT_NO_OF_YEAR`."""
    text = f"{r.url} {r.title}".lower()
    host = urlparse(r.url).netloc.lower()
    scoped = _ACT_NO_OF_YEAR.search(text)
    if scoped:
        return f"{host}:act:{scoped.group(1)}/{scoped.group(2)}"
    for m in _ACT_NO.findall(text):
        digits = m.lstrip("abcdefghijklmnopqrstuvwxyz")
        if not m[0].isalpha() and _YEARISH_FULL.fullmatch(digits):
            continue
        return f"{host}:act:{m}"
    return _canonical_key(r.url)


def _merge_into(agg: dict[str, DiscoveryResult], r: DiscoveryResult, key: str | None = None) -> None:
    """Fold a hit into the instrument-keyed accumulator, keeping the best-scoring
    representative but never losing a KNOWN tag or a captured full-text URL."""
    key = key or _canonical_key(r.url)
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
    # Keep an authoritative lifecycle verdict across the merge (a connector-set
    # status on either side beats the winner's title-only "UNKNOWN").
    if winner.status == "UNKNOWN" and loser.status != "UNKNOWN":
        winner.status = loser.status
    winner.law_number = winner.law_number or loser.law_number
    winner.last_amended = winner.last_amended or loser.last_amended
    agg[key] = winner


def _get_embedder(use_semantic: bool):
    """Return a cached embedder if the optional dense backend is installed and
    enabled, else None (callers degrade to keyword-only ranking)."""
    if not use_semantic:
        return None
    from lexora.semantic import embedder as emb_mod

    if not emb_mod.is_available():
        return None
    try:
        return emb_mod.get_embedder()
    except Exception:
        return None


def _semantic_rerank(
    agg: dict[str, DiscoveryResult],
    indicators_by_key: dict[str, set[str]],
    indicators: list,
    embedder,
    *,
    bonus: float = 0.2,
) -> None:
    """Add a concept-similarity *bonus* to each candidate's score in place.

    Token-overlap / fuzzy scores saturate near 1.0 on a full-text portal (a long
    Act title trivially contains the query words), so the budget cut can't tell a
    record-keeping statute from a stray match. The cosine between a candidate's
    title and the concept(s) that surfaced it adds a discriminating tie-breaker.

    It is strictly *additive* (``score += bonus * cosine``), never a blend that
    could pull a strong keyword hit down: on portals whose titles are filenames
    (MY Fess, e.g. ``Act 730_ONLINE.pdf``) a title has near-zero cosine, and a
    weighted average would wrongly demote a correctly-found statute out of budget.
    Only-up means the dense signal can promote a clean-title sectoral law without
    ever penalising a filename-titled one.
    """
    if embedder is None or not agg:
        return
    concept_by_id = {
        ind.submission_id: " ".join([ind.name, ind.description, *ind.query_phrases()])
        for ind in indicators
    }
    keys = list(agg.keys())
    title_vecs = embedder.encode([agg[k].title or "" for k in keys])
    concept_ids = list(concept_by_id)
    concept_vecs = embedder.encode([concept_by_id[i] for i in concept_ids])
    cidx = {cid: i for i, cid in enumerate(concept_ids)}

    for row, key in enumerate(keys):
        res = agg[key]
        hit_ids = indicators_by_key.get(key) or set(concept_ids)
        rows = [cidx[i] for i in hit_ids if i in cidx]
        if not rows or title_vecs.size == 0:
            continue
        sim = float((concept_vecs[rows] @ title_vecs[row]).max())
        res.score = min(1.0, res.score + bonus * max(sim, 0.0))


def discover_for_indicators(
    portal: PortalSpec,
    indicators: list,
    *,
    client: httpx.Client | None = None,
    per_indicator_limit: int = 8,
    max_queries_per_indicator: int = 4,
    budget: int = 20,
    min_score: float = 0.1,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_UA,
    force_browser: bool = False,
    known_instruments: list[str] | None = None,
    known_instrument_ids: dict[str, str] | None = None,
    use_semantic: bool = True,
    extra_seed_queries: list[str] | None = None,
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
    # AU full-catalogue enumeration as the discovery source (opt-in via
    # LEXORA_AU_ENUMERATE). The AU portal has no concept search, so name queries
    # miss sectoral/surveillance statutes; enumerating the principal-Act catalogue
    # and judging each on its EPUB full text recovers them. The full-text judge's
    # per-indicator verdict IS the attribution, so the mapper uses it directly
    # (regime-1) — no per-document re-judge needed. Needs an LLM key; if absent the
    # judge is None and we fall through to the normal name/crosswalk discovery.
    from lexora.collect.au_enumerate import enumerate_enabled

    if enumerate_enabled() and "legislation.gov.au" in urlparse(str(portal.url)).netloc.lower():
        from lexora.classify.brute_judge import make_brute_judge
        from lexora.collect.au_enumerate import enumerate_au_candidates

        judge = make_brute_judge(enabled=True)
        if judge is not None:
            return enumerate_au_candidates(
                indicators, judge=judge, source_type=portal.source_type,
                known_instruments=known_instruments,
                catalogue_cache=os.environ.get("LEXORA_AU_CATALOGUE_CACHE"),
                verdict_cache=os.environ.get("LEXORA_AU_ENUM_VERDICTS"),
                judge_workers=int(os.environ.get("LEXORA_AU_ENUM_WORKERS", "32")),
                timeout=timeout, user_agent=user_agent,
            )

    # MY statute-book enumeration (opt-in via LEXORA_MY_ENUMERATE). Malaysia's search
    # proxy refuses about two requests in five and on 20 July stopped filtering
    # altogether, so discovery there was hostage to a service we do not control. The
    # portal's own listings give all 1,287 Acts in four requests; a two-stage judge
    # (title, then table of contents, OCR'ing the scans) narrows that to a working pool.
    #
    # Unlike Australia this does NOT return the enumeration alone. Enumeration decides
    # relevance, and a relevance judgement can be wrong: it cut Income Tax Act 1967 and
    # Service Tax Act 2018, both gold. A KNOWN instrument must never depend on being
    # judged relevant -- we already know it is -- so the gold backstop is merged on top
    # and the two together are what discovery returns.
    from lexora.collect.my_enumerate import enumerate_enabled as my_enumerate_enabled

    if my_enumerate_enabled() and "lom.agc.gov.my" in urlparse(str(portal.url)).netloc.lower():
        from lexora.classify.brute_judge import make_brute_judge
        from lexora.collect.my_enumerate import enumerate_my_candidates
        from lexora.collect.strategies import known_resolver_for

        judge = make_brute_judge(enabled=True)
        if judge is not None:
            def _judge(text: str, inds: list) -> set[str]:
                return judge.relevant(text, inds)

            enumerated = enumerate_my_candidates(
                indicators, title_judge=_judge, toc_judge=_judge,
                source_type=portal.source_type,
                known_instruments=known_instruments,
                known_instrument_ids=known_instrument_ids,
                verdict_cache=os.environ.get("LEXORA_MY_ENUM_VERDICTS"),
                judge_workers=int(os.environ.get("LEXORA_MY_ENUM_WORKERS", "16")),
                timeout=timeout, client=client,
            )
            merged: dict[str, DiscoveryResult] = {}
            for r in enumerated:
                _merge_into(merged, r, _identity_key(r))
            resolve_known = known_resolver_for(portal)
            if resolve_known is not None and known_instrument_ids:
                recovered = 0
                for r in resolve_known(portal, known_instrument_ids=known_instrument_ids,
                                       timeout=timeout, client=client):
                    key = _identity_key(r)
                    if key not in merged:
                        recovered += 1
                    _merge_into(merged, r, key)
                if recovered:
                    _LOG.info("gold backstop: %d known instrument(s) the enumeration "
                              "did not shortlist, added anyway", recovered)
            return sorted(merged.values(), key=lambda r: (r.score, r.n_variants),
                          reverse=True)

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

    # Secondary-source recall booster (WS-S, USE 1): query the primary-law NAMES a
    # secondary tracker pointed to but our concept/known queries might miss. No
    # attribution (scored against all indicators, like a name query), and NOT added
    # to known_instruments — so a seed keeps its natural KNOWN/NEW fuzzy tag.
    for seed in extra_seed_queries or []:
        phrase_indicators.setdefault(seed, set())

    # Browser portals (SG SSO) render every query; hold ONE Chromium session open
    # for the whole sweep so a burst of queries doesn't relaunch the browser each
    # time (slow, and trips anti-bot rate limits into serving the homepage).
    needs_browser = force_browser or portal.fetch_method is FetchMethod.playwright
    # SG SSO rate-limits a burst of headless navigations by serving the empty
    # browse shell; validate each render and re-render with backoff when no
    # instrument link is present (see `sg_results_present`). Other browser portals
    # keep the no-retry default.
    is_valid = None
    render_retries = 0
    render_wait_until = "networkidle"
    sweep_interval = 0.0
    if "sso.agc.gov.sg" in urlparse(str(portal.url)).netloc.lower():
        # Retry a REFUSAL, not an answer. The former predicate re-rendered whenever no
        # instrument link was present, so a genuine empty result cost three backoff
        # renders (~20 s) before being accepted — and a CloudFront block cost the same
        # three and was then accepted as "no such law". See `classify_page`.
        is_valid = page_is_usable
        render_retries = 3
        # SG SSO's search page long-polls, so a navigation NEVER reaches the
        # `networkidle` state — the default render then deterministically times
        # out (60s) before any retry. Wait for the `load` event instead (it fires
        # regardless of the background long-poll XHR); settle_ms + the
        # `sg_results_present` retry remain the content backstop.
        render_wait_until = "load"
        # SSO applies a CUMULATIVE per-IP rate limit: a rapid burst of navigations
        # (the name + concept query sweep, now ~12 known names) starts serving the
        # empty browse shell, which drops whole instruments from the working set even
        # though they are discoverable. Space the navigations so the burst stays under
        # the limit (the per-render `sg_results_present` retry handles a stray shell).
        # The SG sweep is ~40 unique phrases (28 concept + 12 known names), so the
        # spacing fires ~39 times; at 1.5s that is only ~+58s wall-time against the
        # 10-min budget while roughly halving the burst rate the limiter sees — cheap
        # insurance against the cumulative throttle, since render time dominates anyway.
        sweep_interval = 1.5
    # A portal may declare its own spacing and override the default above. SSO is not
    # the only limiter we have met: Malaysia's Fess proxy answered a single POST with a
    # filtered result set and then 500'd every request of the sweep that followed
    # (measured 2026-08-01, 16 consecutive 500s starting at the FIRST query), which the
    # adapter turns into an empty list and discovery silently replaces with harvested
    # page furniture. Spacing is the only knob that addresses the burst itself.
    if portal.sweep_interval is not None:
        sweep_interval = portal.sweep_interval
    if http_cache.serving_from_recording():
        # Replay renders come off the local store, so the cumulative per-IP limiter
        # this spacing exists for is not in the loop at all. Measured on the Singapore
        # sweep: 87s of the ~90s this leg took was these two sleeps.
        sweep_interval = 0.0
    session_cm = None
    if needs_browser:
        # Use the browser's own realistic Chrome UA, NOT the polite HTTP bot UA —
        # anti-bot portals (SG SSO) serve 403/an empty shell to "Lexora/0.1".
        from lexora.collect.browser import DEFAULT_UA as BROWSER_UA
        from lexora.collect.browser import BrowserSession, is_available

        if is_available():
            session_cm = BrowserSession(user_agent=BROWSER_UA, timeout=timeout)

    embedder = _get_embedder(use_semantic)

    agg: dict[str, DiscoveryResult] = {}
    indicators_by_key: dict[str, set[str]] = {}
    session = session_cm.__enter__() if session_cm is not None else None
    # This sweep is the long silent stretch of a run: on a browser portal it renders one
    # page per phrase, spaced under the portal's rate limit, and until it finishes the
    # engine has nothing to print. Reporting each query is what turns "is it hung?" into
    # a visible count — which matters most in front of an audience.
    _LOG.info("discovery sweep: %d quer(ies) on %s", len(phrase_indicators), portal.name)
    try:
        for i, (phrase, ind_ids) in enumerate(phrase_indicators.items()):
            # Space the queries under the portal's rate limit. This used to also require
            # a browser session, which silently made the knob unreachable for every
            # portal that is not SG SSO -- an API portal has no session, so any interval
            # it declared was skipped and the sweep ran at full speed anyway.
            if i and sweep_interval:
                time.sleep(sweep_interval)
            # A single flaky query must not abort the whole sweep. render() already
            # degrades a hung/failed browser nav to an empty result, but guard the
            # call anyway so any other per-query fault (strategy, harvest) just skips
            # that phrase instead of losing every instrument found so far.
            try:
                hits = discover(
                    portal, query=phrase, client=client, limit=per_indicator_limit,
                    min_score=min_score, timeout=timeout, user_agent=user_agent,
                    force_browser=force_browser, known_instruments=known_instruments,
                    known_instrument_ids=known_instrument_ids, browser_session=session,
                    is_valid=is_valid, render_retries=render_retries,
                    render_wait_until=render_wait_until,
                )
            except Exception:  # noqa: BLE001 — a failed query skips its phrase, not the run
                continue
            for r in hits:
                key = _identity_key(r)
                _merge_into(agg, r, key)
                indicators_by_key.setdefault(key, set()).update(ind_ids)
            _LOG.info("  [%2d/%2d] %-46s %2d hit(s), %d instrument(s) so far",
                      i + 1, len(phrase_indicators), phrase[:46], len(hits), len(agg))

        # Second pass over the queries the portal refused. Retrying a rate-limited
        # query inside its own hot window is what the per-render backoff already
        # tried and, measured, kept failing; by the end of a sweep the window has had
        # minutes to clear, so one late attempt is worth far more than a fourth
        # immediate one. Only refusals come back here — an empty answer is an answer.
        refused = list(dict.fromkeys(acquisition_log().blocked_queries))
        retryable = [q for q in refused if q in phrase_indicators]
        if retryable:
            _LOG.info("retrying %d refused quer(ies) after the sweep cooled",
                      len(retryable))
            for phrase in retryable:
                if sweep_interval:
                    time.sleep(sweep_interval)
                try:
                    hits = discover(
                        portal, query=phrase, client=client, limit=per_indicator_limit,
                        min_score=min_score, timeout=timeout, user_agent=user_agent,
                        force_browser=force_browser, known_instruments=known_instruments,
                        known_instrument_ids=known_instrument_ids, browser_session=session,
                        is_valid=is_valid, render_retries=render_retries,
                        render_wait_until=render_wait_until,
                    )
                except Exception:  # noqa: BLE001 — a late retry never breaks a run
                    continue
                for r in hits:
                    key = _identity_key(r)
                    _merge_into(agg, r, key)
                    indicators_by_key.setdefault(key, set()).update(
                        phrase_indicators[phrase])
                if hits:
                    _LOG.info("  recovered %-46s %2d hit(s)", phrase[:46], len(hits))
    finally:
        if session_cm is not None:
            session_cm.__exit__(None, None, None)

    # Name-only portals (AU) can't send a concept query to the server, so the
    # phrase loop above only re-finds known names. The semantic crosswalk bridges
    # concepts -> statute titles locally and is the only source of NEW AU hits.
    if embedder is not None and not portal.full_text:
        from lexora.collect.strategies import concept_strategy_for

        crosswalk = concept_strategy_for(portal)
        if crosswalk is not None:
            for r in crosswalk(
                portal, indicators, embedder=embedder, timeout=timeout, client=client,
                known_instruments=known_instruments,
                known_instrument_ids=known_instrument_ids,
            ):
                key = _identity_key(r)
                _merge_into(agg, r, key)
                indicators_by_key.setdefault(key, set()).update(r.indicator_hits)

    # Gold backstop. A KNOWN instrument is one whose identity we already have, so losing
    # it to a search failure is losing something we never had to search for. Where the
    # portal publishes a numbered listing of its own, resolve the known Acts from that
    # listing and merge in the ones the sweep did not surface. Measured on Malaysia: the
    # Fess proxy refuses about two queries in five, and on 20 July it silently stopped
    # filtering altogether and cost the run every gold instrument at once.
    #
    # It merges rather than replaces, so an Act discovery DID find keeps the indicator
    # attribution its concept query earned -- the backstop knows the law exists, not
    # which indicator it answers.
    if known_instrument_ids:
        from lexora.collect.strategies import known_resolver_for

        resolve_known = known_resolver_for(portal)
        if resolve_known is not None:
            recovered = 0
            for r in resolve_known(
                portal, known_instrument_ids=known_instrument_ids,
                timeout=timeout, client=client,
            ):
                key = _identity_key(r)
                if key not in agg:
                    recovered += 1
                _merge_into(agg, r, key)
            if recovered:
                _LOG.info("gold backstop: %d known instrument(s) the sweep missed, "
                          "resolved from the portal's own listing", recovered)

    for key, res in agg.items():
        res.indicator_hits = sorted(indicators_by_key.get(key, set()))

    # On full-text portals the keyword score saturates; a dense re-rank gives the
    # budget cut a discriminating signal so the right sectoral law survives.
    if embedder is not None and portal.full_text:
        _semantic_rerank(agg, indicators_by_key, indicators, embedder)

    ranked = sorted(agg.values(), key=lambda r: (r.score, r.n_variants), reverse=True)
    # KNOWN instruments are the official gold inventory: every one of them must be mapped,
    # so they are never cut (this is what once dropped MY's Cyber Security / Computer
    # Crimes Acts). But they must not SPEND the budget either.
    #
    # They used to: `(known + rest)[:budget]` let a large gold inventory fill the budget on
    # its own and starve discovery of every NEW candidate. Measured on the 2026-07-12 run,
    # all three economies came back with new_instruments = 0 -- Malaysia matched 19 KNOWN
    # against a budget of 20, so exactly one non-KNOWN slot survived, while Fess had in
    # fact surfaced 35 NEW candidates including Malaysia's Data Sharing Act 2025. NEW
    # provisions are worth 20 of the 40 Substantive Accuracy points, and the budget was
    # silently throwing them all away.
    #
    # So the budget now means what its name says: how many candidates to EXPLORE beyond
    # the gold inventory. The working set is every KNOWN instrument plus the top `budget`
    # of the rest.
    known = [r for r in ranked if r.discovery_tag == TAG_KNOWN]
    rest = [r for r in ranked if r.discovery_tag != TAG_KNOWN]
    return known + rest[:budget]


def discover_secondary(
    profile,
    indicators: list,
    *,
    timeout: float = 45.0,
    limit_per_portal: int = 40,
) -> list[DiscoveryResult]:
    """Harvest guidance/codes/notices from a profile's *secondary* portals.

    The primary portal (``portals[0]``) is the statute portal handled by
    :func:`discover_for_indicators`. Regulator portals (PDPC, IMDA, OAIC, …) carry
    the subsidiary-instrument gold rows that never appear on the statute portal;
    each has a registered connector (:func:`strategies.connector_for`). A single
    headless-browser session is shared across all browser-rendered connectors.
    """
    from lexora.collect.strategies import connector_for

    secondary = [p for p in profile.portals[1:] if connector_for(p) is not None]
    if not secondary:
        return []

    # Guidance connectors render JS hub pages, so share one browser session.
    from lexora.collect.browser import DEFAULT_UA as BROWSER_UA
    from lexora.collect.browser import BrowserSession, is_available

    session_cm = BrowserSession(user_agent=BROWSER_UA, timeout=timeout) if is_available() else None

    session = session_cm.__enter__() if session_cm is not None else None
    results: list[DiscoveryResult] = []
    try:
        for portal in secondary:
            conn = connector_for(portal)
            try:
                results.extend(conn(
                    portal, indicators, browser_session=session, limit=limit_per_portal,
                    timeout=timeout, known_instruments=profile.known_instruments,
                    known_instrument_ids=profile.known_instrument_ids,
                ))
            except Exception:
                continue
    finally:
        if session_cm is not None:
            session_cm.__exit__(None, None, None)
    return results


__all__ = [
    "DiscoveryResult",
    "harvest_candidates",
    "discover",
    "discover_for_indicators",
    "discover_secondary",
    "resolve_fulltext",
    "sg_results_present",
]
