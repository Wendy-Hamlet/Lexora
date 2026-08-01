"""Malaysia's statute book as the portal itself publishes it, in a handful of requests.

`lom.agc.gov.my` renders its "Principal Acts" / "Amendment Acts" / "Repealed" listings
from DataTables JSON feeds, and those feeds are a different service from the Fess search
proxy the discovery layer normally talks to. That matters, because measured 2026-08-01
the search proxy answers roughly three requests in five and returns 500 for the rest,
while these feeds answered every request in the survey. They also carry, per Act and for
free, four things we currently pay for or go without:

    * the ENGLISH title              -- so a law is named, not numbered. Before this,
                                        four Communications and Multimedia Act 1998
                                        provisions sat in the submission under the name
                                        "Act 588", which a scorer matching a gold
                                        inventory by name reads as a miss.
    * a direct English PDF URL       -- so a known instrument no longer has to be found
                                        by search before it can be fetched.
    * the commencement remark        -- including "NOT YET IN FORCE".
    * the repeal chain               -- `REPEALEDBY` plus the repealing Act's title, which
                                        is authoritative and costs nothing, where our own
                                        currency layer has to infer it.

This module only READS the feeds and normalises them. It decides nothing about relevance:
the inventory is a candidate pool of ~1,287 Acts with no subject metadata at all, and a
title-only keyword filter was measured to reach 5 of the 8 Malaysian gold statutes with
Cyber Security Act 2024 scoring exactly zero -- titles do not contain provisions. Whoever
uses this pool has to bring its own judgement of relevance.

Every request goes through httpx, so a recorded run captures the feeds and an offline
replay serves them.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field

import httpx

_LOG = logging.getLogger(__name__)

BASE = "https://lom.agc.gov.my"

# DataTables sends these; the feeds answer without them, but a request that looks like
# the page's own is the one least likely to be treated as a robot.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": f"{BASE}/principal.php?type=original",
    "X-Requested-With": "XMLHttpRequest",
}

# (feed, title field, commencement field, English-PDF field, act-number field).
# `principal` and `updated` overlap heavily; both are read because a reprint sometimes
# carries an English PDF where the original record has none. `reprint` and `revised` are
# deliberately absent: revised adds Malay-only reprints, and reprint answered 500 during
# the survey, which is a reason to leave it out rather than to depend on it.
_FEEDS: tuple[tuple[str, str, str, str, str], ...] = (
    ("json-principal-2024.php", "LEGISLATIONTITLEBI", "COMMENCEMENTREMARKBI",
     "DOC2DOWNLOADBI", "ACTNO_LEGISLATION"),
    ("json-updated-2024.php", "LEGISLATIONTITLEBI", "COMMENCEMENTREMARKBI",
     "DOC2DOWNLOADBI", "ACTNO_LEGISLATION"),
    # `LEGISLATIONTITLEBI`, NOT `TajukBI`. On the amendment feed `TajukBI` carries the
    # MALAY title -- Act A1727 has "AKTA PERLINDUNGAN DATA PERIBADI (PINDAAN) 2024" in
    # the field named BI, and the English name only in LEGISLATIONTITLEBI. A field named
    # for a language is not evidence about the language of its contents.
    ("json-amendment-2024.php", "LEGISLATIONTITLEBI", "COMMENCEMENTREMARKBI",
     "DOC2DOWNLOADBI", "ACTNO_LEGISLATION"),
)
_REPEALED_FEED = "json-repealed-2024.php"

_TAG = re.compile(r"<[^>]+>")
_HREF = re.compile(r'href="([^"]+)"')


@dataclass
class InventoryEntry:
    """One Act as the portal lists it. Every field is the portal's own, never inferred."""

    act_no: str
    title_en: str = ""
    commencement: str = ""
    pdf_url: str = ""
    repealed_by: str = ""       # act number of the repealing instrument
    repealed_by_title: str = ""
    feeds: list[str] = field(default_factory=list)

    @property
    def repealed(self) -> bool:
        return bool(self.repealed_by or self.repealed_by_title)

    @property
    def in_force(self) -> bool | None:
        """False when the portal says NOT YET IN FORCE, None when it does not say."""
        if not self.commencement:
            return None
        return "not yet in force" not in self.commencement.lower()


def _text(value: str | None) -> str:
    return re.sub(r"\s+", " ", _TAG.sub(" ", value or "")).strip()


def _is_malay(title: str) -> bool:
    """A Malaysian Act's Malay title opens with AKTA / PERINTAH / PERATURAN / KAEDAH."""
    return bool(re.match(r"(?i)^(akta|perintah|peraturan|kaedah|ordinan)\b", title.strip()))


def _better_title(current: str, candidate: str) -> str:
    """Prefer an English title, then the first one we saw.

    NOT "the longest wins", which was the first rule here and picked the wrong language
    on the very first Act it met: A1727's Malay title is one character longer than its
    English one, so the amendment feed named the Personal Data Protection (Amendment)
    Act 2024 in Malay.
    """
    if not candidate:
        return current
    if not current:
        return candidate
    if _is_malay(current) and not _is_malay(candidate):
        return candidate
    return current


def _url(cell: str | None) -> str:
    """Absolute URL from a listing cell's anchor.

    The feeds hand back page-relative hrefs (`../../../ilims/upload/...`). Joining them
    naively against the feed URL is what put a bare `downloadPDF.php?...` into the
    crawler once and took the whole Malaysian leg down with an UnsupportedProtocol.
    """
    m = _HREF.search(cell or "")
    if not m:
        return ""
    href = m.group(1).strip().replace("\\/", "/")
    if href.startswith(("http://", "https://")):
        return href
    return httpx.URL(f"{BASE}/").join(href.lstrip("./")).__str__()


def _generated_pdf(cell: str | None) -> str:
    """URL from a `<field>generatepdf` blob: ``{"path": "/upload/...", "docName": "x.pdf"}``.

    Act A1727 -- the Personal Data Protection (Amendment) Act 2024, a gold instrument --
    has an EMPTY direct anchor and its English PDF only here.
    """
    if not cell:
        return ""
    try:
        blob = json.loads(cell)
    except (TypeError, ValueError):
        return ""
    path, name = (blob.get("path") or "").strip(), (blob.get("docName") or "").strip()
    if not path or not name:
        return ""
    return f"{BASE}/ilims{path}{name}"


def _fetch_json(client: httpx.Client, feed: str, timeout: float) -> dict:
    response = client.get(f"{BASE}/{feed}", headers=_HEADERS, timeout=timeout)
    if response.status_code != 200:
        raise httpx.HTTPStatusError(
            f"{feed} -> HTTP {response.status_code}", request=response.request,
            response=response)
    return json.loads(response.text)


_CACHE: dict[str, InventoryEntry] | None = None
_CACHE_LOCK = threading.Lock()


def fetch_inventory(
    client: httpx.Client | None = None, *, timeout: float = 120.0, refresh: bool = False,
) -> dict[str, InventoryEntry]:
    """Every Act the portal lists, keyed by act number.

    A feed that fails is logged and skipped rather than raising: three quarters of an
    inventory is worth far more than an exception, and the caller can see what it got
    from the entry count. Cached for the process because a run reads it from several
    places and the payload is ~2.5 MB.
    """
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is not None and not refresh:
            return _CACHE

    owns = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=timeout)
    entries: dict[str, InventoryEntry] = {}
    try:
        for feed, tkey, ckey, pkey, akey in _FEEDS:
            try:
                payload = _fetch_json(client, feed, timeout)
            except Exception as exc:  # noqa: BLE001 — a missing feed is not a dead run
                _LOG.warning("Malaysian inventory: %s unavailable (%s: %s)",
                             feed, type(exc).__name__, exc)
                continue
            for row in payload.get("records") or []:
                act_no = str(row.get(akey) or "").strip()
                if not act_no:
                    continue
                entry = entries.setdefault(act_no, InventoryEntry(act_no=act_no))
                entry.feeds.append(feed)
                entry.title_en = _better_title(entry.title_en, _text(row.get(tkey)))
                entry.commencement = entry.commencement or (_text(row.get(ckey)) if ckey else "")
                entry.pdf_url = entry.pdf_url or (_url(row.get(pkey)) if pkey else "")
                if not entry.pdf_url:
                    # The direct anchor is empty on some amendment rows, but the row still
                    # carries the file under `<field>generatepdf` as a JSON blob with a
                    # `path`. Same document, one indirection further in.
                    entry.pdf_url = _generated_pdf(row.get(f"{pkey}generatepdf"))

        try:
            payload = _fetch_json(client, _REPEALED_FEED, timeout)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Malaysian inventory: %s unavailable (%s: %s)",
                         _REPEALED_FEED, type(exc).__name__, exc)
        else:
            for row in payload.get("records") or []:
                act_no = str(row.get("ILA_ACT_NO") or "").strip()
                if not act_no:
                    continue
                entry = entries.setdefault(act_no, InventoryEntry(act_no=act_no))
                entry.feeds.append(_REPEALED_FEED)
                entry.title_en = entry.title_en or _text(row.get("TITLEBI"))
                entry.repealed_by = _text(row.get("REPEALEDBY"))
                entry.repealed_by_title = _text(row.get("REPEALTITLEBI"))
    finally:
        if owns:
            client.close()

    _LOG.info("Malaysian inventory: %d Act(s), %d with an English PDF, %d repealed",
              len(entries), sum(1 for e in entries.values() if e.pdf_url),
              sum(1 for e in entries.values() if e.repealed))
    with _CACHE_LOCK:
        _CACHE = entries
    return entries


def reset_cache() -> None:
    """Drop the process cache (tests, and a run that wants the feeds re-read)."""
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None
