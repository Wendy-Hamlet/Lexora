"""AU full-catalogue enumeration as a discovery source (full-text relevance gate).

Why this exists. Australia's Federal Register has NO concept search — its OData
``$search`` is a no-op, so name/known-instrument queries (the only thing the
portal answers) can never surface a sectoral or surveillance statute whose RDTII
relevance is one buried clause (government-access powers, data-retention duties).
That capped discovery recall. The fix is to enumerate the in-force PRINCIPAL Acts
and decide relevance with the regime-2 LLM judge, so the working set is built from
the corpus instead of name look-ups. NOTE this is the discovery SOURCE — distinct
from ``classify/brute_judge.py``, the per-document judge it reuses.

Two stages, to stay affordable:

1. **Cheap shell prefilter (gate).** The ``/{id}/latest/text`` SPA shell is a
   table of contents (section *titles*, no prose). A prior sweep judged all 1k+
   principals on that cheap text; its verdicts (a resumable JSONL) gate the corpus
   down to the ~120 plausible candidates. Section titles are enough to EXCLUDE the
   clearly-irrelevant majority. When no prefilter is supplied the gate is the whole
   principal set.

2. **High-quality full-text judge (this module's core).** Each surviving candidate
   is fetched to its EPUB — the single static XHTML holding the whole compilation's
   operative PROSE, the exact text the structure parser turns into clauses — and
   the regime-2 judge runs on that prose (NOT the shell titles). This is the
   quality upgrade: the candidacy decision and the per-indicator attribution come
   from real provisions, so the mapper can then use that attribution directly
   (regime-1) without re-judging the same prose.

The AU CloudFront edge has a cumulative per-IP limit, so full-text FETCH is SERIAL
with a small interval (and rejects the ~555-char throttle shell); the LLM JUDGE,
which is the slow part, runs in parallel over the fetched texts.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from lexora.collect import http_cache
from lexora.collect.discovery import TAG_KNOWN, TAG_NEW, DiscoveryResult, _fuzzy_known
from lexora.collect.strategies import _AU_DOC, au_act_catalogue
from lexora.models.source import SourceType

_LOG = logging.getLogger(__name__)
_THROTTLE = "request could not be satisfied"


def enumerate_enabled() -> bool:
    """True when ``LEXORA_AU_ENUMERATE`` is set truthy."""
    return os.environ.get("LEXORA_AU_ENUMERATE", "").lower() in ("1", "true", "yes", "on")


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _shell_text(tid: str, *, timeout: float, tries: int, user_agent: str) -> str:
    """Fetch a principal Act's ``/latest/text`` shell and return tag-stripped text.

    The shell is the Act's TABLE OF CONTENTS — its section TITLES, concentrated and
    high-signal. The candidacy judge runs on this (validated 8/9 on AU): a full-text
    9-in-1 skim over the whole Act is noisier and buries sectoral provisions, so the
    relevance decision is deferred to the per-clause judge in the mapping stage; here
    the titles only decide which Acts enter the working set. Returns "" on a throttle
    shell / non-200 / error (retried with backoff)."""
    headers = {"User-Agent": user_agent}
    for attempt in range(tries):
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as c:
                r = c.get(f"https://www.legislation.gov.au/{tid}/latest/text")
            if r.status_code == 200 and len(r.text) > 2000 and _THROTTLE not in r.text.lower():
                return _WS_RE.sub(" ", _TAG_RE.sub(" ", r.text)).strip()
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1) + random.random())
    return ""


def _load_jsonl(path: Path) -> dict[str, dict]:
    """Load a resumable JSONL keyed by ``id`` (skips malformed lines)."""
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("id"):
                out[o["id"]] = o
    return out


def enumerate_au_candidates(
    indicators: list,
    *,
    judge,
    source_type: SourceType = SourceType.primary,
    known_instruments: list[str] | None = None,
    catalogue_cache: str | None = None,
    verdict_cache: str | None = None,
    judge_workers: int = 32,
    timeout: float = 90.0,
    fetch_tries: int = 5,
    min_interval: float = 0.3,
    user_agent: str = "Mozilla/5.0",
    log=None,
) -> list[DiscoveryResult]:
    """Enumerate in-force principal Acts and return the relevant candidates.

    ``judge`` is a :class:`~lexora.classify.brute_judge.BruteJudge`. Candidacy is
    decided on the cheap ``/latest/text`` SHELL (section TITLES — concentrated,
    high-signal, validated 8/9 on AU); the precise per-indicator attribution is
    deferred to the per-clause judge in the mapping stage. Each principal's shell is
    fetched SERIALLY (CloudFront cumulative per-IP limit) then judged in PARALLEL;
    verdicts are appended to ``verdict_cache`` (JSONL ``{id,name,relevant}``), so
    pointing it at a prior sweep reuses that work for free. A candidate is any Act
    the judge flagged for >=1 indicator; ``indicator_hits`` records which (advisory —
    the mapper re-decides per clause), KNOWN-tagged when the name matches a profile
    known instrument."""
    emit = log if log is not None else _LOG.info

    def _log(msg: str) -> None:
        emit(msg)

    catalogue = au_act_catalogue(cache_path=catalogue_cache, timeout=timeout)
    name_by_id = {v["id"]: v.get("name", "") for v in catalogue if v.get("isPrincipal")}
    _log(f"enumerate: {len(name_by_id)} principal Acts")

    verdict_path = Path(verdict_cache) if verdict_cache else (
        Path("outputs") / "cache" / "au_enum_verdicts.jsonl"
    )
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    verdicts = _load_jsonl(verdict_path)
    pending = [tid for tid in name_by_id if tid not in verdicts]
    _log(f"enumerate: {len(verdicts)} cached, {len(pending)} to judge")

    if pending:
        # Fetch shells SERIALLY (throttle-safe), then judge in PARALLEL (LLM is the
        # slow part). Shells are transient (verdicts are the durable cache).
        texts: dict[str, str] = {}
        t0 = time.perf_counter()
        for done, tid in enumerate(pending, 1):
            t = _shell_text(tid, timeout=timeout, tries=fetch_tries, user_agent=user_agent)
            if t:
                texts[tid] = t
            if done % 50 == 0:
                _log(f"enumerate: fetched {done}/{len(pending)} shells "
                     f"elapsed {time.perf_counter() - t0:.0f}s")
            if min_interval and not http_cache.serving_from_recording():
                time.sleep(min_interval)  # CloudFront cumulative per-IP limit

        def _judge_one(tid: str) -> dict:
            text = texts.get(tid, "")
            rel = sorted(judge.relevant(text, indicators)) if text else []
            return {"id": tid, "name": name_by_id.get(tid, ""), "relevant": rel}

        t1 = time.perf_counter()
        with verdict_path.open("a", encoding="utf-8") as vf, \
                ThreadPoolExecutor(max_workers=max(1, judge_workers)) as ex:
            for n, rec in enumerate(ex.map(_judge_one, pending), 1):
                verdicts[rec["id"]] = rec
                vf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if n % 50 == 0:
                    vf.flush()
                    _log(f"enumerate: judged {n}/{len(pending)} "
                         f"elapsed {time.perf_counter() - t1:.0f}s")

    known = known_instruments or []
    results: list[DiscoveryResult] = []
    for tid, nm in name_by_id.items():
        rel = (verdicts.get(tid) or {}).get("relevant") or []
        if not rel:
            continue
        fuzzy, matched = _fuzzy_known(nm, known) if known else (0.0, None)
        is_known = fuzzy >= 0.85
        results.append(
            DiscoveryResult(
                url=_AU_DOC.format(id=tid, point="latest"),
                title=nm,
                source_type=source_type,
                score=1.0 if is_known else 0.9,
                via="enumerate",
                is_pdf_link=False,
                discovery_tag=TAG_KNOWN if is_known else TAG_NEW,
                matched_instrument=matched if is_known else None,
                indicator_hits=sorted(rel),
                status="IN_FORCE",
            )
        )
    results.sort(key=lambda r: (r.discovery_tag == TAG_KNOWN, r.score), reverse=True)
    _log(f"enumerate: {len(results)} candidates flagged")
    return results


__all__ = ["enumerate_au_candidates", "enumerate_enabled"]
