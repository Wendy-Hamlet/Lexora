"""Enumerate Malaysia's statute book and shortlist it, once, into a reusable cache.

Australia already works this way: enumerate every principal Act, judge each on cheap
evidence, and append the verdicts to a JSONL so the next run costs nothing. Malaysia
could not, because its discovery went through a search proxy that answers roughly three
requests in five and, on 20 July, silently stopped filtering at all. The portal's own
listings (:mod:`lexora.collect.my_inventory`) removed that dependency: 1,287 Acts with
English titles and direct PDFs in four requests.

What the listings do NOT carry is subject metadata, so relevance has to be decided here.
Two stages, because one is not enough and three is not affordable (measured 2026-08-01):

**Stage 1 — the title.** 1,287 Acts, one call each, ~12 min at 16 workers, 98% of the
prompt served from the model's cache. A keyword filter cannot do this job: over the same
titles it reached 5 of the 8 Malaysian gold statutes and scored CYBER SECURITY ACT 2024
exactly 0.00, because a title shares no tokens with an indicator's phrasing. A model
knows what an Act of that name contains. Shortlist: 249 of 1,287.

**Stage 2 — the table of contents.** Stage 1 flags 220 of those 249 for P7-I3 (retention)
and P7-I5 (government access) alone, and it is right to: most Malaysian Acts *could* hold
a record-keeping or an authorised-officer clause. Only the contents can tell "could" from
"does" -- "47. Access to computerized data" is a heading, not a guess. Scanned Acts are
OCR'd, first twelve pages only: the ARRANGEMENT OF SECTIONS is at the front, and a full
OCR of one Malaysian scan has cost this project hours.

An Act we could not read is KEPT, always. Unreadable is not irrelevant, and a filter that
silently drops what it failed to open is worse than no filter.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from lexora.collect.discovery import TAG_KNOWN, TAG_NEW, DiscoveryResult
from lexora.models.source import InstrumentStatus, SourceType

_LOG = logging.getLogger(__name__)

# Indicators whose evidence is a dedicated framework: a title that plausibly names one is
# worth mapping without a second look. The rest (P7-I3 retention, P7-I5 access) are
# horizontal duties that almost any statute might carry, which is what stage 2 is for.
FRAMEWORK_INDICATORS = frozenset({"P6-I1", "P6-I2", "P6-I3", "P6-I4", "P7-I1", "P7-I2", "P7-I4"})

OCR_PAGES = frozenset(range(1, 13))
TOC_CHARS = 12_000
SIZE_CAP = 90_000_000
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://lom.agc.gov.my/"}
_ACT_DETAIL = "https://lom.agc.gov.my/act-detail.php"
# Judged verdicts for the whole Malaysian statute book, committed to the repository.
SHIPPED_VERDICTS = Path(__file__).resolve().parents[3] / "data" / "reference" / "my_enum_verdicts.jsonl"


def enumerate_enabled() -> bool:
    """True when ``LEXORA_MY_ENUMERATE`` is set truthy."""
    return os.environ.get("LEXORA_MY_ENUMERATE", "").lower() in ("1", "true", "yes", "on")


def _load_jsonl(path: Path) -> dict[str, dict]:
    """Load a resumable JSONL keyed by ``act`` (skips malformed lines)."""
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("act"):
                out[obj["act"]] = obj
    return out


# Function words a legible page of English statute always carries. Measured on five
# Malaysian scans whose OCR is plainly readable: 0.30-0.38 of tokens. The floor is set
# far below that because its job is to catch noise, not to grade quality -- and note
# that RapidOCR's own page confidence on those same five ran 0.11-0.54, so confidence
# is not a usable proxy for legibility here.
_FUNCTION_WORDS = frozenset({
    "the", "of", "and", "to", "in", "for", "any", "this", "act", "section", "shall",
    "be", "or", "with", "by", "person", "may", "not", "under", "such", "other",
    "than", "which", "is", "are", "that", "from", "on", "at", "into",
})
_LEGIBLE_FLOOR = 0.10


def _is_legible(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z]{2,}", text.lower())
    if len(tokens) < 120:
        return False
    return sum(1 for t in tokens if t in _FUNCTION_WORDS) / len(tokens) >= _LEGIBLE_FLOOR


def _toc_from_pdf(url: str, *, timeout: float, ocr_engine=None) -> tuple[str, str]:
    """(table-of-contents text, reason it is empty). Never raises."""
    from lexora.extract.pdf_text_extractor import assemble_global_text, extract_pdf_bytes

    if not url:
        return "", "no-pdf-link"
    try:
        with (httpx.Client(follow_redirects=True, timeout=timeout,
                           headers=_HEADERS) as client,
              client.stream("GET", url) as response):
            if response.status_code != 200:
                return "", f"http-{response.status_code}"
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                chunks.append(chunk)
                size += len(chunk)
                if size > SIZE_CAP:
                    return "", "over-size-cap"
            raw = b"".join(chunks)
    except Exception as exc:  # noqa: BLE001 — an unfetchable Act is kept, not raised
        return "", type(exc).__name__
    if raw[:4] != b"%PDF":
        return "", "not-a-pdf"
    try:
        pages = extract_pdf_bytes(raw)
        text = assemble_global_text(pages[:14])
    except Exception as exc:  # noqa: BLE001
        return "", f"parse-{type(exc).__name__}"

    if len(text.strip()) < 400 and ocr_engine is not None:
        # Image-only. The front matter is what we need, so OCR twelve pages, not the Act.
        try:
            from lexora.extract.ocr_extractor import extract_ocr

            text = "\n".join(p.text for p in extract_ocr(
                raw, page_numbers=set(OCR_PAGES), engine=ocr_engine) if p.text)
        except Exception as exc:  # noqa: BLE001
            return "", f"ocr-{type(exc).__name__}"
    if len(text.strip()) < 300:
        return "", "no-text-layer"
    if not _is_legible(text):
        # OCR that came out as noise must NOT reach the judge. A judge shown noise
        # answers "nothing relevant here", which is indistinguishable from a real
        # verdict and would delete the Act -- the one outcome this module refuses.
        return "", "ocr-illegible"

    # OCR drops spaces inside headings, so locate the anchor on a space-stripped copy.
    flat = text.upper().replace(" ", "")
    at = flat.find("ARRANGEMENTOFSECTIONS")
    return (text[max(0, at - 200): at + TOC_CHARS] if at >= 0 else text[:TOC_CHARS]), ""


def enumerate_my_candidates(
    indicators: list,
    *,
    title_judge,
    toc_judge=None,
    source_type: SourceType = SourceType.primary,
    known_instruments: list[str] | None = None,
    known_instrument_ids: dict[str, str] | None = None,
    verdict_cache: str | None = None,
    judge_workers: int = 16,
    fetch_workers: int = 6,
    timeout: float = 180.0,
    ocr: bool = True,
    client: httpx.Client | None = None,
    log=None,
) -> list[DiscoveryResult]:
    """Every Act the two stages judged relevant, as discovery results.

    ``title_judge`` and ``toc_judge`` are callables ``(text, indicators) -> set[str]``
    of submission ids — a :class:`~lexora.classify.brute_judge.BruteJudge` satisfies
    both. Verdicts are appended to ``verdict_cache`` (JSONL, one line per Act), so a
    second run over the same statute book costs nothing and a interrupted run resumes.
    """
    from lexora.collect.my_inventory import fetch_inventory

    emit = log if log is not None else _LOG.info
    inventory = fetch_inventory(client, timeout=timeout)
    emit(f"enumerate MY: {len(inventory)} Act(s) in the portal's listings")

    # Two files, one read order. `data/reference/` ships WITH the repository, so a fresh
    # clone inherits the statute book already judged and a re-derivation of our numbers
    # costs an auditor nothing — the claim we make about reproducibility has to be true
    # for someone who is not us. `outputs/cache/` is this machine's writable copy and
    # wins on conflict, because a re-judged Act is a newer answer than a shipped one.
    path = Path(verdict_cache) if verdict_cache else (
        Path("outputs") / "cache" / "my_enum_verdicts.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    verdicts = _load_jsonl(SHIPPED_VERDICTS)
    if verdicts:
        emit(f"enumerate MY: {len(verdicts)} verdict(s) from the shipped reference set")
    verdicts.update(_load_jsonl(path))
    pending = [a for a in inventory if a not in verdicts]
    emit(f"enumerate MY: {len(verdicts)} cached, {len(pending)} to judge")

    lock = threading.Lock()
    if pending:
        def _stage1(act: str) -> dict:
            entry = inventory[act]
            prompt = f"LAW TITLE: {entry.title_en}\nAct number: {act}"
            hits = sorted(title_judge(prompt, indicators) or ())
            return {"act": act, "title": entry.title_en, "title_hits": hits,
                    "toc_hits": None, "reason": ""}

        t0 = time.perf_counter()
        staged: list[dict] = []
        with ThreadPoolExecutor(max_workers=max(1, judge_workers)) as ex:
            for n, rec in enumerate(ex.map(_stage1, pending), 1):
                staged.append(rec)
                if n % 200 == 0:
                    emit(f"enumerate MY: stage 1 judged {n}/{len(pending)} "
                         f"({time.perf_counter() - t0:.0f}s)")

        # Stage 2 only where stage 1 found nothing but horizontal duties.
        second = [r for r in staged
                  if r["title_hits"] and not (FRAMEWORK_INDICATORS & set(r["title_hits"]))]
        emit(f"enumerate MY: stage 2 over {len(second)} horizontal-only candidate(s)")
        engine = None
        if ocr and second and toc_judge is not None:
            try:
                from lexora.extract.ocr_extractor import make_engine

                engine = make_engine(use_gpu=True)
            except Exception as exc:  # noqa: BLE001 — scans then stay unread, and kept
                emit(f"enumerate MY: OCR unavailable ({type(exc).__name__}: {exc})")
        ocr_lock = threading.Lock()

        def _stage2(rec: dict) -> dict:
            entry = inventory[rec["act"]]
            if engine is not None:
                with ocr_lock:
                    toc, why = _toc_from_pdf(entry.pdf_url, timeout=timeout, ocr_engine=engine)
            else:
                toc, why = _toc_from_pdf(entry.pdf_url, timeout=timeout)
            if why:
                rec["reason"] = why          # unreadable -> keep the stage-1 verdict
                return rec
            rec["toc_hits"] = sorted(toc_judge(toc, indicators) or ())
            return rec

        if second and toc_judge is not None:
            with ThreadPoolExecutor(max_workers=max(1, fetch_workers)) as ex:
                list(ex.map(_stage2, second))

        with lock, path.open("a", encoding="utf-8") as fh:
            for rec in staged:
                verdicts[rec["act"]] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    known_ids = known_instrument_ids or {}
    results: list[DiscoveryResult] = []
    for act, rec in verdicts.items():
        entry = inventory.get(act)
        if entry is None:
            continue
        # The contents overrule the title when we managed to read them; when we did not,
        # the title stands. An empty toc_hits is a real verdict: the headings show nothing.
        hits = rec["toc_hits"] if rec.get("toc_hits") is not None else rec.get("title_hits")
        if not hits:
            continue
        if entry.repealed:
            status = InstrumentStatus.repealed.value
        elif entry.in_force is False:
            status = InstrumentStatus.draft.value
        elif entry.in_force is True:
            status = InstrumentStatus.in_force.value
        else:
            status = InstrumentStatus.unknown.value
        results.append(DiscoveryResult(
            url=f"{_ACT_DETAIL}?act={act}&lang=BI",
            title=entry.title_en or rec.get("title", ""),
            source_type=source_type,
            score=1.0,
            via="api",
            is_pdf_link=False,
            discovery_tag=TAG_KNOWN if act in known_ids else TAG_NEW,
            matched_instrument=known_ids.get(act),
            fulltext_url=entry.pdf_url,
            law_number=f"Act {act}",
            status=status,
            indicator_hits=list(hits),
        ))
    emit(f"enumerate MY: {len(results)} candidate(s) of {len(inventory)} Acts")
    return results
