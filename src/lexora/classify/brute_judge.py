"""Brute CANDIDACY judge: one LLM call decides which of ALL 9 indicators a law may touch.

Why this exists. The AU portal has no concept search, so the discovery layer enumerates
the principal-Act catalogue and must cheaply decide which Acts are worth examining. This
judge runs over an Act's TABLE OF CONTENTS (its ``/latest/text`` shell — section titles,
concentrated and high-signal) and returns the indicators the Act plausibly touches; the
caller (:mod:`lexora.collect.au_enumerate`) keeps any Act flagged for >=1 indicator. It is
a RECALL gate only — the precise per-indicator attribution is decided later, clause by
clause, by the per-clause 9-in-1 verifier (``classify.verifier`` ``mode="per_clause"``),
because a single 9-in-1 call over a whole Act's FULL TEXT is a noisy skim that buries
sectoral provisions (measured) whereas section titles judge candidacy well (validated 8/9).

The legacy ``subset`` path (full-text 9-in-1, opt-in ``LEXORA_BRUTE_JUDGE``) remains for
back-compat but is inert by default; the per-clause verifier supersedes it for relevance.

Model choice (validated 2026-06-29). The default backend ``gpt-5.4`` is a reasoning
model that returns EMPTY content ~45-55% of the time (independent of concurrency,
unfixable by prompt/params), so a single 9-in-1 call is unreliable there. A
NON-reasoning model — ``DeepSeek-V4-Flash`` — returns valid JSON ~100% first try, has a
~1M-token context (a whole Act fits, no chunking) and matched/beat gpt's recall. Hence
``LEXORA_BRUTE_MODEL`` defaults to it, independent of ``LEXORA_LLM_MODEL``. The id is
CASE-SENSITIVE at the gateway; see ``DEFAULT_BRUTE_MODEL``.

Prompt is RECALL-oriented (candidate shortlisting, not a final strict verdict): a strict
"is this relevant" framing dropped 0.5/boundary cases (e.g. a general-law computer-offence
provision for 7.2). A POLARITY note handles the "Lack of <framework>" indicators
(P7-I1/P7-I2) — a law that PROVIDES the framework IS the evidence. deepseek is
non-deterministic even at temperature 0, so we UNION ``passes`` samples (temp 0.0 then
0.4) to stabilise recall.

Inert by default: ``make_brute_judge`` returns None when disabled or no API key, so the
pipeline keeps its existing discovery-attribution behaviour.
"""
from __future__ import annotations

import json
import logging
import os
import re

import httpx

from lexora.models.indicator import RDTIIIndicator

_LOG = logging.getLogger(__name__)

# EXACTLY as the endpoint spells it. Measured 2026-08-01: the gateway matches model ids
# CASE-SENSITIVELY and answers a lowercase "deepseek-v4-flash" with HTTP 400, "There are
# no healthy deployments for this model" -- so this feature could not start at all, and
# it failed silently, because `_one` swallows the error and the caller reads an empty
# result as "the judge found nothing" rather than "the judge never ran". `/v1/models`
# lists the deployed ids; check there before changing this string.
DEFAULT_BRUTE_MODEL = "DeepSeek-V4-Flash"


def brute_enabled() -> bool:
    """True when ``LEXORA_BRUTE_JUDGE`` is set truthy."""
    return os.environ.get("LEXORA_BRUTE_JUDGE", "").lower() in ("1", "true", "yes", "on")


def _indicator_block(indicators: list[RDTIIIndicator]) -> str:
    parts = []
    for i in indicators:
        b = f"- {i.submission_id} ({i.name})\n  Definition: {i.description}"
        if i.long_definition:
            b += ("\n  Official long definition (use its scope and boundaries to pick the "
                  f"CORRECT indicator, NOT to exclude an on-topic law): {i.long_definition}")
        if i.scoring_criteria:
            b += ("\n  Scoring (partial / sector-specific / non-dedicated 0.5 cases still "
                  f"COUNT as relevant evidence): {i.scoring_criteria}")
        parts.append(b)
    return "\n".join(parts)


def _system(indicators: list[RDTIIIndicator]) -> str:
    ids = [i.submission_id for i in indicators]
    shape = '{"verdicts":[' + ",".join(
        f'{{"indicator_id":"{s}","relevant":false,"evidence":"","confidence":0.9}}' for s in ids
    ) + "]}"
    return (
        "You are building a CANDIDATE shortlist for the UN ESCAP RDTII from a law's TABLE OF "
        "CONTENTS (its section titles). Decide for EACH indicator whether the law plausibly "
        "contains a provision that could be CITED AS EVIDENCE for that indicator — INCLUDING "
        "partial, sector-specific, non-dedicated, or weak (0.5-score) cases. This is a RECALL "
        "step deciding only whether the law is worth examining clause-by-clause later, so when a "
        "title is arguably on-topic, mark relevant=true. Use the long definition to pick the "
        "CORRECT indicator (its boundaries separate look-alike indicators), NOT to exclude a "
        "borderline-but-on-topic law. Mark relevant=false only when the law has nothing on that "
        "topic.\n"
        "POLARITY NOTE: indicators phrased as 'Lack of <framework>' (P7-I1 comprehensive "
        "data-protection framework; P7-I2 dedicated cybersecurity framework) are assessed by "
        "EXAMINING the framework laws — a law that PROVIDES or CONTRIBUTES to such a framework IS "
        "relevant evidence (mark true); do not mark it irrelevant merely because it supplies rather "
        "than lacks the framework.\n"
        'Output ONE JSON object only, no markdown or prose. Key "verdicts": an array of EXACTLY '
        f"{len(ids)} objects, one per indicator, each "
        '{"indicator_id","relevant","evidence","confidence"}; evidence = a verbatim substring of '
        f"the text or empty.\nEXACT SHAPE (values illustrative):\n{shape}"
    )


class BruteJudge:
    """Full-text 9-in-1 relevance judge over a non-reasoning LLM."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = DEFAULT_BRUTE_MODEL,
        user_agent: str = "Mozilla/5.0",
        passes: int = 2,
        max_chars: int = 900_000,
        timeout: float = 240.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.user_agent = user_agent
        self.passes = max(1, passes)
        self.max_chars = max_chars
        self.timeout = timeout
        self.calls = 0
        self.total_tokens = 0
        self.error_count = 0
        self.last_error: str = ""

    def _report(self, detail: str) -> None:
        """Record a backend failure, and log the first one so a dead judge is visible."""
        self.error_count += 1
        self.last_error = detail
        if self.error_count == 1:
            _LOG.error("brute judge backend refused model %r: %s -- every candidate will "
                       "fall back to its existing attribution", self.model, detail)

    def _one(self, system: str, text: str, temperature: float) -> set[str] | None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"LAW TEXT:\n{text[: self.max_chars]}"},
            ],
            "temperature": temperature,
            "max_completion_tokens": 4000,
        }
        try:
            r = httpx.post(self.base_url + "/chat/completions", headers=headers, json=body, timeout=self.timeout)
            self.calls += 1
            if r.status_code != 200:
                # Say it ONCE. Returning None here is correct -- the caller degrades to
                # its existing attribution -- but a judge that never ran and a judge that
                # found nothing produce the same empty result, and on 2026-08-01 a single
                # wrong character in the model id made every call a 400 with nothing in
                # the log to say so.
                self._report(f"HTTP {r.status_code}: {r.text[:160]}")
                return None
            j = r.json()
            self.total_tokens += int((j.get("usage") or {}).get("total_tokens", 0) or 0)
            content = (j["choices"][0]["message"].get("content") or "").strip()
            data = _parse(content)
            if data is None:
                return None
            return {
                v["indicator_id"]
                for v in data.get("verdicts", [])
                if isinstance(v, dict) and v.get("relevant")
            }
        except Exception:
            return None

    def relevant(self, text: str, indicators: list[RDTIIIndicator]) -> set[str]:
        """Union of relevant submission_ids across ``passes`` samples.

        Returns an empty set when the text is too short to judge or every pass
        failed — the caller then keeps its existing attribution (degrade, never
        invent). The system prompt embeds the indicators, so the user message is
        just the law text (the whole Act fits in the model's ~1M-token window)."""
        if not text or len(text) < 400:
            return set()
        system = _system(indicators)
        union: set[str] = set()
        got = False
        for p in range(self.passes):
            rel = self._one(system, text, 0.0 if p == 0 else 0.4)
            if rel is not None:
                got = True
                union |= rel
        return union if got else set()

    def subset(self, text: str, indicators: list[RDTIIIndicator]) -> list[RDTIIIndicator] | None:
        """The indicators the law is relevant to (regime-2). None on total failure
        so the caller can fall back to its discovery-attribution subset."""
        rel = self.relevant(text, indicators)
        if not rel:
            return None
        return [i for i in indicators if i.submission_id in rel]


def _parse(content: str) -> dict | None:
    """Parse a JSON object from a model reply: plain, ```json fenced, or with
    surrounding prose. Returns None when nothing parseable is present."""
    if not content:
        return None
    s = content.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def make_brute_judge(*, enabled: bool | None = None, model: str | None = None) -> BruteJudge | None:
    """Construct a :class:`BruteJudge`, or None when disabled / unconfigured.

    Reads endpoint + key + UA from the shared project config (``load_config`` —
    process env AND the local ``.env``, same source as ``LlmClient``), so the judge
    is configured wherever the rest of the LLM stack is; reading raw ``os.environ``
    would silently no-op when creds live only in ``.env``. The model is a SEPARATE
    ``LEXORA_BRUTE_MODEL`` (default ``deepseek-v4-flash``) so the brute judge stays
    on a reliable non-reasoning model even when ``LEXORA_LLM_MODEL`` points at a
    reasoning backend."""
    if enabled is None:
        enabled = brute_enabled()
    if not enabled:
        return None
    from lexora.config import env_value, load_config

    cfg = load_config()
    base = cfg.llm_base_url
    key = cfg.llm_api_key
    if not base or not key:
        return None
    return BruteJudge(
        base_url=base,
        api_key=key,
        # Same rule as the per-clause judge: an unset override means "whatever
        # LEXORA_LLM_MODEL says", so a config-only swap to a self-hosted model
        # reaches this lane too instead of asking that server for a vendor name.
        model=model or env_value("LEXORA_BRUTE_MODEL", "") or cfg.llm_model,
        user_agent=cfg.llm_user_agent or "Mozilla/5.0",
        passes=int(env_value("LEXORA_BRUTE_PASSES", "2")),
    )


__all__ = ["BruteJudge", "make_brute_judge", "brute_enabled", "DEFAULT_BRUTE_MODEL"]
