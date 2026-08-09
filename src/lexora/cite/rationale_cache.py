"""Persistent cache for LLM-authored Mapping Rationales.

The rationale is a pure function of three things: the model, the system prompt, and the
rendered user prompt (indicator definition, jurisdiction, locator, provision text). Nothing
else feeds it. So a re-run over an unchanged corpus asks the model the identical question
once per citation and pays for it again.

WHY THIS EXISTS. The judge has had :mod:`lexora.classify.judge_cache` since July, so a
replay serves 8019/8019 clause verdicts for free. The rationale generator had nothing, so a
full Malaysia run cost 2,545 s with ``--rationale-llm`` against 554 s without. The flag was
therefore dropped from every full-economy command for speed -- and on 2026-08-03 a full
Singapore run was demonstrated to judges whose Mapping Rationale column was 181/181
deterministic template, strictly worse than the submitted file they were already holding.
That chain starts here. With the rationale cached the way the judge is cached, a run with
the layer ON replays for free and there is no longer any reason to drop the flag.

WHAT IS STORED IS THE MODEL'S RAW ANSWER, NOT OUR ACCEPTED ONE. ``RationaleGenerator``
rejects a returned rationale that is empty, over-length, talks about scoring, or copies a
6-word run from the provision, and falls back to the template. Those guards are *ours*, and
tuning them is open work: 29.1% of the Round-1 rationales were template with the layer on
and we still cannot say how much of that is the copy check misfiring on formulaic legal
prose. If the cache stored the post-guard result, changing a guard would leave every cached
row untouched and the experiment would silently measure nothing. Storing the raw response
means the guards re-run on every read, so a guard change can be evaluated over the whole
corpus for free -- which is the point of having the corpus.

The key covers the model, the system prompt and the rendered user prompt. Reword the system
prompt, add a field to the user prompt, change the provision text by one character, or swap
the model, and the key changes and the model is asked again. There is no path by which a
cached rationale outlives the question that produced it. Note the consequence: the first run
after any prompt edit pays full price again. That is correct, not a defect.

Failures are never cached -- a backend error must not become a permanent template row. An
empty rationale IS cached: it is a real answer to a well-formed question, and the guards
will keep turning it into a template row for free.

Set ``LEXORA_RATIONALE_CACHE=0`` to bypass entirely; ``LEXORA_RATIONALE_CACHE_PATH`` moves
the store.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rationales (
    key         TEXT PRIMARY KEY,
    response    TEXT NOT NULL,
    model       TEXT NOT NULL,
    fingerprint TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL DEFAULT (julianday('now'))
);
"""


def cache_enabled() -> bool:
    return os.environ.get("LEXORA_RATIONALE_CACHE", "1").lower() not in (
        "0", "false", "no", "off",
    )


def default_path() -> Path:
    p = os.environ.get("LEXORA_RATIONALE_CACHE_PATH")
    if p:
        return Path(p)
    return Path(__file__).resolve().parents[3] / "data" / "cache" / "rationale.sqlite"


def system_fingerprint(system: str) -> str:
    """Hash the stable half of the question.

    Unlike the judge -- whose user prompt is 95% the same indicator catalogue every call,
    so it needs a sentinel to separate catalogue from clause -- the rationale's user prompt
    varies in full per (clause x indicator). There is nothing to separate: the whole user
    prompt goes into the key. What this fingerprint buys is :meth:`RationaleCache.coverage`,
    which answers "how many stored rationales were written under the prompt I am about to
    send" *before* a run rather than after it.
    """
    return hashlib.sha256(system.encode()).hexdigest()


class RationaleCache:
    """SQLite-backed store of raw model responses, safe under the rationale thread pool."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._lock = threading.Lock()
        # check_same_thread=False + our own lock: `pipeline._execute_specs` runs rationale
        # generation on a ThreadPoolExecutor whenever llm_workers > 1.
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    @staticmethod
    def key(model: str, system: str, user: str) -> str:
        h = hashlib.sha256()
        h.update(model.encode())
        h.update(b"\0")
        h.update(system.encode())
        h.update(b"\0")
        h.update(user.encode())
        return h.hexdigest()

    def get(self, key: str) -> dict | None:
        # Counters inside the lock: `self.hits += 1` is a read-modify-write, and up to
        # llm_workers threads run this. An undercounted hit rate is exactly the number that
        # would let a broken cache pass for a cheap run.
        with self._lock:
            row = self._db.execute(
                "SELECT response FROM rationales WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
        try:
            data = json.loads(row[0])
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def put(self, key: str, model: str, response: dict, fingerprint: str = "") -> None:
        """Store one RAW model response. Backend failures must never reach here."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO rationales (key, response, model, fingerprint) "
                "VALUES (?, ?, ?, ?)",
                (key, json.dumps(response, sort_keys=True), model, fingerprint),
            )
            self._db.commit()
            self.writes += 1

    def coverage(self, model: str, fingerprint: str) -> tuple[int, int]:
        """``(rationales answering THIS system prompt, rationales stored in total)``."""
        with self._lock:
            live = self._db.execute(
                "SELECT COUNT(*) FROM rationales WHERE model = ? AND fingerprint = ?",
                (model, fingerprint),
            ).fetchone()[0]
            total = self._db.execute("SELECT COUNT(*) FROM rationales").fetchone()[0]
        return int(live), int(total)

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def close(self) -> None:
        with self._lock:
            with contextlib.suppress(sqlite3.Error):
                self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._db.close()

    def __enter__(self) -> RationaleCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = [
    "RationaleCache",
    "cache_enabled",
    "default_path",
    "system_fingerprint",
]
