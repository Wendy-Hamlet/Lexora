"""Persistent cache for per-clause relevance verdicts.

The 9-in-1 judge is asked, for one clause and the full indicator catalogue, which
indicators that clause supports. The answer is a pure function of three things: the clause
text, the indicator definitions, and the model. Nothing else in the pipeline feeds it.

So re-running the pipeline over the same corpus asks the model the *identical* question
tens of thousands of times and pays for it again. The 2026-07-14 run made 22,445 judge
calls; a re-run to validate a retrieval tweak, or a rehearsal for the live pitch, repeats
every one of them. This cache makes the second run of an unchanged question free.

WHAT IT IS NOT: it is not a way to ship a stale verdict. The key is a hash over the model
id, the full indicator catalogue (every long definition, verbatim), and the clause text. If
a definition is edited, a boundary rule reworded, the model swapped, or the parser emits a
clause with one character different, the key changes and the model is asked again. There is
no path by which a cached answer outlives the question that produced it.

Failures are never cached. A backend error yields ``None`` (the caller drops the clause);
storing that would turn one outage into a permanently missing citation.

Set ``LEXORA_JUDGE_CACHE=0`` to bypass entirely -- e.g. to demonstrate the true cold cost of
a run, or to reproduce a number from scratch.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from lexora.models.clause import Clause
    from lexora.models.indicator import RDTIIIndicator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verdicts (
    key        TEXT PRIMARY KEY,
    indicators TEXT NOT NULL,
    model      TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (julianday('now'))
);
"""


def cache_enabled() -> bool:
    return os.environ.get("LEXORA_JUDGE_CACHE", "1").lower() not in (
        "0", "false", "no", "off",
    )


def default_path() -> Path:
    p = os.environ.get("LEXORA_JUDGE_CACHE_PATH")
    if p:
        return Path(p)
    return Path(__file__).resolve().parents[3] / "data" / "cache" / "judge.sqlite"


def catalogue_fingerprint(indicators: list[RDTIIIndicator]) -> str:
    """Hash the exact indicator text the judge is shown.

    Everything the prompt puts in front of the model, in prompt order. Editing a long
    definition -- the boundary rules that decide 6.4-vs-6.1 -- must invalidate every verdict
    taken under the old wording, so the definitions go in verbatim rather than by id.
    """
    h = hashlib.sha256()
    for i in indicators:
        h.update(i.submission_id.encode())
        h.update(b"\0")
        h.update((i.name or "").encode())
        h.update(b"\0")
        h.update((i.description or "").encode())
        h.update(b"\0")
        h.update((getattr(i, "long_definition", "") or "").encode())
        h.update(b"\x01")
    return h.hexdigest()


class JudgeCache:
    """SQLite-backed verdict store, safe under the judge's thread pool."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._lock = threading.Lock()
        # check_same_thread=False + our own lock: the judge runs on a ThreadPoolExecutor,
        # and sqlite3 connections are not thread-affine once we serialise access ourselves.
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    @staticmethod
    def key(model: str, fingerprint: str, clause: Clause) -> str:
        h = hashlib.sha256()
        h.update(model.encode())
        h.update(b"\0")
        h.update(fingerprint.encode())
        h.update(b"\0")
        # The clause text as the model sees it. Not the clause_id: ids are positional and
        # would collide across documents, and a re-parse can renumber them while the text
        # is unchanged.
        h.update(clause.span.text.encode())
        return h.hexdigest()

    def get(self, key: str) -> set[str] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT indicators FROM verdicts WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return set(json.loads(row[0]))

    def put(self, key: str, model: str, verdict: set[str]) -> None:
        """Store a verdict. ``None`` verdicts (backend failures) must never get here."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO verdicts (key, indicators, model) VALUES (?, ?, ?)",
                (key, json.dumps(sorted(verdict)), model),
            )
            self._db.commit()
            self.writes += 1

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def close(self) -> None:
        with self._lock:
            # Fold the WAL back into the main file so a run leaves one artifact, not three.
            with contextlib.suppress(sqlite3.Error):
                self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._db.close()

    def __enter__(self) -> JudgeCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
