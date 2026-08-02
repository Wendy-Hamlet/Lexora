"""Persistent cache for OCR page text.

OCR is the only expensive step of a run that the record/replay layer does not cover.
``LEXORA_HTTP_CACHE=replay`` intercepts the network, so a replayed run re-reads the same
recorded PDF bytes — and then re-OCRs them from scratch, every time, because OCR is local
computation and no socket is involved. Measured 2026-08-02: a full three-economy replay
took 1,249 s, and the Malaysian leg spent most of it re-recognising scans it had already
recognised twice that day, cuDNN crash and CPU retry included.

So this stores what the engine saw, keyed by the page it saw it on.

WHY THE KEY IGNORES THE ACCELERATOR. The key covers the document bytes, the page number,
the render DPI and the engine *family* (``rapidocr:1.4.4``) — deliberately NOT whether that
family ran on the GPU or the CPU. The engine's own docstring claimed GPU and CPU produce
identical text; on 2026-08-02 that turned out to be false in the way that matters. The
runtime GPU->CPU fallback landed, two Malaysian Acts re-parsed with slightly different
text, every clause on them hashed to a new key, and the judge cache -- which hashes the
rendered clause -- answered for none of them. A full-corpus replay degraded 40 documents to
the keyword lane. Nothing was wrong with either transcription; they simply disagreed, and
one of them had already been paid for.

Which accelerator answered is a property of the machine, not of the document. Folding it
into the key means a laptop that falls back to CPU mid-run can no longer reuse -- or
reproduce -- what the same laptop computed on the GPU an hour earlier. So the first
transcription of a page wins and stays won, and ``produced_by`` records which engine
actually made it, in the clear, for provenance.

WHAT STILL INVALIDATES A ROW: different document bytes, a different page, a different DPI,
or a different engine family (a model upgrade bumps the version in the name). Cleaning --
the running-head stripper and the offset reflow in ``ocr_fill_pages`` -- runs AFTER this
layer and is not cached, so a parser fix takes effect without touching the store.

Empty pages ARE cached: a scan of a blank page is a real and repeatable answer. Engine
*failures* are not -- they raise, and nothing gets here.

``LEXORA_OCR_CACHE=0`` bypasses the store entirely, e.g. to time a cold run or to check
that a page still reads the same way.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    key         TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    confidence  REAL NOT NULL,
    produced_by TEXT NOT NULL,
    created_at  REAL NOT NULL DEFAULT (julianday('now'))
);
"""


def cache_enabled() -> bool:
    return os.environ.get("LEXORA_OCR_CACHE", "1").lower() not in (
        "0", "false", "no", "off",
    )


def default_path() -> Path:
    p = os.environ.get("LEXORA_OCR_CACHE_PATH")
    if p:
        return Path(p)
    return Path(__file__).resolve().parents[3] / "data" / "cache" / "ocr.sqlite"


def engine_family(name: str) -> str:
    """``rapidocr:1.4.4+cuda`` -> ``rapidocr:1.4.4``.

    The suffix reports which providers the ONNX sessions actually got, which is exactly
    what must NOT reach the key -- see the module docstring. Stripping it here, in one
    named function, is also what makes the intent testable.
    """
    return name.split("+", 1)[0]


def document_digest(source: Path | str | bytes) -> str:
    """SHA-256 of the PDF bytes, whether they arrived as bytes or as a path.

    Hashing the CONTENT and not the path or URL is what lets the same Act cached from a
    live fetch answer a replayed one: the recording serves back the identical bytes under
    a different temporary file name.
    """
    h = hashlib.sha256()
    if isinstance(source, bytes | bytearray):
        h.update(bytes(source))
        return h.hexdigest()
    with Path(source).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class OcrCache:
    """SQLite-backed page store, safe under the document thread pool."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    @staticmethod
    def key(doc_sha: str, page_number: int, dpi: int, engine: str) -> str:
        h = hashlib.sha256()
        h.update(doc_sha.encode())
        h.update(b"\0")
        h.update(str(page_number).encode())
        h.update(b"\0")
        h.update(str(dpi).encode())
        h.update(b"\0")
        h.update(engine_family(engine).encode())
        return h.hexdigest()

    def get(self, key: str) -> tuple[str, float, str] | None:
        """``(text, confidence, produced_by)`` or ``None``."""
        with self._lock:
            row = self._db.execute(
                "SELECT text, confidence, produced_by FROM pages WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
        return str(row[0]), float(row[1]), str(row[2])

    def put(self, key: str, text: str, confidence: float, produced_by: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO pages (key, text, confidence, produced_by) "
                "VALUES (?, ?, ?, ?)",
                (key, text, float(confidence), produced_by),
            )
            self._db.commit()
            self.writes += 1

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def close(self) -> None:
        with self._lock:
            with contextlib.suppress(sqlite3.Error):
                self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._db.close()

    def __enter__(self) -> OcrCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_SHARED: OcrCache | None = None
_SHARED_LOCK = threading.Lock()


def shared() -> OcrCache | None:
    """One process-wide store, or ``None`` when disabled.

    Shared rather than per-call because documents are OCR'd from a thread pool and each
    connection would otherwise take its own WAL lock on the same file.
    """
    global _SHARED
    if not cache_enabled():
        return None
    if _SHARED is None:
        with _SHARED_LOCK:
            if _SHARED is None:
                _SHARED = OcrCache()
    return _SHARED


__all__ = [
    "OcrCache",
    "cache_enabled",
    "default_path",
    "document_digest",
    "engine_family",
    "shared",
]
