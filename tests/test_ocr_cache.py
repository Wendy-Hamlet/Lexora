"""OCR must not be recomputed, and must not change answer when the GPU drops out.

Replay covers the network. It does not cover OCR, which is local computation -- so every
replayed run used to re-recognise the same scans, and a mid-run GPU->CPU fallback silently
produced *different text* for the same page, which then invalidated every judge verdict on
that document. Both are the same defect: the transcription of a page was treated as a
property of the machine instead of a property of the page.
"""
from __future__ import annotations

import numpy as np
import pytest

from lexora.extract import ocr_cache
from lexora.extract.ocr_cache import OcrCache, document_digest, engine_family


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_cache, "_SHARED", None)
    monkeypatch.setenv("LEXORA_OCR_CACHE_PATH", str(tmp_path / "ocr.sqlite"))
    monkeypatch.delenv("LEXORA_OCR_CACHE", raising=False)
    return OcrCache(tmp_path / "ocr.sqlite")


def test_the_gpu_suffix_never_reaches_the_key(store):
    """The whole point. `rapidocr:1.4.4` and `rapidocr:1.4.4+cuda` are the same engine
    asked the same question; only the hardware that answered differs. On 2026-08-02 the
    runtime CPU fallback landed, two Acts re-transcribed slightly differently, and a
    full-corpus replay degraded 40 documents to the keyword lane."""
    on_gpu = store.key("deadbeef", 7, 300, "rapidocr:1.4.4+cuda")
    on_cpu = store.key("deadbeef", 7, 300, "rapidocr:1.4.4")
    assert on_gpu == on_cpu


def test_a_model_upgrade_does_invalidate(store):
    """Ignoring the accelerator must not become ignoring the model."""
    assert store.key("deadbeef", 7, 300, "rapidocr:1.4.4") != \
        store.key("deadbeef", 7, 300, "rapidocr:2.0.0")
    assert store.key("deadbeef", 7, 300, "rapidocr:1.4.4") != \
        store.key("deadbeef", 7, 300, "paddleocr:3.7")


def test_page_dpi_and_document_all_separate_rows(store):
    base = store.key("deadbeef", 7, 300, "rapidocr:1.4.4")
    assert base != store.key("deadbeef", 8, 300, "rapidocr:1.4.4")
    assert base != store.key("deadbeef", 7, 150, "rapidocr:1.4.4")
    assert base != store.key("cafe", 7, 300, "rapidocr:1.4.4")


def test_engine_family_strips_only_the_suffix():
    assert engine_family("rapidocr:1.4.4+cuda") == "rapidocr:1.4.4"
    assert engine_family("rapidocr:1.4.4") == "rapidocr:1.4.4"
    assert engine_family("paddleocr:3.7") == "paddleocr:3.7"


def test_round_trip_keeps_text_confidence_and_provenance(store):
    k = store.key("deadbeef", 1, 300, "rapidocr:1.4.4+cuda")
    store.put(k, "AKTA 709", 0.91, "rapidocr:1.4.4+cuda")
    assert store.get(k) == ("AKTA 709", 0.91, "rapidocr:1.4.4+cuda")
    assert store.hits == 1 and store.misses == 0
    assert store.get(store.key("deadbeef", 2, 300, "rapidocr:1.4.4")) is None
    assert store.misses == 1


def test_an_empty_page_is_a_real_answer_and_is_cached(store):
    """A blank scan is repeatable. Treating "" as a miss would re-OCR it every run --
    and image-only front matter is exactly what a 98 MB scan is full of."""
    k = store.key("deadbeef", 1, 300, "rapidocr:1.4.4")
    store.put(k, "", 0.0, "rapidocr:1.4.4")
    assert store.get(k) == ("", 0.0, "rapidocr:1.4.4")


def test_the_same_bytes_hash_the_same_from_a_path_or_from_memory(store, tmp_path):
    """A recording serves the identical PDF back under a different temp filename, so the
    key has to come from the content."""
    blob = b"%PDF-1.4 fake bytes"
    f = tmp_path / "Act 709 ori.pdf"
    f.write_bytes(blob)
    assert document_digest(blob) == document_digest(f)
    assert document_digest(blob) != document_digest(b"%PDF-1.4 other bytes")


def test_disabled_by_env_returns_no_store(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr_cache, "_SHARED", None)
    monkeypatch.setenv("LEXORA_OCR_CACHE", "0")
    assert ocr_cache.shared() is None
    monkeypatch.setenv("LEXORA_OCR_CACHE", "1")
    monkeypatch.setenv("LEXORA_OCR_CACHE_PATH", str(tmp_path / "ocr.sqlite"))
    monkeypatch.setattr(ocr_cache, "_SHARED", None)
    assert ocr_cache.shared() is not None


class _CountingEngine:
    """Stands in for RapidOCR. `name` flips to CPU mid-document, exactly as the real
    engine does when cuDNN throws."""

    def __init__(self, fail_on_page: int | None = None) -> None:
        self.name = "rapidocr:1.4.4+cuda"
        self.calls = 0
        self._fail_on = fail_on_page

    def recognize(self, image):  # noqa: ANN001
        self.calls += 1
        if self._fail_on is not None and self.calls == self._fail_on:
            self.name = "rapidocr:1.4.4"          # dropped to CPU
            return [("PAGE TEXT cpu", 0.90)]
        return [("PAGE TEXT", 0.95)]


@pytest.fixture
def two_page_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for _ in range(2):
        doc.new_page(width=200, height=200)
    p = tmp_path / "scan.pdf"
    doc.save(str(p))
    doc.close()
    return p


def test_a_second_run_over_the_same_pdf_calls_the_engine_zero_times(
    two_page_pdf, tmp_path, monkeypatch
):
    from lexora.extract import ocr_extractor

    monkeypatch.setattr(ocr_cache, "_SHARED", None)
    monkeypatch.setenv("LEXORA_OCR_CACHE_PATH", str(tmp_path / "ocr.sqlite"))
    monkeypatch.delenv("LEXORA_OCR_CACHE", raising=False)
    monkeypatch.setattr(ocr_extractor, "_render_page",
                        lambda page, dpi: np.zeros((4, 4, 3), dtype=np.uint8))

    first = _CountingEngine()
    a = ocr_extractor.extract_ocr(two_page_pdf, engine=first)
    assert first.calls == 2

    second = _CountingEngine()
    b = ocr_extractor.extract_ocr(two_page_pdf, engine=second)
    assert second.calls == 0, "the second run re-OCR'd a page it had already read"
    assert [p.text for p in a] == [p.text for p in b]
    assert [p.char_start for p in a] == [p.char_start for p in b]


def test_a_gpu_that_dies_midway_still_answers_from_the_gpu_cache(
    two_page_pdf, tmp_path, monkeypatch
):
    """The 2026-08-02 failure, in miniature: page 1 is transcribed on the GPU, the GPU
    dies, page 2 comes off the CPU. A re-run must serve BOTH from cache -- if the
    accelerator were in the key, page 1 would miss and be transcribed a second time, with
    different text, and every judge verdict resting on it would be voided."""
    from lexora.extract import ocr_extractor

    monkeypatch.setattr(ocr_cache, "_SHARED", None)
    monkeypatch.setenv("LEXORA_OCR_CACHE_PATH", str(tmp_path / "ocr.sqlite"))
    monkeypatch.delenv("LEXORA_OCR_CACHE", raising=False)
    monkeypatch.setattr(ocr_extractor, "_render_page",
                        lambda page, dpi: np.zeros((4, 4, 3), dtype=np.uint8))

    dying = _CountingEngine(fail_on_page=2)
    first = ocr_extractor.extract_ocr(two_page_pdf, engine=dying)
    assert dying.calls == 2
    assert [p.engine for p in first] == ["rapidocr:1.4.4+cuda", "rapidocr:1.4.4"], \
        "each page must record the engine that really produced it"

    cpu_only = _CountingEngine()
    cpu_only.name = "rapidocr:1.4.4"
    again = ocr_extractor.extract_ocr(two_page_pdf, engine=cpu_only)
    assert cpu_only.calls == 0
    assert [p.text for p in again] == [p.text for p in first]
