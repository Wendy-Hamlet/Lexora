"""OCR pipeline with page-level confidence triage.

Scanned (image-only) legal PDFs have no text layer, so the text-layer extractor
(:mod:`lexora.extract.pdf_text_extractor`) returns those pages blank but keeps
the page *slot* (its running char offsets) so OCR can fill it in without shifting
the offsets of later pages — the verbatim contract is anchored on those offsets.

Engine split (see ``make_engine``):
    rapidocr  (default) — PaddleOCR's PP-OCR models run on ONNX Runtime. Bundled
                          with the wheel (no flaky model download), stable on
                          Windows / Python 3.13, CPU-fast, GPU-capable via
                          onnxruntime-gpu. This is what runs locally.
    paddleocr (opt-in)  — native PaddleOCR. Kept for a future GPU server, where
                          paddlepaddle-gpu sidesteps the local CPU oneDNN/PIR
                          executor bug. Import-guarded; ``LEXORA_OCR_ENGINE=paddleocr``.

Confidence triage (page-level, not token-level):
    >= CITABLE_THRESHOLD    → page is citable
    <  CITABLE_THRESHOLD    → page is marked UNVERIFIED_SCAN; clauses on it
                              cannot back a citation until manually corrected.

VLM fallback for medium-confidence pages is intentionally NOT wired: a VLM
re-reads and may paraphrase, which would break the char-offset verbatim contract.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from lexora.extract.pdf_text_extractor import PAGE_SEPARATOR, PdfPage

# Page-level mean confidence at/above which an OCR page may back a citation.
CITABLE_THRESHOLD = 0.80
_DEFAULT_DPI = 300


@dataclass
class OcrPage:
    page_number: int
    text: str
    char_start: int
    char_end: int
    confidence: float       # 0..1, page-level mean of line confidences
    engine: str             # e.g. "rapidocr:1.2.3", "paddleocr:3.7"


# --- OCR engines -----------------------------------------------------------

class OcrEngine(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> list[tuple[str, float]]:
        """Return (text, confidence) per detected line, in reading order."""
        ...


def _reading_order(items: list[tuple[list, str, float]]) -> list[tuple[str, float]]:
    """Sort detected boxes top-to-bottom then left-to-right and drop the box.

    ``items`` are ``[box, text, score]`` as the detectors emit them; ``box`` is
    four ``[x, y]`` corners. Rows are bucketed by their top-y (so words on the
    same visual line stay left-to-right) before sorting."""
    def top_left(box) -> tuple[float, float]:
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        return min(ys), min(xs)

    ordered = sorted(items, key=lambda it: (round(top_left(it[0])[0] / 16.0), top_left(it[0])[1]))
    return [(text, float(score)) for _box, text, score in ordered]


class _RapidEngine:
    """RapidOCR (PP-OCR models on ONNX Runtime). Lazy, single shared instance."""

    def __init__(self) -> None:
        import rapidocr_onnxruntime as _r

        self._RapidOCR = _r.RapidOCR
        self._ocr = None
        try:
            from importlib.metadata import version

            ver = version("rapidocr-onnxruntime")
        except Exception:
            ver = getattr(_r, "__version__", "?")
        self.name = f"rapidocr:{ver}"

    def recognize(self, image: np.ndarray) -> list[tuple[str, float]]:
        if self._ocr is None:
            self._ocr = self._RapidOCR()
        result, _elapse = self._ocr(image)
        if not result:
            return []
        return _reading_order([[row[0], row[1], row[2]] for row in result])


class _PaddleEngine:  # pragma: no cover - validated on the GPU server, not locally
    """Native PaddleOCR (3.x ``predict`` API). For a future GPU server only.

    Local paddlepaddle 3.x on Windows CPU hits an oneDNN/PIR executor bug and
    flaky model downloads; paddlepaddle-gpu on a server avoids both. Kept so the
    engine path exists; ``LEXORA_OCR_ENGINE=paddleocr`` selects it."""

    def __init__(self, use_gpu: bool = False) -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang="en", enable_mkldnn=False,
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        self.name = "paddleocr:3"

    def recognize(self, image: np.ndarray) -> list[tuple[str, float]]:
        res = self._ocr.predict(image)
        if not res:
            return []
        r0 = res[0]
        texts = r0.get("rec_texts", []) or []
        scores = r0.get("rec_scores", []) or []
        boxes = r0.get("rec_polys", None) or r0.get("dt_polys", None)
        if boxes is not None and len(boxes) == len(texts):
            items = [[list(map(list, box)), t, s]
                     for box, t, s in zip(boxes, texts, scores, strict=False)]
            return _reading_order(items)
        return [(t, float(s)) for t, s in zip(texts, scores, strict=False)]


def make_engine(name: str | None = None, *, use_gpu: bool = False) -> OcrEngine:
    """Build an OCR engine. ``name`` defaults to ``LEXORA_OCR_ENGINE`` or rapidocr."""
    name = (name or os.environ.get("LEXORA_OCR_ENGINE") or "rapidocr").lower()
    if name in ("paddle", "paddleocr"):
        return _PaddleEngine(use_gpu=use_gpu)
    return _RapidEngine()


# --- PDF page rendering + OCR ----------------------------------------------

def _render_page(page, dpi: int) -> np.ndarray:
    """Render one PyMuPDF page to an RGB ndarray at ``dpi``."""
    import fitz

    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:  # RGBA -> RGB
        arr = arr[:, :, :3]
    return np.ascontiguousarray(arr)


def _page_text_and_conf(lines: list[tuple[str, float]]) -> tuple[str, float]:
    if not lines:
        return "", 0.0
    text = "\n".join(t for t, _ in lines)
    conf = sum(c for _, c in lines) / len(lines)
    return text, conf


def _open_doc(source: Path | str | bytes):
    """Open a PyMuPDF document from a path or in-memory PDF bytes."""
    import fitz

    if isinstance(source, bytes | bytearray):
        return fitz.open(stream=bytes(source), filetype="pdf")
    return fitz.open(Path(source))


def extract_ocr(
    source: Path | str | bytes,
    languages: list[str] | None = None,
    *,
    dpi: int = _DEFAULT_DPI,
    engine: OcrEngine | None = None,
    page_numbers: set[int] | None = None,
) -> list[OcrPage]:
    """OCR pages of a scanned PDF (path or bytes); return per-page text + page-level
    confidence.

    ``page_numbers`` (1-indexed) restricts OCR to those pages — used to fill only
    the blank slots of a mixed PDF. char offsets here are page-local placeholders;
    :func:`ocr_fill_pages` reflows them into the global document offsets.
    """
    eng = engine or make_engine()
    pages: list[OcrPage] = []
    with _open_doc(source) as doc:
        cursor = 0
        for idx, page in enumerate(doc, start=1):
            if page_numbers is not None and idx not in page_numbers:
                text, conf = "", 0.0
            else:
                text, conf = _page_text_and_conf(eng.recognize(_render_page(page, dpi)))
            pages.append(OcrPage(idx, text, cursor, cursor + len(text), conf, eng.name))
            cursor += len(text) + len(PAGE_SEPARATOR)
    return pages


# --- pipeline integration --------------------------------------------------

def ocr_fill_pages(
    pages: list[PdfPage],
    source: Path | str | bytes,
    *,
    dpi: int = _DEFAULT_DPI,
    engine: OcrEngine | None = None,
) -> tuple[list[PdfPage], dict[int, float]]:
    """Fill the blank (image-only) page slots of ``pages`` with OCR text.

    Returns a NEW ``PdfPage`` list whose global char offsets are recomputed so
    the document text stays consistent (verbatim spans slice it by offset), plus
    a ``{page_number: confidence}`` map for the OCR'd pages so the caller can
    mark sub-threshold pages UNVERIFIED_SCAN. Text-layer pages are untouched."""
    blank = {p.page_number for p in pages if not p.has_text_layer}
    if not blank:
        return pages, {}

    eng = engine or make_engine()
    ocr_pages = extract_ocr(source, dpi=dpi, engine=eng, page_numbers=blank)
    ocr_by_num = {op.page_number: op for op in ocr_pages}

    page_conf: dict[int, float] = {}
    specs: list[tuple[int, str, bool]] = []
    for p in pages:
        if p.page_number in blank and p.page_number in ocr_by_num:
            op = ocr_by_num[p.page_number]
            page_conf[p.page_number] = op.confidence
            specs.append((p.page_number, op.text, bool(op.text.strip())))
        else:
            specs.append((p.page_number, p.text, p.has_text_layer))

    filled: list[PdfPage] = []
    cursor = 0
    for pn, text, has_text in specs:
        filled.append(PdfPage(pn, text, cursor, cursor + len(text), has_text))
        cursor += len(text) + len(PAGE_SEPARATOR)
    return filled, page_conf


__all__ = [
    "OcrPage",
    "OcrEngine",
    "CITABLE_THRESHOLD",
    "make_engine",
    "extract_ocr",
    "ocr_fill_pages",
]
