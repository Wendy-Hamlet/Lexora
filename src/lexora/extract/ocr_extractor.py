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
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from lexora.extract import ocr_cache
from lexora.extract.ocr_cache import document_digest
from lexora.extract.pdf_text_extractor import (
    PAGE_SEPARATOR,
    PdfPage,
    _strip_running_lines,
)

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


# A vertical whitespace corridor this wide (as a fraction of the page's text width, with
# an absolute floor) separates two columns. Deliberately conservative: on single-column
# prose every full-width line crosses the middle of the page, so no corridor can exist and
# no split is possible. It takes a genuinely empty vertical channel to trigger one.
_MIN_CORRIDOR_FRAC = 0.04
_MIN_CORRIDOR_ABS = 8.0


def _x_columns(items: list) -> list[list]:
    """Group boxes into vertical columns separated by a whitespace corridor.

    Returns one list per column, left to right; a single-column page returns one group.

    A page is not always one column. Malaysia's statutes set marginal notes in a column
    beside the body, and gazettes print two languages side by side. Bucketing by y and
    sorting by x — a single-column reading order — does not merely reorder such a page, it
    splices one column's words INTO the other's sentences, because each wrapped fragment of
    the note lands between two body lines.

    Measured: MY Communications and Multimedia Act s.254 shipped in
    ``submission_final_20260714.csv`` as "... for the purposes of the Aditional execution
    of this Act ... have powers. power to do all or any of the following:". The marginal
    note is "Additional powers." (OCR read "Aditional"); the real provision reads "for the
    purposes of the execution of this Act ... have power to do". The quote was labelled
    verbatim and was not.
    """
    spans = sorted((min(p[0] for p in box), max(p[0] for p in box)) for box, _t, _s in items)
    if not spans:
        return [items]
    merged: list[list[float]] = []
    for x0, x1 in spans:
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    if len(merged) == 1:
        return [items]
    width = merged[-1][1] - merged[0][0]
    floor = max(_MIN_CORRIDOR_FRAC * width, _MIN_CORRIDOR_ABS)
    cuts = [
        (merged[i][1] + merged[i + 1][0]) / 2.0
        for i in range(len(merged) - 1)
        if merged[i + 1][0] - merged[i][1] >= floor
    ]
    if not cuts:
        return [items]
    groups: list[list] = [[] for _ in range(len(cuts) + 1)]
    for it in items:
        x0 = min(p[0] for p in it[0])
        idx = sum(1 for c in cuts if x0 > c)
        groups[idx].append(it)
    return [g for g in groups if g]


def _reading_order(items: list[tuple[list, str, float]]) -> list[tuple[str, float]]:
    """Sort detected boxes into reading order and drop the box.

    ``items`` are ``[box, text, score]`` as the detectors emit them; ``box`` is
    four ``[x, y]`` corners. Columns are separated first (see :func:`_x_columns`), then
    within each column rows are bucketed by their top-y so words on the same visual line
    stay left-to-right. Column-major is the right order for both layouts this affects: a
    two-language gazette reads down one column and then the other, and a marginal note
    ends up after the body it annotates instead of inside it."""
    def top_left(box) -> tuple[float, float]:
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        return min(ys), min(xs)

    ordered: list = []
    for column in _x_columns(list(items)):
        ordered.extend(sorted(
            column,
            key=lambda it: (round(top_left(it[0])[0] / 16.0), top_left(it[0])[1]),
        ))
    return [(text, float(score)) for _box, text, score in ordered]


def _register_cuda_dlls() -> None:
    """Put the pip-installed NVIDIA runtime on PATH, once, before ORT loads its CUDA EP.

    cuDNN pulls its sublibraries (``cudnn_engines_tensor_ir64_9.dll``) at RUN time with
    LoadLibrary, which searches PATH. ``ort.preload_dlls()`` only pins the main dll and
    ``os.add_dll_directory`` does not cover LoadLibrary, so without this the CUDA provider
    is created happily and then dies on the first Conv with
    ``CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`` -- and ORT silently retries on CPU. That is
    why OCR ran on the CPU for months while reporting "+cuda"."""
    global _CUDA_DLLS_REGISTERED
    if _CUDA_DLLS_REGISTERED:
        return
    _CUDA_DLLS_REGISTERED = True
    try:
        import nvidia
    except ImportError:
        return  # no pip-installed CUDA runtime; a system CUDA on PATH still works
    root = Path(nvidia.__file__).parent
    bins = [str(root / sub / "bin") for sub in
            ("cudnn", "cublas", "cuda_runtime", "cufft", "curand", "cusparse", "cusolver")
            if (root / sub / "bin").is_dir()]
    if bins:
        os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ.get("PATH", "")


_CUDA_DLLS_REGISTERED = False
_GPU_HINT_SHOWN = False


def _hint_if_gpu_wasted() -> None:
    """Say something when there is an NVIDIA GPU but OCR is about to run on the CPU wheel.

    This is the one failure the provider read-back cannot catch: the CPU build of
    onnxruntime has no CUDA provider to fall back FROM, so every session is legitimately
    on CPU and nothing looks wrong -- the run is just 5x slower for no reason. It is what
    a fresh venv gets by default, and it cost us a full submission run to notice."""
    global _GPU_HINT_SHOWN
    if _GPU_HINT_SHOWN:
        return
    _GPU_HINT_SHOWN = True

    import logging
    import shutil
    from importlib.metadata import PackageNotFoundError, version

    if shutil.which("nvidia-smi") is None:
        return  # no NVIDIA driver, so the CPU wheel is the right answer here
    try:
        version("onnxruntime-gpu")
        return  # GPU wheel installed; whatever happened next, _build() reports it
    except PackageNotFoundError:
        pass
    logging.getLogger(__name__).warning(
        "OCR: this machine has an NVIDIA GPU, but onnxruntime is the CPU-only wheel, so "
        "OCR will run ~5x slower than it needs to. To use the GPU: pip uninstall -y "
        "onnxruntime && pip install -r requirements-gpu.txt  (set LEXORA_OCR_GPU=0 to "
        "silence this and stay on the CPU deliberately)")


def _ocr_cuda_ready() -> bool:
    """True if the ONNX Runtime CUDA provider is usable for OCR.

    ``LEXORA_OCR_GPU`` = ``0``/``cpu`` forces CPU; otherwise the GPU is used when present.
    Availability is necessary but NOT sufficient — the provider can list as available and
    still fail at the first convolution (see :func:`_register_cuda_dlls`), so the engine
    verifies the real providers after building and reports what it actually got."""
    pref = os.environ.get("LEXORA_OCR_GPU", "auto").lower()
    if pref in ("0", "cpu", "false", "no", "off"):
        return False
    try:
        import contextlib

        _register_cuda_dlls()
        import onnxruntime as ort

        with contextlib.suppress(Exception):
            ort.preload_dlls()
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return True
    except Exception:
        return False
    _hint_if_gpu_wasted()
    return False


def _patch_rapidocr_cuda_kwargs() -> None:
    """Make ``cls_use_cuda`` / ``rec_use_cuda`` actually reach the ORT session.

    rapidocr-onnxruntime <= 1.2.x (the newest build for Python 3.13) strips the ``det_``
    prefix from every kwarg but strips ``cls_``/``rec_`` only for a whitelist of keys. So
    ``cls_use_cuda`` arrives at ``OrtInferSession`` still prefixed, while it reads
    ``config["use_cuda"]`` -- which stays False. Detection lands on the GPU, classification
    and RECOGNITION (the expensive stage) stay on the CPU, and nothing says so.

    1.3+ fixed this upstream, so patch only the broken versions. Also make the whitelist
    strippers tolerate a missing ``model_path`` (they index it unconditionally, which is
    the ``KeyError: 'model_path'`` raised by passing ``det_use_cuda`` alone)."""
    from rapidocr_onnxruntime.utils import UpdateParameters

    if getattr(UpdateParameters, "_lexora_patched", False):
        return

    def _stripper(prefix: str):
        def _update(self, config, sub: dict):  # noqa: ANN001
            if sub:
                sub = {(k[len(prefix):] if k.startswith(prefix) else k): v
                       for k, v in sub.items()}
                if not sub.get("model_path"):
                    sub["model_path"] = config["model_path"]
                config.update(sub)
            return config
        return _update

    UpdateParameters.update_det_params = _stripper("det_")
    UpdateParameters.update_cls_params = _stripper("cls_")
    UpdateParameters.update_rec_params = _stripper("rec_")
    UpdateParameters._lexora_patched = True


def _session_providers(ocr) -> dict[str, str]:
    """The execution provider each of the three models really ended up on.

    Two shapes to probe, and both matter. The model attribute is named ``text_det`` /
    ``text_rec`` on rapidocr 1.4.x but ``text_detector`` / ``text_recognizer`` on older
    builds; the ORT session then hangs off ``.infer.session`` (det, cls) or
    ``.session.session`` (rec). A name we fail to resolve is a model we cannot see, and an
    unseen model is one this dict silently omits -- which would let ``_build`` compare
    ``len(on_cuda) == len(provs)`` over a single stage and stamp ``+cuda`` on an engine
    whose expensive stages sat on the CPU. That is the exact lie this function exists to
    prevent, so probe every alias."""
    out: dict[str, str] = {}
    for label, aliases in (("det", ("text_det", "text_detector")),
                           ("cls", ("text_cls", "text_classifier")),
                           ("rec", ("text_rec", "text_recognizer"))):
        model = next((m for m in (getattr(ocr, a, None) for a in aliases) if m), None)
        if model is None:
            continue
        holder = getattr(model, "infer", None) or getattr(model, "session", None)
        session = getattr(holder, "session", None)
        if session is not None and hasattr(session, "get_providers"):
            provs = session.get_providers()
            out[label] = provs[0] if provs else "?"
    return out


class _RapidEngine:
    """RapidOCR (PP-OCR models on ONNX Runtime). Lazy, single shared instance.

    Runs on GPU when onnxruntime-gpu + a working CUDA/cuDNN runtime are present (det, cls
    and rec all on CUDA — measured 5.6x faster on a scanned statute, identical text), and
    falls back to CPU otherwise. ``name`` reports the providers the sessions ACTUALLY got,
    never what was requested."""

    def __init__(self) -> None:
        import rapidocr_onnxruntime as _r

        self._RapidOCR = _r.RapidOCR
        self._ocr = None
        self._want_cuda = _ocr_cuda_ready()
        # One engine is shared across document workers, so the mid-document fallback
        # below must not have two threads rebuilding it at once.
        self._lock = threading.Lock()
        try:
            from importlib.metadata import version

            self._ver = version("rapidocr-onnxruntime")
        except Exception:
            self._ver = getattr(_r, "__version__", "?")
        # Provisional; _build() overwrites it with the observed providers.
        self.name = f"rapidocr:{self._ver}"

    def _build(self) -> None:
        import logging

        log = logging.getLogger(__name__)
        ocr = None
        if self._want_cuda:
            try:
                _patch_rapidocr_cuda_kwargs()
                ocr = self._RapidOCR(
                    det_use_cuda=True, det_model_path=None,
                    cls_use_cuda=True, cls_model_path=None,
                    rec_use_cuda=True, rec_model_path=None,
                )
            except Exception as exc:  # never silently: a CPU fallback is a 5x slowdown
                log.warning("OCR: CUDA requested but engine build failed (%s: %s); "
                            "falling back to CPU", type(exc).__name__, exc)
                ocr = None
        if ocr is None:
            ocr = self._RapidOCR()
        self._ocr = ocr

        provs = _session_providers(ocr)
        on_cuda = {k for k, v in provs.items() if v.startswith("CUDA")}
        if self._want_cuda and len(on_cuda) < len(provs):
            log.warning("OCR: CUDA available but only %s reached the GPU (%s) — the rest "
                        "run on CPU", sorted(on_cuda) or "nothing", provs)
        suffix = "+cuda" if provs and len(on_cuda) == len(provs) else ""
        self.name = f"rapidocr:{self._ver}{suffix}"

    def recognize(self, image: np.ndarray) -> list[tuple[str, float]]:
        if self._ocr is None:
            with self._lock:
                if self._ocr is None:
                    self._build()
        try:
            result, _elapse = self._ocr(image)
        except Exception as exc:  # noqa: BLE001 — see below; this is not a swallow
            # A GPU that builds is not a GPU that runs. Measured 2026-08-02 on the
            # Communications and Multimedia Act 1998 -- a GOLD instrument, 33 MB of scan:
            # every session came up on CUDA, then a page threw
            # `CUDNN_FE failure 11: CUDNN_BACKEND_API_FAILED` mid-document and the whole
            # Act was skipped. The same pages OCR fine on CPU. Losing a statute to a
            # cuDNN error is a far worse trade than reading it five times slower, so the
            # engine drops to CPU and retries the page ONCE. If CPU fails too, the error
            # is real and propagates.
            import logging

            if not self._want_cuda:
                raise
            logging.getLogger(__name__).warning(
                "OCR: the GPU failed mid-document (%s: %s); rebuilding on CPU and "
                "retrying this page. The run gets slower, not shorter.",
                type(exc).__name__, str(exc)[:160])
            with self._lock:
                self._want_cuda = False
                self._ocr = self._RapidOCR()
                self.name = f"rapidocr:{self._ver}"
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

    Pages already transcribed at this DPI by this engine family come back from
    :mod:`lexora.extract.ocr_cache` without being rendered or recognised again — OCR is
    the one expensive step the record/replay layer cannot cover, because no socket is
    involved. See that module for why the key ignores GPU vs CPU.
    """
    eng = engine or make_engine()
    cache = ocr_cache.shared()
    doc_sha = document_digest(source) if cache is not None else ""
    hits = misses = 0
    pages: list[OcrPage] = []
    with _open_doc(source) as doc:
        cursor = 0
        for idx, page in enumerate(doc, start=1):
            if page_numbers is not None and idx not in page_numbers:
                text, conf, produced = "", 0.0, eng.name
            else:
                key = cache.key(doc_sha, idx, dpi, eng.name) if cache is not None else ""
                found = cache.get(key) if cache is not None else None
                if found is not None:
                    text, conf, produced = found
                    hits += 1
                else:
                    text, conf = _page_text_and_conf(eng.recognize(_render_page(page, dpi)))
                    # eng.name is read AFTER recognize on purpose: a mid-document GPU
                    # failure rewrites it, and the row must say which engine really
                    # produced this page, not which one was asked.
                    produced = eng.name
                    misses += 1
                    if cache is not None:
                        cache.put(key, text, conf, produced)
            pages.append(OcrPage(idx, text, cursor, cursor + len(text), conf, produced))
            cursor += len(text) + len(PAGE_SEPARATOR)
    if cache is not None and (hits or misses):
        import logging

        logging.getLogger(__name__).info(
            "OCR cache: %d/%d page(s) served from cache, %d recognised",
            hits, hits + misses, misses)
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
    mark sub-threshold pages UNVERIFIED_SCAN. Text-layer pages are untouched.

    The running-head cleaner is re-run over the FILLED pages. It ran once already,
    in `_pages_from_doc`, but a scanned page is blank there — so on an image-only
    PDF the cross-page repetition test saw nothing at all, and every page header
    ("Act 762", "Part 4" + a page number) survived into the text and then into the
    verbatim quotes. Once OCR has filled the pages, the repetition is finally
    visible, so this is the first point at which a scan CAN be cleaned. Offsets are
    recomputed below from the cleaned text, so spans stay consistent."""
    blank = {p.page_number for p in pages if not p.has_text_layer}
    if not blank:
        return pages, {}

    # Pass the engine through (default None); extract_ocr builds it lazily only
    # when it actually OCRs a page. Building it here would import the OCR backend
    # eagerly — unwanted when the backend is absent (e.g. CI without the [ocr] extra).
    ocr_pages = extract_ocr(source, dpi=dpi, engine=engine, page_numbers=blank)
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

    cleaned = _strip_running_lines([text for _, text, _ in specs])

    filled: list[PdfPage] = []
    cursor = 0
    for (pn, _, has_text), text in zip(specs, cleaned, strict=True):
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
