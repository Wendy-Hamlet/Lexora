"""Cross-encoder reranking via fastembed (optional, ONNX — no torch).

The BM25 / dense channels (:mod:`lexora.classify.retrieval`) are bi-encoders: they
score a clause against the indicator INDEPENDENTLY, so a clause that merely shares
vocabulary can outrank the on-point provision. A cross-encoder reads the
(indicator, clause) PAIR jointly and scores their actual relevance — the standard
precision stage that sits on top of a recall-oriented retriever. It is the closest
deterministic analogue to the LLM verifier, at a fraction of the cost.

This runs the reranker through ``fastembed``'s ``TextCrossEncoder`` (ONNX via
onnxruntime), so it adds the precision stage WITHOUT pulling in torch — mirroring
:mod:`lexora.semantic.embedder`. The model downloads once and is cached; callers
reuse the singleton from :func:`get_reranker`.

Optional-dependency contract (same as the embedder): :func:`is_available` reports
whether the backend is importable, and every caller falls back to the un-reranked
order when it is not — a bare install and the offline test suite never require it.

Model choice is env-overridable (``LEXORA_RERANK_MODEL``). The default,
``BAAI/bge-reranker-base`` (XLM-RoBERTa-base, ~278M, multilingual), is the small
sibling of the promised ``BAAI/bge-reranker-v2-m3`` (XLM-R-large, ~568M), so a
later upgrade to v2-m3 is an apples-to-apples swap. fastembed 0.8's supported
multilingual rerankers also include ``jinaai/jina-reranker-v2-base-multilingual``.

Red line (same as the verifier / synonyms): the reranker only REORDERS clauses the
retriever already surfaced. It never writes, quotes, or attributes clause text; the
verbatim citation always comes from canonical storage. A bad rerank can at worst
cost a little precision — never a fabricated quote.
"""
from __future__ import annotations

import os
from functools import lru_cache

from lexora.semantic.embedder import _default_cache_dir

# Default: bge-reranker-base — multilingual XLM-R-base, the small sibling of the
# promised bge-reranker-v2-m3, so the eventual upgrade is a one-env-var swap. Env
# override lets a stronger/larger model drop in without code changes.
_FALLBACK_MODEL = "BAAI/bge-reranker-base"


def resolve_model_name() -> str:
    """Pick the reranker model from the environment, else the multilingual base."""
    return os.environ.get("LEXORA_RERANK_MODEL") or _FALLBACK_MODEL


DEFAULT_MODEL = resolve_model_name()


def is_available() -> bool:
    """True if the fastembed cross-encoder backend can be imported."""
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: F401
    except Exception:
        return False
    return True


def _cuda_ready() -> bool:
    """True if the ONNX Runtime CUDA provider is available and wanted.

    Mirrors :func:`lexora.semantic.embedder._cuda_ready`: ``LEXORA_RERANK_DEVICE`` =
    ``cpu`` forces CPU; ``cuda``/``auto`` (default) use the GPU when present.
    onnxruntime-gpu only finds the nvidia pip-wheel DLLs after ``preload_dlls()``,
    so we call it before probing providers."""
    if os.environ.get("LEXORA_RERANK_DEVICE", "auto").lower() == "cpu":
        return False
    try:
        import contextlib

        import onnxruntime as ort

        with contextlib.suppress(Exception):
            ort.preload_dlls()
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


class Reranker:
    """Thin wrapper over a fastembed ``TextCrossEncoder``.

    :meth:`rerank` scores each document against the query and returns the document
    indices best-first with their relevance scores. Uses the GPU automatically when
    onnxruntime-gpu + CUDA are present, falling back to CPU on any error — the
    scores are identical either way."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, cache_dir: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        cache = cache_dir or _default_cache_dir()
        os.makedirs(cache, exist_ok=True)
        self.model_name = model_name
        # Offline / mirror escape hatch: when ``LEXORA_RERANK_MODEL_PATH`` points at a
        # local onnx model dir (one holding ``onnx/model.onnx`` + tokenizer/config),
        # load straight from it with no Hub call. Needed where huggingface.co is
        # unreachable and the weights were fetched via a mirror (this machine's case).
        local = os.environ.get("LEXORA_RERANK_MODEL_PATH") or None
        extra = {"specific_model_path": local} if local else {}
        self.device = "cpu"
        self._model = None
        if _cuda_ready():
            try:
                self._model = TextCrossEncoder(model_name, cache_dir=cache, cuda=True, **extra)
                self.device = "cuda"
            except Exception:
                self._model = None  # fastembed without GPU extra, or CUDA init failed
        if self._model is None:
            self._model = TextCrossEncoder(model_name, cache_dir=cache, **extra)

    def scores(self, query: str, documents: list[str]) -> list[float]:
        """Cross-encoder relevance score for each document against ``query``
        (aligned to ``documents``; empty input -> empty list)."""
        if not query or not documents:
            return []
        return [float(s) for s in self._model.rerank(query, documents)]

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """``(doc_index, score)`` pairs sorted best-first. Empty if either side is
        empty, so callers can treat it uniformly without a special case."""
        scores = self.scores(query, documents)
        if not scores:
            return []
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in order]


@lru_cache(maxsize=2)
def get_reranker(model_name: str = DEFAULT_MODEL) -> Reranker:
    """Return a process-wide cached :class:`Reranker` (model load is expensive)."""
    return Reranker(model_name)


__all__ = ["DEFAULT_MODEL", "Reranker", "is_available", "get_reranker", "resolve_model_name"]
