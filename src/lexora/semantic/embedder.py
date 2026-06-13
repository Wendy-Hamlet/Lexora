"""Dense sentence embeddings via fastembed (optional, ONNX — no torch).

`fastembed` runs a small BGE model through onnxruntime, so it adds a dense
retrieval channel without pulling in torch. The model is downloaded once and
cached; loading it costs ~30s the first time, so callers reuse the singleton from
:func:`get_embedder` rather than constructing per call.

Optional-dependency contract (mirrors :mod:`lexora.collect.browser`):
:func:`is_available` reports whether fastembed is importable, and every caller
falls back to keyword-only behaviour when it is not — a bare install and the
offline test suite never require this package.

Model choice is env-overridable (``LEXORA_EMBED_MODEL``) so a small model can be
swapped for a stronger one without code changes. The model cache defaults to an
ASCII path (``LEXORA_MODEL_CACHE``) because this machine's ``%USERPROFILE%`` /
``%TEMP%`` are non-ASCII, which makes the default HuggingFace cache fail.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np

# Default 384-dim English model: small (~130MB), fast on CPU, strong enough to
# bridge the statutory-vs-concept vocabulary gap for the (English) round-1
# economies. For non-Latin generalization, set the model to a multilingual one
# (the submission's promised ``BAAI/bge-m3``, ~2.3GB, cross-lingual) — fastembed
# downloads it on first use.
#
# One knob, two accepted names: ``LEXORA_EMBED_MODEL`` is the legacy explicit
# override; ``LEXORA_EMBEDDING_MODEL`` is the unified config var (see config.py).
# Honouring both fixes a dead-config bug — config defaulted ``LEXORA_EMBEDDING_
# MODEL=BAAI/bge-m3`` while the embedder only ever read ``LEXORA_EMBED_MODEL``, so
# the promised multilingual model was never actually reachable.
_FALLBACK_MODEL = "BAAI/bge-small-en-v1.5"


def resolve_model_name() -> str:
    """Pick the embedding model from the environment (legacy name wins), else the
    light English fallback. Centralised so the two accepted env vars never drift."""
    return (
        os.environ.get("LEXORA_EMBED_MODEL")
        or os.environ.get("LEXORA_EMBEDDING_MODEL")
        or _FALLBACK_MODEL
    )


DEFAULT_MODEL = resolve_model_name()


def _default_cache_dir() -> str:
    """An ASCII model cache. fastembed/HF default to ``~/.cache`` /``%TEMP%``,
    which are under this machine's non-ASCII ``%USERPROFILE%`` (金持恒) and fail
    silently — so we pin an ASCII directory."""
    env = os.environ.get("LEXORA_MODEL_CACHE")
    if env:
        return env
    if sys.platform == "win32":
        return r"C:\tmp\lexora_models"
    return os.path.join(os.path.expanduser("~"), ".cache", "lexora", "models")


def is_available() -> bool:
    """True if the fastembed backend can be imported."""
    try:
        import fastembed  # noqa: F401
    except Exception:
        return False
    return True


def _cuda_ready() -> bool:
    """True if the ONNX Runtime CUDA provider is available and wanted.

    ``LEXORA_EMBED_DEVICE`` = ``cpu`` forces CPU; ``cuda``/``auto`` (default) use
    the GPU when present. onnxruntime-gpu finds the nvidia pip-wheel DLLs only
    after ``preload_dlls()``, so we call it before probing the providers — without
    it the CUDA provider lists as available but silently falls back to CPU."""
    if os.environ.get("LEXORA_EMBED_DEVICE", "auto").lower() == "cpu":
        return False
    try:
        import contextlib

        import onnxruntime as ort

        with contextlib.suppress(Exception):
            ort.preload_dlls()
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


class Embedder:
    """Thin wrapper over a fastembed text model returning L2-normalized vectors.

    Normalizing on the way out makes a dot product a cosine similarity, so the
    downstream crosswalk / re-rank code can use plain matrix multiplies. Uses the
    GPU automatically when onnxruntime-gpu + CUDA are present (~tens of x faster on
    a batch), falling back to CPU on any error — the vectors are identical either way.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, *, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        cache = cache_dir or _default_cache_dir()
        os.makedirs(cache, exist_ok=True)
        self.model_name = model_name
        self.device = "cpu"
        self._model = None
        if _cuda_ready():
            try:
                self._model = TextEmbedding(model_name, cache_dir=cache, cuda=True)
                self.device = "cuda"
            except Exception:
                self._model = None  # fastembed without GPU extra, or CUDA init failed
        if self._model is None:
            self._model = TextEmbedding(model_name, cache_dir=cache)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` -> ``(len(texts), dim)`` float32, L2-normalized rows.

        An empty input returns an empty ``(0, 0)`` array so callers can treat it
        uniformly without a special case."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vecs = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


@lru_cache(maxsize=4)
def get_embedder(model_name: str = DEFAULT_MODEL) -> Embedder:
    """Return a process-wide cached :class:`Embedder` (model load is expensive)."""
    return Embedder(model_name)


def cosine_topk(
    query_vec: np.ndarray, doc_vecs: np.ndarray, k: int
) -> list[tuple[int, float]]:
    """Top-``k`` ``(doc_index, cosine)`` for one normalized query against
    normalized ``doc_vecs`` (rows). Returns fewer than ``k`` if there are fewer
    docs; empty if either side is empty."""
    if doc_vecs.size == 0 or query_vec.size == 0:
        return []
    sims = doc_vecs @ query_vec
    k = min(k, sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def reciprocal_rank_fusion(
    rankings: list[list[str]], *, k: int = 60, weights: list[float] | None = None
) -> dict[str, float]:
    """Reciprocal-rank fusion of several ranked id lists into one score map.

    RRF score of an id = sum over each ranking of ``weight / (k + rank)`` (rank
    from 1, ``weight`` defaulting to 1). It fuses heterogeneous scorers (BM25's
    unbounded scores and cosine's [-1,1]) without normalizing either — only the
    within-list rank matters, which is exactly what we want when BM25 scores
    saturate and can't discriminate. ``weights`` (one per ranking) lets a caller
    trust the precise channel more than the recall channel: a higher BM25 weight
    keeps a confident BM25 top-1 from being demoted by a noisy dense match while
    still letting dense pull a BM25-missed clause into the lower ranks.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return scores


__all__ = [
    "DEFAULT_MODEL",
    "Embedder",
    "is_available",
    "get_embedder",
    "cosine_topk",
    "reciprocal_rank_fusion",
]
