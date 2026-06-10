"""OpenAI-compatible LLM client.

Works with any OpenAI-compatible endpoint:
    - vLLM (recommended for hackathon demo)
    - llama.cpp server
    - Ollama
    - OpenAI API itself

Reads endpoint + key + model from LEXORA_LLM_* env vars (see config.py).

Optional-dependency contract (mirrors :mod:`lexora.semantic.embedder`):
:func:`is_available` reports whether the ``openai`` SDK is importable, and the
verifier falls back to a no-op (keyword/BM25 + verbatim only) when it is not. A
bare install and the offline test suite never require this package or a server.
"""
from __future__ import annotations

import json

from lexora.config import load_config


def is_available() -> bool:
    """True if the ``openai`` SDK can be imported (the LLM backend's only dep)."""
    try:
        import openai  # noqa: F401
    except Exception:
        return False
    return True


class LlmClient:
    """Thin wrapper over an OpenAI-compatible chat endpoint.

    The client is constructed lazily on first :meth:`chat` so importing this
    module (and constructing the verifier) never requires the ``openai`` SDK or a
    reachable server — only an actual call does.
    """

    def __init__(self, *, timeout: float = 60.0) -> None:
        cfg = load_config()
        self.base_url = cfg.llm_base_url
        self.api_key = cfg.llm_api_key
        self.model = cfg.llm_model
        self.timeout = timeout
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
            )
        return self._client

    def chat(self, system: str, user: str, json_schema: dict | None = None) -> dict:
        """Send a chat completion and return the parsed JSON object.

        ``temperature=0`` for determinism. When ``json_schema`` is given we ask
        the server for JSON-object output (``response_format``) — supported by
        vLLM, Ollama and the OpenAI API alike — and the schema itself is also
        spelled out in the system prompt so a server without strict structured
        output still returns the right shape. Returns ``{}`` if the response is
        not parseable JSON (the caller treats that as an abstain).
        """
        client = self._ensure_client()
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return {}


__all__ = ["LlmClient", "is_available"]
