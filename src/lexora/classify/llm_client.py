"""OpenAI-compatible LLM client.

Works with any OpenAI-compatible endpoint:
    - vLLM (recommended for hackathon demo)
    - llama.cpp server
    - Ollama
    - OpenAI API itself

Reads endpoint + key + model from ``LEXORA_LLM_*`` env vars or local dotenv
files. ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` are supported as compatibility
aliases.

Optional-dependency contract (mirrors :mod:`lexora.semantic.embedder`):
:func:`is_available` reports whether the ``openai`` SDK is importable, and the
verifier falls back to a no-op (keyword/BM25 + verbatim only) when it is not. A
bare install and the offline test suite never require this package or a server.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

from lexora.config import load_config


class LlmResponseError(RuntimeError):
    """Raised when the endpoint returns no parseable verifier JSON."""


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
        self.max_tokens = cfg.llm_max_tokens
        self.max_retries = cfg.llm_max_retries
        self.timeout = timeout
        self._client = None
        # Some endpoints accept response_format=json_object but return empty
        # content (and burn latency) for it; the client then falls back to a
        # plain completion. Set LEXORA_LLM_JSON_MODE=0 to skip the json-mode
        # attempt entirely on such endpoints. Default preserves prior behaviour.
        self.use_json_mode = os.environ.get("LEXORA_LLM_JSON_MODE", "1").lower() not in (
            "0", "false", "no", "off",
        )
        # Token accounting (summed across attempts and retries) so a run can
        # report cost. A lock keeps the counters exact when one client is shared
        # across a thread pool (LLM-call-layer parallelism). Reset by the caller
        # between phases if desired.
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._account_lock = threading.Lock()

    def _account(self, resp) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        with self._account_lock:
            self.calls += 1
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.total_tokens += getattr(usage, "total_tokens", 0) or 0

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
            )
        return self._client

    def chat(self, system: str, user: str, json_schema: dict | None = None) -> dict:
        """Send a chat completion and return the parsed JSON object.

        ``temperature=0`` for determinism. When ``json_schema`` is given, the
        client first asks the server for JSON-object output
        (``response_format``). Some OpenAI-compatible endpoints reject or ignore
        that mode, so the client retries once without ``response_format`` while
        keeping the prompt constrained to JSON. If the structured response is
        still empty or not parseable, :class:`LlmResponseError` is raised so the
        verifier can count and report a backend response error instead of
        silently treating it as a valid abstention.
        """
        client = self._ensure_client()
        if json_schema is not None and "json" not in system:
            system = f"{system}\nRespond with a valid json object."
        if json_schema is not None and "json" not in user:
            user = f"{user}\nReturn valid json."
        attempts: list[bool] = []
        if json_schema is not None:
            if self.use_json_mode:
                attempts.append(True)
            attempts.extend([False] * max(1, self.max_retries))
        else:
            attempts.append(False)
        content = ""
        parse_error: Exception | None = None
        for json_mode in attempts:
            kwargs = self._chat_kwargs(system, user, json_mode=json_mode)
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as exc:
                if json_schema is None or getattr(exc, "status_code", None) != 400:
                    raise
                parse_error = exc
                continue
            self._account(resp)
            content = resp.choices[0].message.content or ""
            if not content and json_schema is not None:
                continue
            if json_schema is None:
                break
            try:
                return _parse_json_object(content)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                parse_error = exc
                continue
        if json_schema is not None and not content:
            raise LlmResponseError("empty response content") from parse_error
        try:
            return _parse_json_object(content)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            if json_schema is not None:
                raise LlmResponseError("response content is not valid JSON") from exc
            return {}

    def _chat_kwargs(self, system: str, user: str, *, json_mode: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_completion_tokens": self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs


def _parse_json_object(content: str) -> dict:
    text = content.strip()
    if not text:
        raise ValueError("empty content")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.IGNORECASE | re.DOTALL)
        if fenced:
            data = json.loads(fenced.group(1))
        else:
            start = text.find("{")
            if start < 0:
                raise
            decoder = json.JSONDecoder()
            data, _ = decoder.raw_decode(text[start:])
    if not isinstance(data, dict):
        raise ValueError("JSON response is not an object")
    return data


__all__ = ["LlmClient", "LlmResponseError", "is_available"]
