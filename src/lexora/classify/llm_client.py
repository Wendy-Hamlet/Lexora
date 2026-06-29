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
import time
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

    def __init__(self, *, timeout: float = 60.0, model: str | None = None) -> None:
        cfg = load_config()
        self.base_url = cfg.llm_base_url
        self.api_key = cfg.llm_api_key
        # ``model`` overrides the configured default (e.g. the per-clause verifier
        # pins the reliable non-reasoning deepseek-v4-flash regardless of
        # LEXORA_LLM_MODEL, which may point at a reasoning backend).
        self.model = model or cfg.llm_model
        self.max_tokens = cfg.llm_max_tokens
        self.max_retries = cfg.llm_max_retries
        # Optional User-Agent override. Some hosted OpenAI-compatible gateways sit
        # behind Cloudflare, which rejects the SDK's default UA with HTTP 403
        # "error code: 1010". Set LEXORA_LLM_USER_AGENT to a browser UA to pass.
        # Empty = leave the SDK default (no change for normal endpoints).
        self.user_agent = cfg.llm_user_agent
        self.timeout = timeout
        self._client = None
        # Some endpoints accept response_format=json_object but return empty
        # content (and burn latency) for it; the client then falls back to a
        # plain completion. Set LEXORA_LLM_JSON_MODE=0 to skip the json-mode
        # attempt entirely on such endpoints. Default preserves prior behaviour.
        self.use_json_mode = os.environ.get("LEXORA_LLM_JSON_MODE", "1").lower() not in (
            "0", "false", "no", "off",
        )
        # Backoff (seconds) slept between failed retry attempts, doubled each
        # attempt and capped. Default 0 = retry immediately (our endpoint is not
        # rate-limited, and temperature escalation — not waiting — is what breaks
        # a deterministic bad-JSON reply). Set LEXORA_LLM_RETRY_BACKOFF>0 for a
        # rate-limited endpoint where spacing the retries helps.
        try:
            self.retry_backoff = float(os.environ.get("LEXORA_LLM_RETRY_BACKOFF", "0"))
        except ValueError:
            self.retry_backoff = 0.0
        self.retry_backoff_cap = 8.0
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

            default_headers = (
                {"User-Agent": self.user_agent} if self.user_agent else None
            )
            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout,
                default_headers=default_headers,
            )
        return self._client

    def chat(self, system: str, user: str, json_schema: dict | None = None) -> dict:
        """Send a chat completion and return the parsed JSON object.

        When ``json_schema`` is given the call is retried up to
        ``llm_max_retries`` times until a parseable JSON object comes back, and
        only then does :class:`LlmResponseError` propagate so the caller can
        DEGRADE (the verifier to keep-all, the rationale to its template, the
        metadata extractor to portal-only) instead of silently treating a flaky
        backend reply as a real verdict. A retry is triggered by any of: an API
        exception (5xx / network / a 400 rejecting ``response_format``), an empty
        response, or unparseable content.

        Retries escalate to actually break a *deterministic* bad reply rather
        than re-issue the identical request: the first attempt asks for JSON mode
        (if enabled, ``temperature=0``); the first plain attempt drops JSON mode
        (still ``temperature=0`` for best quality); later plain attempts nudge the
        temperature up (0.2 → 0.6) so a model that deterministically emitted
        malformed JSON at temp 0 gets a fresh sample. ``LEXORA_LLM_RETRY_BACKOFF``
        optionally spaces the attempts for a rate-limited endpoint.
        """
        client = self._ensure_client()
        if json_schema is not None and "json" not in system:
            system = f"{system}\nRespond with a valid json object."
        if json_schema is not None and "json" not in user:
            user = f"{user}\nReturn valid json."
        plan = self._attempt_plan(json_schema)
        content = ""
        last_error: Exception | None = None
        for i, (json_mode, temperature) in enumerate(plan):
            kwargs = self._chat_kwargs(system, user, json_mode=json_mode,
                                       temperature=temperature)
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as exc:
                # Plain (non-schema) calls preserve the old contract: surface the
                # API error to the caller. Schema calls retry the whole request.
                if json_schema is None:
                    raise
                last_error = exc
                self._backoff(i, len(plan))
                continue
            self._account(resp)
            content = resp.choices[0].message.content or ""
            if json_schema is None:
                break
            if not content:
                last_error = LlmResponseError("empty response content")
                self._backoff(i, len(plan))
                continue
            try:
                return _parse_json_object(content)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc
                self._backoff(i, len(plan))
                continue
        if json_schema is None:
            try:
                return _parse_json_object(content)
            except (json.JSONDecodeError, TypeError, ValueError):
                return {}
        raise LlmResponseError(
            f"no parseable response after {len(plan)} attempt(s)"
        ) from last_error

    def _attempt_plan(self, json_schema: dict | None) -> list[tuple[bool, float]]:
        """Ordered (json_mode, temperature) attempts for one :meth:`chat` call.

        Plain calls get a single attempt. Schema calls get an optional JSON-mode
        attempt followed by ``max(1, llm_max_retries)`` plain attempts whose
        temperature ramps after the first so retries are not identical repeats.
        """
        if json_schema is None:
            return [(False, 0.0)]
        plan: list[tuple[bool, float]] = []
        if self.use_json_mode:
            plan.append((True, 0.0))
        n_plain = max(1, self.max_retries)
        for k in range(n_plain):
            temperature = 0.0 if k == 0 else min(0.2 * k, 0.6)
            plan.append((False, temperature))
        return plan

    def _backoff(self, attempt: int, total: int) -> None:
        """Sleep before the next retry (no sleep after the final attempt)."""
        if self.retry_backoff > 0 and attempt < total - 1:
            time.sleep(min(self.retry_backoff * (2 ** attempt), self.retry_backoff_cap))

    def _chat_kwargs(self, system: str, user: str, *, json_mode: bool,
                     temperature: float = 0.0) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
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
