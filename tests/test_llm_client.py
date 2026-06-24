"""OpenAI-compatible LLM client behavior."""
from __future__ import annotations

import pytest

from lexora.classify.llm_client import LlmClient, LlmResponseError


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, fallback_content: str | list[str] = '{"ok": true}'):
        self.calls = []
        self.fallback_content = fallback_content

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("response_format"):
            return _Response("")
        if isinstance(self.fallback_content, list):
            content = self.fallback_content.pop(0)
        else:
            content = self.fallback_content
        return _Response(content)


class _Chat:
    def __init__(self, fallback_content: str | list[str] = '{"ok": true}'):
        self.completions = _Completions(fallback_content)


class _FakeOpenAI:
    def __init__(self, fallback_content: str | list[str] = '{"ok": true}'):
        self.chat = _Chat(fallback_content)


def test_json_mode_can_be_disabled_to_skip_the_dead_attempt(monkeypatch):
    # Some endpoints return empty content for response_format=json_object (and burn
    # latency). LEXORA_LLM_JSON_MODE=0 skips that attempt entirely.
    monkeypatch.setenv("LEXORA_LLM_JSON_MODE", "0")
    fake = _FakeOpenAI()
    client = LlmClient()
    client._client = fake

    data = client.chat("Return json.", 'Return {"ok": true}.', json_schema={"type": "object"})

    assert data == {"ok": True}
    calls = fake.chat.completions.calls
    assert all("response_format" not in c for c in calls)  # json mode never attempted


class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = p + c


def test_token_accounting_sums_usage(monkeypatch):
    monkeypatch.setenv("LEXORA_LLM_JSON_MODE", "0")
    fake = _FakeOpenAI()

    def _create(**kwargs):
        fake.chat.completions.calls.append(kwargs)
        resp = _Response('{"ok": true}')
        resp.usage = _Usage(40, 10)
        return resp

    fake.chat.completions.create = _create
    client = LlmClient()
    client._client = fake

    client.chat("Return json.", "Return json.", json_schema={"type": "object"})
    assert client.calls == 1
    assert client.prompt_tokens == 40
    assert client.completion_tokens == 10
    assert client.total_tokens == 50


def test_chat_retries_without_json_mode_when_endpoint_returns_empty_content():
    fake = _FakeOpenAI()
    client = LlmClient()
    client._client = fake

    data = client.chat(
        "Return only a JSON object.",
        'Return {"ok": true}.',
        json_schema={"type": "object"},
    )

    assert data == {"ok": True}
    calls = fake.chat.completions.calls
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["max_completion_tokens"] == client.max_tokens
    assert "response_format" not in calls[1]
    assert calls[1]["max_completion_tokens"] == client.max_tokens
    assert "json" in calls[1]["messages"][0]["content"]
    assert "json" in calls[1]["messages"][1]["content"]


def test_chat_retries_transient_empty_plain_responses():
    fake = _FakeOpenAI(fallback_content=["", '{"ok": true}'])
    client = LlmClient()
    client._client = fake

    data = client.chat("Return json.", "Return json.", json_schema={"type": "object"})

    assert data == {"ok": True}
    calls = fake.chat.completions.calls
    assert len(calls) == 3
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]
    assert "response_format" not in calls[2]


def test_chat_raises_when_endpoint_returns_no_parseable_json():
    fake = _FakeOpenAI(fallback_content="")
    client = LlmClient()
    client._client = fake

    with pytest.raises(LlmResponseError):
        client.chat("Return json.", "Return json.", json_schema={"type": "object"})


def test_chat_retries_api_exception_then_succeeds(monkeypatch):
    # A transient API exception (5xx / network) must be retried, not surfaced, as
    # long as a later attempt returns a parseable object.
    monkeypatch.setenv("LEXORA_LLM_JSON_MODE", "0")
    fake = _FakeOpenAI()
    calls = {"n": 0}

    def _create(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 service unavailable")
        return _Response('{"ok": true}')

    fake.chat.completions.create = _create
    client = LlmClient()
    client._client = fake

    assert client.chat("Return json.", "Return json.",
                       json_schema={"type": "object"}) == {"ok": True}
    assert calls["n"] == 3  # two failures retried, third succeeded


def test_chat_raises_after_exhausting_retries_on_api_exception(monkeypatch):
    monkeypatch.setenv("LEXORA_LLM_JSON_MODE", "0")
    fake = _FakeOpenAI()

    def _create(**kwargs):
        raise RuntimeError("503 service unavailable")

    fake.chat.completions.create = _create
    client = LlmClient()
    client._client = fake

    with pytest.raises(LlmResponseError):
        client.chat("Return json.", "Return json.", json_schema={"type": "object"})


def test_chat_escalates_temperature_across_retries(monkeypatch):
    # The JSON-mode attempt and the first plain attempt both stay at temp 0 (best
    # quality); later identical-prompt retries ramp the temperature so a model
    # that deterministically emitted bad JSON at temp 0 gets a fresh sample.
    monkeypatch.setenv("LEXORA_LLM_JSON_MODE", "1")
    # plain attempts pop these in turn (the json-mode attempt returns "" and does
    # not pop); two empties then a valid object -> success on the 3rd plain call.
    fake = _FakeOpenAI(fallback_content=["", "", '{"ok": true}'])
    client = LlmClient()
    client._client = fake

    assert client.chat("Return json.", "Return json.",
                       json_schema={"type": "object"}) == {"ok": True}
    temps = [c["temperature"] for c in fake.chat.completions.calls]
    assert temps[0] == 0.0  # json-mode attempt
    assert temps[1] == 0.0  # first plain attempt
    assert temps[2] > 0.0   # later retry nudges temperature up


def test_plain_chat_still_surfaces_api_exception():
    # Non-schema calls keep the old contract: the API error propagates.
    fake = _FakeOpenAI()

    def _create(**kwargs):
        raise RuntimeError("boom")

    fake.chat.completions.create = _create
    client = LlmClient()
    client._client = fake

    with pytest.raises(RuntimeError):
        client.chat("hi", "there")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('```json\n{"ok": true}\n```', {"ok": True}),
        ('Here is the json object: {"ok": true}', {"ok": True}),
    ],
)
def test_chat_extracts_json_object_from_wrapped_content(content, expected):
    fake = _FakeOpenAI(fallback_content=content)
    client = LlmClient()
    client._client = fake

    assert client.chat("Return json.", "Return json.", json_schema={"type": "object"}) == expected
