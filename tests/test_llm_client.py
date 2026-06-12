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
