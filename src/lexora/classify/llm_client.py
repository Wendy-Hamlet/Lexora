"""OpenAI-compatible LLM client.

Works with any OpenAI-compatible endpoint:
    - vLLM (recommended for hackathon demo)
    - llama.cpp server
    - Ollama
    - OpenAI API itself

Reads endpoint + key + model from LEXORA_LLM_* env vars (see config.py).
"""
from __future__ import annotations

from lexora.config import load_config


class LlmClient:
    def __init__(self) -> None:
        cfg = load_config()
        self.base_url = cfg.llm_base_url
        self.api_key = cfg.llm_api_key
        self.model = cfg.llm_model

    def chat(self, system: str, user: str, json_schema: dict | None = None) -> dict:  # pragma: no cover
        """Send a chat completion request. If json_schema is provided, request
        structured output enforcing that schema.

        TODO: implement with the `openai` Python SDK (works for vLLM with
        base_url override).
        """
        raise NotImplementedError("Implement LLM client.")
