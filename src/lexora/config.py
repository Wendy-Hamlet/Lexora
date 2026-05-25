"""Runtime configuration loaded from environment variables (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class LexoraConfig:
    db_url: str = _env("LEXORA_DB_URL", "sqlite:///data/lexora.sqlite") or ""
    object_store: Path = Path(_env("LEXORA_OBJECT_STORE", "./data/raw") or "./data/raw")
    llm_base_url: str = _env("LEXORA_LLM_BASE_URL", "http://localhost:8000/v1") or ""
    llm_api_key: str = _env("LEXORA_LLM_API_KEY", "not-needed") or ""
    llm_model: str = _env("LEXORA_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct") or ""
    embedding_model: str = _env("LEXORA_EMBEDDING_MODEL", "BAAI/bge-m3") or ""
    ocr_lang: str = _env("LEXORA_OCR_LANG", "eng") or "eng"
    ocr_citable_threshold: float = float(_env("LEXORA_OCR_CITABLE_THRESHOLD", "0.85") or 0.85)
    api_host: str = _env("LEXORA_API_HOST", "127.0.0.1") or "127.0.0.1"
    api_port: int = int(_env("LEXORA_API_PORT", "8001") or 8001)


def load_config() -> LexoraConfig:
    return LexoraConfig()
