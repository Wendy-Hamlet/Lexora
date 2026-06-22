"""Runtime configuration loaded from process environment and local .env files."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULTS = {
    "LEXORA_DB_URL": "sqlite:///data/lexora.sqlite",
    "LEXORA_OBJECT_STORE": "./data/raw",
    "LEXORA_LLM_BASE_URL": "http://localhost:8000/v1",
    "LEXORA_LLM_API_KEY": "not-needed",
    "LEXORA_LLM_MODEL": "Qwen/Qwen2.5-7B-Instruct",
    "LEXORA_LLM_MAX_TOKENS": "512",
    "LEXORA_LLM_MAX_RETRIES": "2",
    "LEXORA_EMBEDDING_MODEL": "BAAI/bge-m3",
    "LEXORA_OCR_LANG": "eng",
    "LEXORA_OCR_CITABLE_THRESHOLD": "0.85",
    "LEXORA_API_HOST": "127.0.0.1",
    "LEXORA_API_PORT": "8001",
}


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or not name.replace("_", "").isalnum() or name[0].isdigit():
            continue
        values[name] = _strip_env_value(value)
    return values


def _candidate_env_files(filename: str) -> list[Path]:
    """Return matching dotenv files from lower to higher precedence."""
    paths: list[Path] = []
    seen: set[Path] = set()

    cwd = Path.cwd().resolve()
    for directory in reversed((cwd, *cwd.parents)):
        path = directory / filename
        if path not in seen:
            paths.append(path)
            seen.add(path)

    explicit = os.environ.get("LEXORA_ENV_FILE")
    if explicit:
        explicit_path = Path(explicit).expanduser().resolve()
        path = explicit_path if filename == ".env" else explicit_path.with_name(filename)
        if path not in seen:
            paths.append(path)
            seen.add(path)

    return paths


def _dotenv_layers(filename: str) -> list[dict[str, str]]:
    layers: list[dict[str, str]] = []
    for path in _candidate_env_files(filename):
        values = _parse_env_file(path)
        if values:
            layers.append(values)
    return layers


def _example_model_layers() -> list[dict[str, str]]:
    layers: list[dict[str, str]] = []
    allowed = {"LEXORA_LLM_MODEL", "OPENAI_MODEL"}
    for values in _dotenv_layers(".env.example"):
        model_values = {key: value for key, value in values.items() if key in allowed}
        if model_values:
            layers.append(model_values)
    return layers


def _env(layers: list[dict[str, str]], name: str, default: str, *aliases: str) -> str:
    for key in (name, *aliases):
        value = os.environ.get(key)
        if value:
            return value
    for values in reversed(layers):
        for key in (name, *aliases):
            value = values.get(key)
            if value:
                return value
    return default


@dataclass(frozen=True)
class LexoraConfig:
    db_url: str
    object_store: Path
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_max_tokens: int
    llm_max_retries: int
    llm_user_agent: str
    embedding_model: str
    ocr_lang: str
    ocr_citable_threshold: float
    api_host: str
    api_port: int


def load_config() -> LexoraConfig:
    layers = _dotenv_layers(".env")
    model_layers = _example_model_layers() + layers
    return LexoraConfig(
        db_url=_env(layers, "LEXORA_DB_URL", _DEFAULTS["LEXORA_DB_URL"]),
        object_store=Path(
            _env(layers, "LEXORA_OBJECT_STORE", _DEFAULTS["LEXORA_OBJECT_STORE"])
        ),
        llm_base_url=_env(
            layers, "LEXORA_LLM_BASE_URL", _DEFAULTS["LEXORA_LLM_BASE_URL"],
            "OPENAI_BASE_URL",
        ),
        llm_api_key=_env(
            layers, "LEXORA_LLM_API_KEY", _DEFAULTS["LEXORA_LLM_API_KEY"],
            "OPENAI_API_KEY",
        ),
        llm_model=_env(
            model_layers, "LEXORA_LLM_MODEL", _DEFAULTS["LEXORA_LLM_MODEL"],
            "OPENAI_MODEL",
        ),
        llm_max_tokens=int(
            _env(
                layers,
                "LEXORA_LLM_MAX_TOKENS",
                _DEFAULTS["LEXORA_LLM_MAX_TOKENS"],
                "OPENAI_MAX_TOKENS",
            )
        ),
        llm_max_retries=int(
            _env(
                layers,
                "LEXORA_LLM_MAX_RETRIES",
                _DEFAULTS["LEXORA_LLM_MAX_RETRIES"],
                "OPENAI_MAX_RETRIES",
            )
        ),
        llm_user_agent=_env(layers, "LEXORA_LLM_USER_AGENT", ""),
        embedding_model=_env(
            layers, "LEXORA_EMBEDDING_MODEL", _DEFAULTS["LEXORA_EMBEDDING_MODEL"]
        ),
        ocr_lang=_env(layers, "LEXORA_OCR_LANG", _DEFAULTS["LEXORA_OCR_LANG"]),
        ocr_citable_threshold=float(
            _env(
                layers,
                "LEXORA_OCR_CITABLE_THRESHOLD",
                _DEFAULTS["LEXORA_OCR_CITABLE_THRESHOLD"],
            )
        ),
        api_host=_env(layers, "LEXORA_API_HOST", _DEFAULTS["LEXORA_API_HOST"]),
        api_port=int(_env(layers, "LEXORA_API_PORT", _DEFAULTS["LEXORA_API_PORT"])),
    )
