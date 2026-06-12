"""Runtime configuration loading."""
from __future__ import annotations

from lexora.config import load_config


def test_load_config_reads_dotenv_and_openai_aliases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEXORA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LEXORA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LEXORA_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("LEXORA_ENV_FILE", raising=False)

    tmp_path.joinpath(".env.example").write_text(
        "LEXORA_LLM_BASE_URL=http://localhost:8000/v1\n"
        "LEXORA_LLM_API_KEY=not-needed\n"
        "LEXORA_LLM_MODEL=example-model\n",
        encoding="utf-8",
    )
    tmp_path.joinpath(".env").write_text(
        "OPENAI_BASE_URL=https://llm.example/v1\n"
        "OPENAI_API_KEY=test-key\n",
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.llm_base_url == "https://llm.example/v1"
    assert cfg.llm_api_key == "test-key"
    assert cfg.llm_model == "example-model"
    assert cfg.llm_max_tokens == 512
    assert cfg.llm_max_retries == 2


def test_process_environment_overrides_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LEXORA_LLM_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("LEXORA_LLM_API_KEY", "env-key")
    monkeypatch.setenv("LEXORA_LLM_MODEL", "env-model")
    monkeypatch.setenv("LEXORA_LLM_MAX_TOKENS", "128")
    monkeypatch.setenv("LEXORA_LLM_MAX_RETRIES", "3")
    monkeypatch.delenv("LEXORA_ENV_FILE", raising=False)

    tmp_path.joinpath(".env").write_text(
        "OPENAI_BASE_URL=https://dotenv.example/v1\n"
        "OPENAI_API_KEY=dotenv-key\n"
        "LEXORA_LLM_MODEL=dotenv-model\n",
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.llm_base_url == "https://env.example/v1"
    assert cfg.llm_api_key == "env-key"
    assert cfg.llm_model == "env-model"
    assert cfg.llm_max_tokens == 128
    assert cfg.llm_max_retries == 3


def test_explicit_env_file_loads_sibling_example_defaults(tmp_path, monkeypatch):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    monkeypatch.delenv("LEXORA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LEXORA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LEXORA_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    env_dir = tmp_path / "config"
    env_dir.mkdir()
    env_dir.joinpath(".env.example").write_text(
        "LEXORA_LLM_MODEL=sibling-example-model\n", encoding="utf-8"
    )
    env_path = env_dir / ".env"
    env_path.write_text(
        "OPENAI_BASE_URL=https://explicit.example/v1\n"
        "OPENAI_API_KEY=explicit-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LEXORA_ENV_FILE", str(env_path))

    cfg = load_config()

    assert cfg.llm_base_url == "https://explicit.example/v1"
    assert cfg.llm_api_key == "explicit-key"
    assert cfg.llm_model == "sibling-example-model"
    assert cfg.llm_max_tokens == 512
    assert cfg.llm_max_retries == 2


def test_env_example_only_supplies_llm_model_not_runtime_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEXORA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LEXORA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LEXORA_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("LEXORA_OCR_LANG", raising=False)
    monkeypatch.delenv("LEXORA_ENV_FILE", raising=False)

    tmp_path.joinpath(".env.example").write_text(
        "LEXORA_LLM_BASE_URL=https://example-should-not-win/v1\n"
        "LEXORA_LLM_API_KEY=example-should-not-win\n"
        "LEXORA_LLM_MODEL=example-model\n"
        "LEXORA_OCR_LANG=eng+jpn+tha+chi_sim\n",
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.llm_base_url == "http://localhost:8000/v1"
    assert cfg.llm_api_key == "not-needed"
    assert cfg.llm_model == "example-model"
    assert cfg.llm_max_tokens == 512
    assert cfg.llm_max_retries == 2
    assert cfg.ocr_lang == "eng"


def test_openai_max_tokens_alias_in_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEXORA_LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)
    monkeypatch.delenv("LEXORA_ENV_FILE", raising=False)

    tmp_path.joinpath(".env").write_text(
        "OPENAI_MAX_TOKENS=64\n",
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.llm_max_tokens == 64
