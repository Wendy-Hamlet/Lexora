"""The replayed/live label. A demonstration that is not labelled is a misrepresentation,
so these tests pin the two directions: a replayed run says so everywhere, and a live run
never claims to be a demonstration."""
from __future__ import annotations

from pathlib import Path

import pytest

from lexora.export import provenance
from lexora.export.html_exporter import to_html


@pytest.fixture
def replaying(monkeypatch):
    monkeypatch.setenv("LEXORA_HTTP_CACHE", "replay")


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setenv("LEXORA_HTTP_CACHE", "off")


def test_replay_is_a_demonstration(replaying):
    assert provenance.is_demonstration()
    assert provenance.provenance()["demonstration"] is True


def test_live_run_is_not_a_demonstration(live):
    assert not provenance.is_demonstration()
    assert provenance.provenance()["demonstration"] is False


def test_recording_is_live_not_a_demonstration(monkeypatch):
    # Recording still fetches from the source, so its output IS current and submittable.
    monkeypatch.setenv("LEXORA_HTTP_CACHE", "record")
    assert not provenance.is_demonstration()
    assert provenance.provenance()["network"] == "live (recorded)"


def test_warm_verdict_cache_alone_is_not_a_demonstration(live):
    # The judge cache is keyed on the rendered prompt and the clause text, so a hit is the
    # same model answering the same question -- a submission run may use it.
    p = provenance.provenance(judged=0, from_cache=900)
    assert p["demonstration"] is False
    assert p["verdict_cache_share"] == 1.0


def test_label_prefixes_only_when_replaying(replaying):
    assert provenance.label("Singapore_P6.csv") == "DEMO_Singapore_P6.csv"


def test_label_is_idempotent(replaying):
    once = provenance.label("x.csv")
    assert provenance.label(once) == once


def test_label_untouched_on_a_live_run(live):
    assert provenance.label("Singapore_P6.csv") == "Singapore_P6.csv"


def test_console_notice_leads_with_the_banner_when_replaying(replaying):
    assert provenance.BANNER in provenance.console_notice()


def test_console_notice_states_live_provenance_too(live):
    # Always saying something is what gives the demonstration banner its meaning.
    notice = provenance.console_notice(judged=10, from_cache=2)
    assert provenance.BANNER not in notice
    assert "live" in notice and "10 clause(s) judged live" in notice


def test_html_carries_the_banner_when_replaying(replaying, tmp_path: Path):
    out = tmp_path / "r.html"
    to_html([], out)
    assert provenance.BANNER in out.read_text(encoding="utf-8")


def test_html_has_no_banner_on_a_live_run(live, tmp_path: Path):
    out = tmp_path / "r.html"
    to_html([], out)
    assert provenance.BANNER not in out.read_text(encoding="utf-8")


def test_config_swap_reaches_every_llm_lane(monkeypatch):
    """The No-Vendor-Lock-in rubric tests one thing: 'can swap OpenAI API for Llama 3
    (self-hosted) by changing a config value, not rewriting pipelines'. So the config value
    has to reach EVERY lane -- including the per-clause judge, which produces essentially
    all of our output and used to default to a hardcoded vendor model name regardless."""
    from types import SimpleNamespace

    import lexora.classify.llm_client as llm_client_mod
    import lexora.config as cfgmod
    from lexora.classify.verifier import make_verifier

    # `llm_client` binds load_config at import time, so patch it where it is USED.
    monkeypatch.setattr(cfgmod, "env_value", lambda name, default=None: default)
    monkeypatch.setattr(llm_client_mod, "load_config", lambda: SimpleNamespace(
        llm_base_url="http://localhost:11434/v1", llm_api_key="none", llm_model="llama3",
        llm_max_tokens=512, llm_max_retries=1, llm_user_agent="",
    ))
    for mode in ("pick_one", "per_cell", "per_clause"):
        v = make_verifier(use_llm=True, mode=mode)
        assert v is not None and v._client.model == "llama3", mode
