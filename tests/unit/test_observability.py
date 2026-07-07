"""Unit tests for observability wiring (Layer 7). All offline, no keys required."""

from __future__ import annotations

from app.config import Settings
from app.observability.tracing import (
    configure_langsmith,
    get_callbacks,
    get_langfuse_handler,
    runnable_config,
    setup_observability,
)


def _disabled_settings() -> Settings:
    return Settings(langsmith_tracing=False, langfuse_enabled=False)


def test_langsmith_disabled_when_no_key(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert configure_langsmith(_disabled_settings()) is False


def test_langsmith_enabled_sets_env(monkeypatch) -> None:
    # Pre-register the vars so monkeypatch restores them on teardown even though
    # configure_langsmith writes to os.environ directly (no leakage across tests).
    for var in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"):
        monkeypatch.setenv(var, "")
    settings = Settings(
        langsmith_tracing=True,
        langsmith_api_key="ls-test",
        langsmith_project="proj-x",
    )
    assert configure_langsmith(settings) is True
    import os

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-test"
    assert os.environ["LANGSMITH_PROJECT"] == "proj-x"


def test_langfuse_handler_none_when_disabled() -> None:
    assert get_langfuse_handler(_disabled_settings()) is None


def test_get_callbacks_empty_when_disabled() -> None:
    assert get_callbacks(_disabled_settings()) == []


def test_setup_observability_returns_list() -> None:
    assert setup_observability(_disabled_settings()) == []


def test_runnable_config_has_thread_and_no_callbacks_when_disabled() -> None:
    cfg = runnable_config("INC-1001", settings=_disabled_settings())
    assert cfg["configurable"]["thread_id"] == "INC-1001"
    assert cfg["recursion_limit"] == 50
    assert "callbacks" not in cfg


def test_runnable_config_includes_extra_callbacks() -> None:
    sentinel = object()
    cfg = runnable_config(
        "INC-2", settings=_disabled_settings(), extra_callbacks=[sentinel]
    )
    assert cfg["callbacks"] == [sentinel]
