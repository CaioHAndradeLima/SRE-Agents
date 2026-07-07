"""Unit tests for the composition root (Layer 8).

Offline: uses the HashingEmbedder and fake gateway keys. Graphs/models are *built*
(not invoked), so no network or LLM call happens — construction alone verifies the
wiring across every layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.registry import load_agent_registry
from app.composition import (
    build_copilot,
    build_retriever,
    build_sre_environment,
    compute_sensitive_agents,
)
from app.config import Settings
from app.integrations.mock import MockSRE
from app.rag.embeddings import HashingEmbedder
from app.tools.catalog import build_tools

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "data" / "scenarios" / "checkout-5xx-spike.yaml"
RUNBOOKS = ROOT / "data" / "runbooks"


def _settings() -> Settings:
    return Settings(openai_api_key="sk-test", anthropic_api_key="sk-test")


def test_build_sre_environment_local_returns_mock() -> None:
    env = build_sre_environment(_settings(), SCENARIO)
    assert isinstance(env, MockSRE)


def test_build_sre_environment_prod_not_implemented() -> None:
    settings = Settings(app_env="prod")
    with pytest.raises(NotImplementedError):
        build_sre_environment(settings, SCENARIO)


def test_build_retriever_indexes_runbooks() -> None:
    retriever = build_retriever(
        _settings(), runbooks_dir=RUNBOOKS, embedder=HashingEmbedder()
    )
    hits = retriever.search("checkout 500 NullPointerException", k=1)
    assert hits[0].source == "checkout-payment-5xx.md"


def test_compute_sensitive_agents_flags_remediation() -> None:
    env = MockSRE.from_scenario(SCENARIO)
    retriever = build_retriever(
        _settings(), runbooks_dir=RUNBOOKS, embedder=HashingEmbedder()
    )
    tool_map = build_tools(env, retriever)
    sensitive = compute_sensitive_agents(tool_map, load_agent_registry())
    assert "remediation" in sensitive
    assert "triage" not in sensitive
    assert "diagnosis" not in sensitive


def test_build_copilot_wires_all_layers() -> None:
    copilot = build_copilot(
        SCENARIO,
        settings=_settings(),
        runbooks_dir=RUNBOOKS,
        embedder=HashingEmbedder(),
    )
    assert callable(copilot.graph.invoke)
    assert copilot.sensitive_agents == {"remediation"}
    assert copilot.approval_ctx.approve_all is False
    assert isinstance(copilot.env, MockSRE)
