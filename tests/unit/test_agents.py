"""Unit tests for the sub-agent factory (Layer 5)."""

from __future__ import annotations

from pathlib import Path

from app.agents.factory import (
    build_agent,
    build_all_agents,
    build_guarded_tools,
    get_agent_spec,
    guard_wrap,
)
from app.agents.registry import load_agent_registry
from app.config import Settings
from app.integrations.mock import MockSRE
from app.rag.embeddings import HashingEmbedder
from app.rag.retriever import RunbookRetriever
from app.security.guard import ApprovalContext, PermissionGuard
from app.tools.catalog import build_tools

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "data" / "scenarios" / "checkout-5xx-spike.yaml"
RUNBOOKS = ROOT / "data" / "runbooks"


def _setup():
    env = MockSRE.from_scenario(SCENARIO)
    retriever = RunbookRetriever(HashingEmbedder(), collection="test_agents")
    retriever.index_directory(RUNBOOKS)
    tools = build_tools(env, retriever)
    guard = PermissionGuard()
    ctx = ApprovalContext()
    return tools, guard, ctx


def test_yaml_registry_defines_three_agents() -> None:
    registry = load_agent_registry()
    assert set(registry.agents) == {"triage", "diagnosis", "remediation"}
    assert registry.workflow == ["triage", "diagnosis", "remediation"]
    assert len(registry.agents["triage"].tools) == 4
    assert len(registry.agents["diagnosis"].tools) == 7
    assert "Triage agent" in registry.agents["triage"].prompt


def test_get_agent_spec_unknown_raises() -> None:
    try:
        get_agent_spec("does-not-exist")
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "does-not-exist" in str(exc)


def test_guard_wrap_blocks_critical_tool() -> None:
    tools, guard, ctx = _setup()
    wrapped = guard_wrap(tools["rollback_deploy"], guard, ctx)
    out = wrapped.invoke({"service": "checkout", "to_version": "v1.4.3"})
    assert "cannot run autonomously" in out.lower()


def test_build_guarded_tools_preserves_names() -> None:
    tools, guard, ctx = _setup()
    spec = get_agent_spec("triage")
    guarded = build_guarded_tools(tools, spec.tools, guard, ctx)
    assert {t.name for t in guarded} == set(spec.tools)


def test_build_all_agents_compiles_graphs() -> None:
    """Smoke test: graphs compile (no LLM call until invoke)."""
    tools, guard, ctx = _setup()
    settings = Settings(openai_api_key="sk-test", anthropic_api_key="sk-test")
    agents = build_all_agents(tools, guard, ctx, settings=settings)
    assert len(agents) == 3
    for name, graph in agents.items():
        assert graph.name == name
        assert callable(graph.invoke)

    triage = build_agent("triage", tools, guard, ctx, settings=settings)
    assert triage.name == "triage"
