"""Unit tests for the tool layer (Layer 3).

Fully offline: mock SRE env + offline retriever. Verifies that tools invoke
correctly, format readable output, and carry the right harm-tier metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.models import HarmTier
from app.integrations.mock import MockSRE
from app.rag.embeddings import HashingEmbedder
from app.rag.retriever import RunbookRetriever
from app.tools.catalog import (
    DIAGNOSIS_TOOLS,
    REMEDIATION_TOOLS,
    TRIAGE_TOOLS,
    build_tools,
    select,
    tool_harm_tier,
)

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "data" / "scenarios" / "checkout-5xx-spike.yaml"
RUNBOOKS = ROOT / "data" / "runbooks"


@pytest.fixture
def tools():
    env = MockSRE.from_scenario(SCENARIO)
    retriever = RunbookRetriever(HashingEmbedder(), collection="test_tools")
    retriever.index_directory(RUNBOOKS)
    return build_tools(env, retriever)


def test_harm_tiers_are_tagged(tools) -> None:
    assert tool_harm_tier(tools["query_logs"]) is HarmTier.READ_ONLY
    assert tool_harm_tier(tools["rerun_ci_job"]) is HarmTier.COMPENSABLE
    assert tool_harm_tier(tools["restart_service"]) is HarmTier.IRREVERSIBLE
    assert tool_harm_tier(tools["rollback_deploy"]) is HarmTier.CRITICAL
    # The dangerous ones are the ones that need approval.
    assert tool_harm_tier(tools["rollback_deploy"]).needs_human_approval is True
    assert tool_harm_tier(tools["query_logs"]).needs_human_approval is False


def test_tool_has_name_and_description(tools) -> None:
    t = tools["query_logs"]
    assert t.name == "query_logs"
    assert "logs" in t.description.lower()


def test_query_logs_tool_invocation(tools) -> None:
    out = tools["query_logs"].invoke(
        {"service": "checkout", "query": "NullPointer", "since_minutes": 60}
    )
    assert "NullPointer" in out


def test_search_runbooks_tool_grounds_in_the_right_doc(tools) -> None:
    out = tools["search_runbooks"].invoke(
        {"query": "checkout 500 NullPointerException after deploy"}
    )
    assert "Checkout 5xx / Payment Errors" in out
    assert "roll back" in out.lower()


def test_write_tool_executes(tools) -> None:
    out = tools["rollback_deploy"].invoke(
        {"service": "checkout", "to_version": "v1.4.3"}
    )
    assert "rolled back checkout to v1.4.3" in out.lower()


def test_invalid_status_is_rejected(tools) -> None:
    out = tools["update_incident_status"].invoke(
        {"incident_id": "INC-1001", "status": "bogus"}
    )
    assert "Invalid status" in out


def test_agent_tool_selections_exist(tools) -> None:
    for names in (TRIAGE_TOOLS, DIAGNOSIS_TOOLS, REMEDIATION_TOOLS):
        selected = select(tools, names)
        assert len(selected) == len(names)
    # Remediation owns the dangerous tools; triage/diagnosis are read-only.
    triage = select(tools, TRIAGE_TOOLS)
    assert all(tool_harm_tier(t) is HarmTier.READ_ONLY for t in triage)
    remediation = select(tools, REMEDIATION_TOOLS)
    assert any(tool_harm_tier(t) is HarmTier.CRITICAL for t in remediation)
