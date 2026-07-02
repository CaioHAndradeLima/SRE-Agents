"""Unit tests for the permission guard (Layer 4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.integrations.mock import MockSRE
from app.rag.embeddings import HashingEmbedder
from app.rag.retriever import RunbookRetriever
from app.security.guard import ApprovalContext, PermissionGuard
from app.security.policy import PermissionDecision
from app.tools.catalog import build_tools

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "data" / "scenarios" / "checkout-5xx-spike.yaml"
RUNBOOKS = ROOT / "data" / "runbooks"


@pytest.fixture
def tools():
    env = MockSRE.from_scenario(SCENARIO)
    retriever = RunbookRetriever(HashingEmbedder(), collection="test_guard")
    retriever.index_directory(RUNBOOKS)
    return build_tools(env, retriever)


@pytest.fixture
def guard() -> PermissionGuard:
    return PermissionGuard()


@pytest.fixture
def ctx() -> ApprovalContext:
    return ApprovalContext()


def test_read_only_tool_runs_without_approval(guard, tools, ctx) -> None:
    tool = tools["query_logs"]
    args = {"service": "checkout", "query": "NullPointer", "since_minutes": 60}
    assert guard.evaluate(tool, args, ctx) is PermissionDecision.ALLOW
    out = guard.invoke(tool, args, ctx)
    assert "NullPointer" in out
    assert guard.audit_log[-1].executed is True


def test_reversible_tool_runs_with_log(guard, tools, ctx) -> None:
    tool = tools["post_incident_note"]
    args = {"incident_id": "INC-1001", "text": "investigating spike"}
    assert guard.evaluate(tool, args, ctx) is PermissionDecision.ALLOW_WITH_LOG
    guard.invoke(tool, args, ctx)
    assert guard.audit_log[-1].decision is PermissionDecision.ALLOW_WITH_LOG


def test_compensable_blocked_until_confirmed(guard, tools, ctx) -> None:
    tool = tools["rerun_ci_job"]
    args = {"run_id": "RUN-501"}
    assert guard.evaluate(tool, args, ctx) is PermissionDecision.NEEDS_CONFIRMATION
    blocked = guard.invoke(tool, args, ctx)
    assert "confirmation" in blocked.lower()
    assert guard.audit_log[-1].executed is False
    guard.confirm(tool, args, ctx)
    out = guard.invoke(tool, args, ctx)
    assert "re-triggered" in out.lower()


def test_irreversible_blocked_until_human_approval(guard, tools, ctx) -> None:
    tool = tools["restart_service"]
    args = {"service": "checkout"}
    assert guard.evaluate(tool, args, ctx) is PermissionDecision.NEEDS_HUMAN_APPROVAL
    blocked = guard.invoke(tool, args, ctx)
    assert "human approval" in blocked.lower()
    guard.approve(tool, args, ctx)
    out = guard.invoke(tool, args, ctx)
    assert "restarted" in out.lower()


def test_critical_is_propose_only_without_approval(guard, tools, ctx) -> None:
    tool = tools["rollback_deploy"]
    args = {"service": "checkout", "to_version": "v1.4.3"}
    assert guard.evaluate(tool, args, ctx) is PermissionDecision.PROPOSE_ONLY
    blocked = guard.invoke(tool, args, ctx)
    assert "cannot run autonomously" in blocked.lower()
    assert guard.audit_log[-1].executed is False


def test_critical_runs_after_explicit_approval(guard, tools, ctx) -> None:
    tool = tools["rollback_deploy"]
    args = {"service": "checkout", "to_version": "v1.4.3"}
    guard.approve(tool, args, ctx)
    assert guard.evaluate(tool, args, ctx) is PermissionDecision.ALLOW
    out = guard.invoke(tool, args, ctx)
    assert "rolled back" in out.lower()


def test_audit_log_records_every_decision(guard, tools, ctx) -> None:
    guard.invoke(
        tools["query_logs"],
        {"service": "checkout", "query": "x", "since_minutes": 60},
        ctx,
    )
    guard.invoke(tools["rollback_deploy"], {"service": "checkout", "to_version": "v1.4.3"}, ctx)
    assert len(guard.audit_log) == 2
    assert guard.audit_log[0].executed is True
    assert guard.audit_log[1].executed is False
