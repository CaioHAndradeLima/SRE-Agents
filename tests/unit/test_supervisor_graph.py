"""Unit tests for the supervisor graph (Layer 6).

Uses stub sub-agents (no LLM) so the outer loop, routing, step budget, and the
human-in-the-loop interrupt are all tested deterministically and offline.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.graph.supervisor import build_supervisor_graph
from app.security.guard import ApprovalContext

WORKFLOW = ["triage", "diagnosis", "remediation"]


class StubAgent:
    """Minimal stand-in for a compiled ReAct agent."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def invoke(self, payload: dict) -> dict:
        self.calls += 1
        return {"messages": [AIMessage(content=f"{self.name} complete", name=self.name)]}


def _agents() -> dict[str, StubAgent]:
    return {name: StubAgent(name) for name in WORKFLOW}


def test_runs_all_agents_in_workflow_order() -> None:
    agents = _agents()
    graph = build_supervisor_graph(agents, workflow=WORKFLOW)
    final = graph.invoke(
        {"incident_id": "INC-1001", "messages": [("user", "checkout 5xx")]},
        config={"configurable": {"thread_id": "t1"}},
    )
    assert final["completed"] == WORKFLOW
    assert set(final["findings"]) == set(WORKFLOW)
    assert all(a.calls == 1 for a in agents.values())


def test_step_budget_forces_finish() -> None:
    agents = _agents()
    graph = build_supervisor_graph(agents, workflow=WORKFLOW, step_budget=1)
    final = graph.invoke(
        {"incident_id": "INC-1", "messages": [("user", "go")]},
        config={"configurable": {"thread_id": "t2"}},
    )
    # Only one agent hop allowed before FINISH.
    assert len(final["completed"]) == 1


def test_hitl_interrupt_pauses_before_sensitive_agent() -> None:
    agents = _agents()
    ctx = ApprovalContext()
    graph = build_supervisor_graph(
        agents,
        workflow=WORKFLOW,
        sensitive_agents={"remediation"},
        approval_ctx=ctx,
    )
    config = {"configurable": {"thread_id": "t3"}}
    result = graph.invoke(
        {"incident_id": "INC-1", "messages": [("user", "go")]},
        config=config,
    )
    # Paused at the approval gate before remediation ran.
    assert "__interrupt__" in result
    assert agents["remediation"].calls == 0
    assert agents["triage"].calls == 1
    assert ctx.approve_all is False


def test_hitl_resume_grants_approval_and_completes() -> None:
    agents = _agents()
    ctx = ApprovalContext()
    graph = build_supervisor_graph(
        agents,
        workflow=WORKFLOW,
        sensitive_agents={"remediation"},
        approval_ctx=ctx,
    )
    config = {"configurable": {"thread_id": "t4"}}
    graph.invoke(
        {"incident_id": "INC-1", "messages": [("user", "go")]},
        config=config,
    )
    final = graph.invoke(Command(resume={"approved": True}), config=config)
    assert ctx.approve_all is True
    assert final["completed"] == WORKFLOW
    assert agents["remediation"].calls == 1
