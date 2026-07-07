"""Supervisor graph (Layer 6) — orchestrates the sub-agents into one workflow.

This is the outer loop of the system. A ``supervisor`` node routes to each
specialized sub-agent in turn (order from ``agents.yaml``'s ``workflow``), each
agent runs its own inner ReAct loop, then control returns to the supervisor until
the workflow is complete or a step budget is hit.

Key LangGraph concepts introduced here:
* **StateGraph** — nodes + edges over a shared ``IncidentState``.
* **Conditional edges** — the supervisor's routing decision (which agent / FINISH).
* **Checkpointer** — persists state so a run is pausable/resumable.
* **interrupt()** — human-in-the-loop: pause before high-risk remediation, resume
  with an approval decision that unlocks the guarded tools.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from app.agents.registry import list_agent_names
from app.graph.state import IncidentState
from app.security.guard import ApprovalContext

FINISH = "FINISH"
APPROVAL_NODE = "approval"
SUPERVISOR_NODE = "supervisor"


def _last_message_text(result: Any) -> str:
    """Extract the final message text from a sub-agent invocation result."""
    if isinstance(result, dict) and result.get("messages"):
        last = result["messages"][-1]
        return getattr(last, "content", str(last))
    return str(result)


def build_supervisor_graph(
    agents: dict[str, Runnable],
    *,
    workflow: list[str] | None = None,
    sensitive_agents: set[str] | None = None,
    approval_ctx: ApprovalContext | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    step_budget: int = 12,
) -> CompiledStateGraph:
    """Wire the supervisor + agent nodes into a compiled, resumable graph.

    Args:
        agents: compiled sub-agents keyed by name (anything with ``.invoke``).
        workflow: routing order; defaults to the YAML registry order.
        sensitive_agents: agents that require human approval before running.
        approval_ctx: shared approval context unlocked by the HITL gate.
        checkpointer: state persistence (defaults to in-memory).
        step_budget: max agent hops before the supervisor force-finishes.
    """
    order = workflow or list_agent_names()
    sensitive = sensitive_agents or set()
    graph: StateGraph = StateGraph(IncidentState)

    # -- supervisor node: pick the next uncompleted agent, or FINISH ------- #
    def supervisor(state: IncidentState) -> dict[str, Any]:
        completed = state.get("completed", [])
        steps = state.get("steps", 0)
        if steps >= step_budget:
            return {"next_agent": FINISH}
        for name in order:
            if name not in completed:
                return {"next_agent": name}
        return {"next_agent": FINISH}

    graph.add_node(SUPERVISOR_NODE, supervisor)

    # -- one node per agent ------------------------------------------------ #
    def _make_agent_node(name: str):
        def _node(state: IncidentState) -> dict[str, Any]:
            result = agents[name].invoke({"messages": state.get("messages", [])})
            text = _last_message_text(result)
            return {
                "messages": [AIMessage(content=text, name=name)],
                "completed": [name],
                "findings": {name: text},
                "steps": state.get("steps", 0) + 1,
            }

        return _node

    for name in order:
        graph.add_node(name, _make_agent_node(name))
        graph.add_edge(name, SUPERVISOR_NODE)

    # -- human-in-the-loop approval gate ---------------------------------- #
    def approval(state: IncidentState) -> dict[str, Any]:
        pending = state.get("next_agent", "")
        decision = interrupt(
            {
                "type": "approval_request",
                "incident_id": state.get("incident_id"),
                "pending_agent": pending,
                "question": (
                    f"Approve high-risk actions for '{pending}'? "
                    f"Reply with {{'approved': true}} to proceed."
                ),
            }
        )
        approved = (
            bool(decision.get("approved"))
            if isinstance(decision, dict)
            else bool(decision)
        )
        if approved and approval_ctx is not None:
            approval_ctx.approve_all = True
        return {"context": {"approved": approved}}

    graph.add_node(APPROVAL_NODE, approval)

    # -- routing ---------------------------------------------------------- #
    def route_from_supervisor(state: IncidentState) -> str:
        nxt = state.get("next_agent", FINISH)
        if nxt == FINISH:
            return END
        already_approved = approval_ctx is not None and approval_ctx.approve_all
        if nxt in sensitive and not already_approved:
            return APPROVAL_NODE
        return nxt

    graph.add_edge(START, SUPERVISOR_NODE)
    graph.add_conditional_edges(
        SUPERVISOR_NODE,
        route_from_supervisor,
        [*order, APPROVAL_NODE, END],
    )
    # After approval, proceed to the agent the supervisor selected.
    graph.add_conditional_edges(
        APPROVAL_NODE,
        lambda state: state.get("next_agent", FINISH),
        {**{name: name for name in order}, FINISH: END},
    )

    return graph.compile(checkpointer=checkpointer or MemorySaver())
