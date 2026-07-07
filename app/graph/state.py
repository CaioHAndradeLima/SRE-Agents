"""Incident state — the shared memory that flows through the supervisor graph.

LangGraph passes a single state object between nodes. We use a ``TypedDict`` with
reducers so updates *merge* instead of overwrite:

* ``messages`` uses ``add_messages`` so each node appends to the conversation.
* ``completed`` / ``findings`` use custom reducers so agent nodes accumulate results
  without clobbering earlier ones.

Because state is checkpointed (Layer 6 uses a checkpointer), an incident run is
pausable and resumable — essential for human-in-the-loop approval.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def _merge_completed(left: list[str], right: list[str]) -> list[str]:
    """Append newly-completed agents, preserving order, without duplicates."""
    merged = list(left)
    for name in right:
        if name not in merged:
            merged.append(name)
    return merged


def _merge_findings(
    left: dict[str, str], right: dict[str, str]
) -> dict[str, str]:
    """Merge each agent's finding into the accumulated dict."""
    return {**left, **right}


class IncidentState(TypedDict, total=False):
    """State carried through the incident-resolution graph."""

    incident_id: str
    # The running conversation across supervisor + sub-agents.
    messages: Annotated[list[AnyMessage], add_messages]
    # Agents that have finished, in execution order.
    completed: Annotated[list[str], _merge_completed]
    # Each agent's final summary, keyed by agent name.
    findings: Annotated[dict[str, str], _merge_findings]
    # The supervisor's routing decision for the next hop.
    next_agent: str
    # Safety budget: incremented per agent hop to bound the outer loop.
    steps: int
    # Free-form scratch space (e.g. approval metadata).
    context: dict[str, Any]
