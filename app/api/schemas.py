"""Request/response DTOs for the HTTP API.

Kept separate from the domain models: these describe the *wire* shape of the
API and are free to evolve independently of the graph's internal state.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IncidentStatus(str, Enum):
    """Lifecycle of an incident session as seen over HTTP."""

    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    ERROR = "error"


class ScenarioSummary(BaseModel):
    """One selectable scenario, surfaced by ``GET /scenarios``."""

    id: str = Field(description="Scenario id (YAML filename stem)")
    incident_id: str
    title: str
    service: str
    severity: str
    description: str


class StartIncidentRequest(BaseModel):
    scenario_id: str = Field(description="Which scenario to run (see /scenarios)")
    prompt: str | None = Field(
        default=None,
        description="Optional operator instruction; a sensible default is used.",
    )


class ApprovalRequest(BaseModel):
    approved: bool = Field(description="Approve or reject the pending high-risk step")


class PendingApproval(BaseModel):
    """The question surfaced when the graph pauses for human approval."""

    pending_agent: str
    question: str
    detail: dict[str, Any] = Field(default_factory=dict)


class AuditItem(BaseModel):
    tool_name: str
    harm_tier: str
    decision: str
    executed: bool
    detail: str = ""


class IncidentState(BaseModel):
    """Snapshot of an incident session returned by most endpoints."""

    session_id: str
    scenario_id: str
    incident_id: str
    status: IncidentStatus
    completed_agents: list[str] = Field(default_factory=list)
    findings: dict[str, str] = Field(default_factory=dict)
    pending_approval: PendingApproval | None = None
    audit: list[AuditItem] = Field(default_factory=list)
    actions_performed: list[str] = Field(default_factory=list)


class EvalDimension(BaseModel):
    dimension: str
    score: int
    passed: bool
    reasoning: str


class EvalResponse(BaseModel):
    session_id: str
    passed: bool
    mean_score: float
    dimensions: list[EvalDimension] = Field(default_factory=list)
