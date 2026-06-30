"""Data-transfer objects returned by the integration (repository) layer.

These are the *shapes of data* the simulated SRE environment hands back: log
lines, metric series, CI runs, commits, deploys, and the outcome of a write
action. They are plain pydantic models with no LangChain/LangGraph dependency.

Why separate from ``app/domain``? ``domain`` holds the core business entities the
whole app reasons about (Incident, HarmTier, …). These are integration-specific
payloads — the equivalent of API/DB row DTOs in a repository layer.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.domain.models import Severity


class Alert(BaseModel):
    """A monitoring alert that may open or relate to an incident."""

    id: str
    service: str
    title: str
    summary: str = ""
    severity: Severity = Severity.MEDIUM
    fired_at: datetime


class LogEntry(BaseModel):
    """A single log line from the log store."""

    timestamp: datetime
    service: str
    level: str  # ERROR / WARN / INFO / DEBUG
    message: str


class MetricPoint(BaseModel):
    """One (timestamp, value) sample in a metric series."""

    timestamp: datetime
    value: float


class MetricSeries(BaseModel):
    """A time series for a single metric (e.g. error_rate, p99_latency_ms)."""

    service: str
    metric: str
    unit: str = ""
    points: list[MetricPoint] = []


class Commit(BaseModel):
    """A VCS commit, used for blame / 'what changed recently'."""

    sha: str
    author: str
    message: str
    timestamp: datetime
    files: list[str] = []


class CIRun(BaseModel):
    """A CI pipeline run (e.g. a GitHub Actions workflow run)."""

    id: str
    service: str
    status: str  # passed / failed / running
    branch: str = "main"
    commit_sha: str = ""
    started_at: datetime


class Deploy(BaseModel):
    """A deployment of a service version to an environment."""

    id: str
    service: str
    version: str
    commit_sha: str = ""
    environment: str = "production"
    deployed_at: datetime
    status: str = "succeeded"  # succeeded / failed / rolled_back


class ActionOutcome(BaseModel):
    """Result of a *write* operation (restart, rerun, rollback, note, …).

    Integrations return this for mutating calls; tools then translate it into the
    domain-level ``ToolResult`` the agents consume.
    """

    ok: bool = True
    message: str = ""
    ref: str | None = None  # id of the created/affected resource, if any
