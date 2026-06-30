"""Integration contracts (the repository interfaces)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.models import Incident, IncidentStatus
from app.integrations.types import (
    ActionOutcome,
    Alert,
    CIRun,
    Commit,
    Deploy,
    LogEntry,
    MetricSeries,
)


@runtime_checkable
class IncidentStore(Protocol):
    """Read/write access to incidents and the alerts that surface them."""

    def get_incident(self, incident_id: str) -> Incident | None: ...

    def list_active_alerts(self) -> list[Alert]: ...

    def post_note(self, incident_id: str, text: str) -> ActionOutcome: ...

    def update_status(
        self, incident_id: str, status: IncidentStatus
    ) -> ActionOutcome: ...


@runtime_checkable
class LogStore(Protocol):
    """Query application/service logs (🟢 read-only)."""

    def query_logs(
        self, service: str, query: str, since_minutes: int = 60
    ) -> list[LogEntry]: ...


@runtime_checkable
class MetricsApi(Protocol):
    """Query time-series metrics like error_rate or p99_latency_ms (🟢 read-only)."""

    def query_metrics(
        self, service: str, metric: str, window_minutes: int = 60
    ) -> MetricSeries: ...


@runtime_checkable
class CiProvider(Protocol):
    """Inspect CI pipelines and the commits behind them."""

    # read (🟢)
    def get_failing_ci_runs(self, service: str) -> list[CIRun]: ...

    def get_ci_logs(self, run_id: str) -> str: ...

    def get_recent_commits(self, service: str, limit: int = 10) -> list[Commit]: ...

    def git_blame(self, service: str, file: str) -> list[Commit]: ...

    # write (🟡 compensable)
    def rerun_ci_job(self, run_id: str) -> ActionOutcome: ...


@runtime_checkable
class DeployController(Protocol):
    """Inspect and control deployments — spans the full harm gradient."""

    # read (🟢)
    def list_recent_deploys(self, service: str, limit: int = 5) -> list[Deploy]: ...

    # write (🟠 irreversible)
    def restart_service(self, service: str) -> ActionOutcome: ...

    def scale_service(self, service: str, replicas: int) -> ActionOutcome: ...

    # write (🔴 critical)
    def rollback_deploy(self, service: str, to_version: str) -> ActionOutcome: ...

    def failover(self, service: str, region: str) -> ActionOutcome: ...
