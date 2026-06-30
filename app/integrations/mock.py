"""In-memory mock SRE environment.

``MockSRE`` is a single deterministic object that implements **all five**
integration protocols (incident store, logs, metrics, CI, deploys). It is seeded
from a scenario file so the diagnosis loop has a coherent story to uncover, and so
tests/evals are fully reproducible — no network, no real infrastructure.

Design notes:
* Time is stored in the scenario as ``minutes_ago`` offsets and converted to
  absolute timestamps relative to a single ``base_now`` captured at construction.
  This keeps time-window filtering (``since_minutes``) meaningful *and*
  deterministic for a given run.
* Write operations are recorded in ``performed_actions`` so tests can assert
  "what did the agent actually do?" and unknown/unavailable services fail loudly
  (so the unhappy paths are testable).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel

from app.domain.models import Incident, IncidentStatus, Severity
from app.integrations.types import (
    ActionOutcome,
    Alert,
    CIRun,
    Commit,
    Deploy,
    LogEntry,
    MetricPoint,
    MetricSeries,
)

# --------------------------------------------------------------------------- #
# Scenario schema (the on-disk YAML shape). Prefixed with `_` where it is an
# internal detail of the loader; `Scenario`/`Expectation` are public so tests
# and evals can read the ground truth.
# --------------------------------------------------------------------------- #


class _RawAlert(BaseModel):
    id: str
    title: str
    summary: str = ""
    severity: Severity = Severity.MEDIUM
    minutes_ago: int = 0


class _RawLog(BaseModel):
    minutes_ago: int
    level: str
    message: str


class _RawMetricPoint(BaseModel):
    minutes_ago: int
    value: float


class _RawMetric(BaseModel):
    unit: str = ""
    points: list[_RawMetricPoint] = []


class _RawCIRun(BaseModel):
    id: str
    status: str
    branch: str = "main"
    commit_sha: str = ""
    minutes_ago: int = 0


class _RawCommit(BaseModel):
    sha: str
    author: str
    message: str
    minutes_ago: int = 0
    files: list[str] = []


class _RawDeploy(BaseModel):
    id: str
    version: str
    commit_sha: str = ""
    environment: str = "production"
    status: str = "succeeded"
    minutes_ago: int = 0


class _RawIncident(BaseModel):
    id: str
    title: str
    service: str
    severity: Severity = Severity.MEDIUM
    status: IncidentStatus = IncidentStatus.OPEN
    description: str = ""


class Expectation(BaseModel):
    """Ground truth for evals: what *should* the copilot conclude/do?"""

    root_cause: str = ""
    safe_action: str = ""
    forbidden_actions: list[str] = []


class Scenario(BaseModel):
    """The full seeded world for one incident."""

    incident: _RawIncident
    alerts: list[_RawAlert] = []
    logs: dict[str, list[_RawLog]] = {}
    metrics: dict[str, dict[str, _RawMetric]] = {}
    ci_runs: dict[str, list[_RawCIRun]] = {}
    ci_logs: dict[str, str] = {}
    commits: dict[str, list[_RawCommit]] = {}
    deploys: dict[str, list[_RawDeploy]] = {}
    unavailable_services: list[str] = []
    expected: Expectation | None = None


# --------------------------------------------------------------------------- #
# The mock environment
# --------------------------------------------------------------------------- #


class MockSRE:
    """Deterministic implementation of every SRE integration protocol."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        base_now: datetime | None = None,
    ) -> None:
        self.scenario = scenario
        self._now = base_now or datetime.now(timezone.utc)
        self._incident = Incident(**scenario.incident.model_dump())
        self._unavailable = set(scenario.unavailable_services)
        self._notes: list[str] = []
        self.performed_actions: list[str] = []
        self._known_services = self._compute_known_services()

    @classmethod
    def from_scenario(cls, path: str | Path, **kwargs: object) -> MockSRE:
        """Load a scenario YAML file and build a ``MockSRE`` from it."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(Scenario(**data), **kwargs)  # type: ignore[arg-type]

    # -- helpers ----------------------------------------------------------- #

    def _ts(self, minutes_ago: int) -> datetime:
        return self._now - timedelta(minutes=minutes_ago)

    def _compute_known_services(self) -> set[str]:
        known = {self._incident.service}
        known.update(self.scenario.logs)
        known.update(self.scenario.metrics)
        known.update(self.scenario.deploys)
        known.update(self.scenario.ci_runs)
        known.update(self.scenario.commits)
        return known

    def _guard_service(self, service: str) -> ActionOutcome | None:
        """Return a failure outcome if a write target is unknown/unavailable."""
        if service in self._unavailable:
            return ActionOutcome(ok=False, message=f"service '{service}' is unavailable")
        if service not in self._known_services:
            return ActionOutcome(ok=False, message=f"service '{service}' not found")
        return None

    # -- IncidentStore ----------------------------------------------------- #

    def get_incident(self, incident_id: str) -> Incident | None:
        return self._incident if incident_id == self._incident.id else None

    def list_active_alerts(self) -> list[Alert]:
        return [
            Alert(
                id=a.id,
                service=self._incident.service,
                title=a.title,
                summary=a.summary,
                severity=a.severity,
                fired_at=self._ts(a.minutes_ago),
            )
            for a in self.scenario.alerts
        ]

    def post_note(self, incident_id: str, text: str) -> ActionOutcome:
        if incident_id != self._incident.id:
            return ActionOutcome(ok=False, message=f"incident {incident_id} not found")
        self._notes.append(text)
        ref = f"note-{len(self._notes)}"
        self.performed_actions.append(f"post_note:{ref}")
        return ActionOutcome(message="note added", ref=ref)

    def update_status(self, incident_id: str, status: IncidentStatus) -> ActionOutcome:
        if incident_id != self._incident.id:
            return ActionOutcome(ok=False, message=f"incident {incident_id} not found")
        self._incident.status = status
        self.performed_actions.append(f"update_status:{status.value}")
        return ActionOutcome(message=f"status set to {status.value}", ref=incident_id)

    # -- LogStore ---------------------------------------------------------- #

    def query_logs(
        self, service: str, query: str, since_minutes: int = 60
    ) -> list[LogEntry]:
        q = query.lower()
        rows = [
            LogEntry(
                timestamp=self._ts(r.minutes_ago),
                service=service,
                level=r.level,
                message=r.message,
            )
            for r in self.scenario.logs.get(service, [])
            if r.minutes_ago <= since_minutes
            and (not query or q in r.message.lower())
        ]
        return sorted(rows, key=lambda e: e.timestamp, reverse=True)

    # -- MetricsApi -------------------------------------------------------- #

    def query_metrics(
        self, service: str, metric: str, window_minutes: int = 60
    ) -> MetricSeries:
        raw = self.scenario.metrics.get(service, {}).get(metric)
        if raw is None:
            return MetricSeries(service=service, metric=metric)
        points = [
            MetricPoint(timestamp=self._ts(p.minutes_ago), value=p.value)
            for p in raw.points
            if p.minutes_ago <= window_minutes
        ]
        points.sort(key=lambda p: p.timestamp)
        return MetricSeries(
            service=service, metric=metric, unit=raw.unit, points=points
        )

    # -- CiProvider -------------------------------------------------------- #

    def _to_ci_run(self, service: str, r: _RawCIRun) -> CIRun:
        return CIRun(
            id=r.id,
            service=service,
            status=r.status,
            branch=r.branch,
            commit_sha=r.commit_sha,
            started_at=self._ts(r.minutes_ago),
        )

    def get_failing_ci_runs(self, service: str) -> list[CIRun]:
        return [
            self._to_ci_run(service, r)
            for r in self.scenario.ci_runs.get(service, [])
            if r.status == "failed"
        ]

    def get_ci_logs(self, run_id: str) -> str:
        return self.scenario.ci_logs.get(run_id, "")

    def get_recent_commits(self, service: str, limit: int = 10) -> list[Commit]:
        rows = sorted(self.scenario.commits.get(service, []), key=lambda c: c.minutes_ago)
        return [
            Commit(
                sha=c.sha,
                author=c.author,
                message=c.message,
                timestamp=self._ts(c.minutes_ago),
                files=c.files,
            )
            for c in rows[:limit]
        ]

    def git_blame(self, service: str, file: str) -> list[Commit]:
        return [c for c in self.get_recent_commits(service, limit=100) if file in c.files]

    def rerun_ci_job(self, run_id: str) -> ActionOutcome:
        known = {r.id for runs in self.scenario.ci_runs.values() for r in runs}
        if run_id not in known:
            return ActionOutcome(ok=False, message=f"ci run {run_id} not found")
        self.performed_actions.append(f"rerun_ci_job:{run_id}")
        return ActionOutcome(message=f"re-triggered {run_id}", ref=f"{run_id}-rerun")

    # -- DeployController -------------------------------------------------- #

    def list_recent_deploys(self, service: str, limit: int = 5) -> list[Deploy]:
        rows = sorted(self.scenario.deploys.get(service, []), key=lambda d: d.minutes_ago)
        return [
            Deploy(
                id=d.id,
                service=service,
                version=d.version,
                commit_sha=d.commit_sha,
                environment=d.environment,
                deployed_at=self._ts(d.minutes_ago),
                status=d.status,
            )
            for d in rows[:limit]
        ]

    def restart_service(self, service: str) -> ActionOutcome:
        bad = self._guard_service(service)
        if bad is not None:
            return bad
        self.performed_actions.append(f"restart_service:{service}")
        return ActionOutcome(message=f"restarted {service}", ref=service)

    def scale_service(self, service: str, replicas: int) -> ActionOutcome:
        bad = self._guard_service(service)
        if bad is not None:
            return bad
        self.performed_actions.append(f"scale_service:{service}:{replicas}")
        return ActionOutcome(message=f"scaled {service} to {replicas} replicas", ref=service)

    def rollback_deploy(self, service: str, to_version: str) -> ActionOutcome:
        bad = self._guard_service(service)
        if bad is not None:
            return bad
        self.performed_actions.append(f"rollback_deploy:{service}:{to_version}")
        return ActionOutcome(
            message=f"rolled back {service} to {to_version}",
            ref=f"deploy-rollback-{to_version}",
        )

    def failover(self, service: str, region: str) -> ActionOutcome:
        bad = self._guard_service(service)
        if bad is not None:
            return bad
        self.performed_actions.append(f"failover:{service}:{region}")
        return ActionOutcome(message=f"failed over {service} to {region}", ref=region)
