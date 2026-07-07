"""Incident session service — the stateful core behind the HTTP API.

Each started incident becomes a :class:`Session` that owns its *own* copilot
(isolated MockSRE world, permission guard, and checkpointer). The graph is driven
synchronously: ``start`` runs until the graph finishes or pauses at a human
approval ``interrupt()``; ``approve`` resumes it. State survives between HTTP
requests via the per-session checkpointer keyed by ``thread_id == session_id``.

The copilot factory and judge builder are injected, so tests swap in fakes and
run fully offline (no LLM, no embeddings, no network).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from langgraph.types import Command

from app.composition import DEFAULT_RUNBOOKS, IncidentCopilot, build_copilot
from app.config import Settings, get_settings
from app.eval.judge import build_judge, evaluate_findings
from app.observability.tracing import runnable_config
from app.api.schemas import (
    AuditItem,
    EvalDimension,
    EvalResponse,
    IncidentState,
    IncidentStatus,
    PendingApproval,
    ScenarioSummary,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS_DIR = ROOT / "data" / "scenarios"

CopilotFactory = Callable[[Path], IncidentCopilot]
JudgeBuilder = Callable[[], Any]


class ScenarioNotFound(Exception):
    """Requested scenario id has no matching YAML file."""


class SessionNotFound(Exception):
    """Unknown incident session id."""


class InvalidApprovalState(Exception):
    """Approve called on a session that is not awaiting approval."""


@dataclass
class Session:
    """A live incident run and its latest graph result."""

    session_id: str
    scenario_id: str
    scenario_path: Path
    incident_id: str
    copilot: IncidentCopilot
    config: dict[str, Any]
    status: IncidentStatus = IncidentStatus.COMPLETED
    result: dict[str, Any] = field(default_factory=dict)
    pending_approval: PendingApproval | None = None


class IncidentService:
    """Create, drive, inspect, and evaluate incident sessions."""

    def __init__(
        self,
        *,
        scenarios_dir: Path = DEFAULT_SCENARIOS_DIR,
        runbooks_dir: Path = DEFAULT_RUNBOOKS,
        settings: Settings | None = None,
        copilot_factory: CopilotFactory | None = None,
        judge_builder: JudgeBuilder | None = None,
    ) -> None:
        self._scenarios_dir = scenarios_dir
        self._runbooks_dir = runbooks_dir
        self._settings = settings or get_settings()
        self._copilot_factory = copilot_factory or self._default_copilot_factory
        self._judge_builder = judge_builder or (
            lambda: build_judge(settings=self._settings)
        )
        self._sessions: dict[str, Session] = {}

    # -- factories --------------------------------------------------------- #

    def _default_copilot_factory(self, path: Path) -> IncidentCopilot:
        return build_copilot(
            path, settings=self._settings, runbooks_dir=self._runbooks_dir
        )

    # -- scenarios --------------------------------------------------------- #

    def _scenario_path(self, scenario_id: str) -> Path:
        path = self._scenarios_dir / f"{scenario_id}.yaml"
        if not path.is_file():
            raise ScenarioNotFound(scenario_id)
        return path

    def list_scenarios(self) -> list[ScenarioSummary]:
        summaries: list[ScenarioSummary] = []
        for path in sorted(self._scenarios_dir.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            inc = data.get("incident", {})
            summaries.append(
                ScenarioSummary(
                    id=path.stem,
                    incident_id=str(inc.get("id", "")),
                    title=str(inc.get("title", path.stem)),
                    service=str(inc.get("service", "")),
                    severity=str(inc.get("severity", "")),
                    description=" ".join(str(inc.get("description", "")).split()),
                )
            )
        return summaries

    # -- session lifecycle ------------------------------------------------- #

    def start(self, scenario_id: str, prompt: str | None = None) -> Session:
        path = self._scenario_path(scenario_id)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        incident_id = str(data.get("incident", {}).get("id", scenario_id))

        session_id = uuid.uuid4().hex
        copilot = self._copilot_factory(path)
        config = runnable_config(session_id, settings=self._settings)
        prompt = prompt or (
            f"Incident {incident_id}: investigate and resolve. "
            f"Triage, find the root cause, and remediate safely."
        )
        session = Session(
            session_id=session_id,
            scenario_id=scenario_id,
            scenario_path=path,
            incident_id=incident_id,
            copilot=copilot,
            config=config,
        )
        result = copilot.graph.invoke(
            {"incident_id": incident_id, "messages": [("user", prompt)]},
            config=config,
        )
        self._apply_result(session, result)
        self._sessions[session_id] = session
        return session

    def approve(self, session_id: str, approved: bool) -> Session:
        session = self.get_session(session_id)
        if session.status != IncidentStatus.AWAITING_APPROVAL:
            raise InvalidApprovalState(session_id)
        result = session.copilot.graph.invoke(
            Command(resume={"approved": approved}), config=session.config
        )
        self._apply_result(session, result)
        return session

    def get_session(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFound(session_id) from exc

    def _apply_result(self, session: Session, result: dict[str, Any]) -> None:
        session.result = result
        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = interrupts[0].value or {}
            session.status = IncidentStatus.AWAITING_APPROVAL
            session.pending_approval = PendingApproval(
                pending_agent=str(payload.get("pending_agent", "")),
                question=str(payload.get("question", "Approve high-risk action?")),
                detail=payload if isinstance(payload, dict) else {},
            )
        else:
            session.status = IncidentStatus.COMPLETED
            session.pending_approval = None

    # -- projections ------------------------------------------------------- #

    def snapshot(self, session: Session) -> IncidentState:
        result = session.result
        audit = [
            AuditItem(
                tool_name=e.tool_name,
                harm_tier=e.harm_tier,
                decision=e.decision.value,
                executed=e.executed,
                detail=e.detail,
            )
            for e in session.copilot.guard.audit_log
        ]
        actions = list(getattr(session.copilot.env, "performed_actions", []))
        return IncidentState(
            session_id=session.session_id,
            scenario_id=session.scenario_id,
            incident_id=session.incident_id,
            status=session.status,
            completed_agents=list(result.get("completed", [])),
            findings=dict(result.get("findings", {})),
            pending_approval=session.pending_approval,
            audit=audit,
            actions_performed=actions,
        )

    def evaluate(self, session_id: str) -> EvalResponse:
        session = self.get_session(session_id)
        data = yaml.safe_load(session.scenario_path.read_text(encoding="utf-8")) or {}
        expected = data.get("expected", {})
        findings = dict(session.result.get("findings", {}))
        judge = self._judge_builder()
        report = evaluate_findings(findings, expected, judge)
        return EvalResponse(
            session_id=session_id,
            passed=report.passed,
            mean_score=report.mean_score,
            dimensions=[
                EvalDimension(
                    dimension=r.dimension,
                    score=r.score,
                    passed=r.passed,
                    reasoning=r.reasoning,
                )
                for r in report.results
            ],
        )
