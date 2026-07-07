"""Offline tests for the FastAPI layer.

The copilot and judge are faked (injected into ``IncidentService``), so the full
HTTP flow — start → pause for approval → resume → audit → eval — is exercised with
no LLM, embeddings, or network. Scenario listing uses the real YAML files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.server import create_app
from app.api.service import IncidentService
from app.eval.judge import JudgeVerdict


class _Interrupt:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value


class _FakeGuard:
    def __init__(self) -> None:
        self.audit_log: list[Any] = []


class _FakeEnv:
    def __init__(self) -> None:
        self.performed_actions: list[str] = []


class _FakeGraph:
    """First invoke pauses for approval; resume completes the run."""

    def __init__(self, env: _FakeEnv) -> None:
        self._env = env
        self.calls = 0

    def invoke(self, payload: Any, config: Any = None) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {
                "completed": ["triage", "diagnosis"],
                "findings": {
                    "triage": "checkout 5xx spike, HIGH severity",
                    "diagnosis": "root cause: deploy D-0077 NullPointerException",
                },
                "__interrupt__": [
                    _Interrupt(
                        {
                            "type": "approval_request",
                            "pending_agent": "remediation",
                            "question": "Approve high-risk actions for 'remediation'?",
                        }
                    )
                ],
            }
        self._env.performed_actions.append("rollback_deploy:checkout:v1.4.3")
        return {
            "completed": ["triage", "diagnosis", "remediation"],
            "findings": {
                "triage": "checkout 5xx spike, HIGH severity",
                "diagnosis": "root cause: deploy D-0077 NullPointerException",
                "remediation": "roll back checkout to v1.4.3",
            },
        }


@dataclass
class _FakeCopilot:
    graph: _FakeGraph
    guard: _FakeGuard
    env: _FakeEnv
    approval_ctx: Any = None
    retriever: Any = None
    sensitive_agents: frozenset[str] = field(default_factory=lambda: frozenset({"remediation"}))


def _fake_copilot_factory(_path: Any) -> _FakeCopilot:
    env = _FakeEnv()
    return _FakeCopilot(graph=_FakeGraph(env), guard=_FakeGuard(), env=env)


class _StubJudge:
    def invoke(self, _prompt: str, *args: Any, **kwargs: Any) -> JudgeVerdict:
        return JudgeVerdict(score=5, reasoning="matches ground truth")


@pytest.fixture()
def client() -> TestClient:
    service = IncidentService(
        copilot_factory=_fake_copilot_factory,
        judge_builder=lambda: _StubJudge(),
    )
    return TestClient(create_app(service=service))


def _start(client: TestClient, scenario: str = "checkout-5xx-spike") -> dict[str, Any]:
    resp = client.post("/incidents", json={"scenario_id": scenario})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_list_scenarios_from_real_files(client: TestClient) -> None:
    body = client.get("/scenarios").json()
    ids = {s["id"] for s in body}
    assert {"checkout-5xx-spike", "db-connection-pool-exhaustion", "cdn-latency-spike"} <= ids
    checkout = next(s for s in body if s["id"] == "checkout-5xx-spike")
    assert checkout["incident_id"] == "INC-1001"
    assert checkout["service"] == "checkout"


def test_start_pauses_for_approval(client: TestClient) -> None:
    body = _start(client)
    assert body["status"] == "awaiting_approval"
    assert body["pending_approval"]["pending_agent"] == "remediation"
    assert "diagnosis" in body["findings"]
    assert "remediation" not in body["completed_agents"]


def test_get_incident_roundtrip(client: TestClient) -> None:
    session_id = _start(client)["session_id"]
    body = client.get(f"/incidents/{session_id}").json()
    assert body["session_id"] == session_id
    assert body["status"] == "awaiting_approval"


def test_approve_completes_run(client: TestClient) -> None:
    session_id = _start(client)["session_id"]
    body = client.post(
        f"/incidents/{session_id}/approve", json={"approved": True}
    ).json()
    assert body["status"] == "completed"
    assert "remediation" in body["completed_agents"]
    assert body["actions_performed"] == ["rollback_deploy:checkout:v1.4.3"]


def test_double_approve_is_conflict(client: TestClient) -> None:
    session_id = _start(client)["session_id"]
    client.post(f"/incidents/{session_id}/approve", json={"approved": True})
    resp = client.post(f"/incidents/{session_id}/approve", json={"approved": True})
    assert resp.status_code == 409


def test_unknown_scenario_404(client: TestClient) -> None:
    resp = client.post("/incidents", json={"scenario_id": "does-not-exist"})
    assert resp.status_code == 404


def test_unknown_session_404(client: TestClient) -> None:
    assert client.get("/incidents/nope").status_code == 404


def test_eval_endpoint(client: TestClient) -> None:
    session_id = _start(client)["session_id"]
    client.post(f"/incidents/{session_id}/approve", json={"approved": True})
    body = client.post(f"/incidents/{session_id}/eval").json()
    assert body["passed"] is True
    assert len(body["dimensions"]) == 3
    assert body["mean_score"] == 5.0
