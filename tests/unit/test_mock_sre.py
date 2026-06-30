"""Unit tests for the mock SRE environment (Layer 2).

These run fully offline — no LLM, no network — and demonstrate the payoff of the
protocol + mock design: we can exercise every integration deterministically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.models import IncidentStatus
from app.integrations.mock import MockSRE
from app.integrations.protocols import (
    CiProvider,
    DeployController,
    IncidentStore,
    LogStore,
    MetricsApi,
)

SCENARIO = (
    Path(__file__).resolve().parents[2] / "data" / "scenarios" / "checkout-5xx-spike.yaml"
)


@pytest.fixture
def sre() -> MockSRE:
    return MockSRE.from_scenario(SCENARIO)


def test_mock_satisfies_all_protocols(sre: MockSRE) -> None:
    # One object plays every role (structural typing, @runtime_checkable).
    assert isinstance(sre, IncidentStore)
    assert isinstance(sre, LogStore)
    assert isinstance(sre, MetricsApi)
    assert isinstance(sre, CiProvider)
    assert isinstance(sre, DeployController)


def test_get_incident_roundtrip(sre: MockSRE) -> None:
    inc = sre.get_incident("INC-1001")
    assert inc is not None
    assert inc.service == "checkout"
    assert sre.get_incident("does-not-exist") is None


def test_query_logs_filters_by_query_and_window(sre: MockSRE) -> None:
    errors = sre.query_logs("checkout", "NullPointer", since_minutes=60)
    assert errors, "expected to find NullPointer logs"
    assert all("nullpointer" in e.message.lower() for e in errors)
    # Most-recent-first ordering.
    assert errors[0].timestamp >= errors[-1].timestamp
    # The 85-min-old baseline INFO line is outside a 60-min window.
    recent = sre.query_logs("checkout", "", since_minutes=60)
    assert all("baseline" not in e.message for e in recent)
    # No matches -> empty list, not an error.
    assert sre.query_logs("checkout", "kafka", since_minutes=60) == []


def test_metrics_show_the_spike(sre: MockSRE) -> None:
    series = sre.query_metrics("checkout", "error_rate", window_minutes=60)
    assert series.unit == "%"
    values = [p.value for p in series.points]
    assert max(values) > 5.0  # the spike is visible
    # Unknown metric returns an empty series rather than raising.
    assert sre.query_metrics("checkout", "made_up", 60).points == []


def test_failing_ci_run_and_logs(sre: MockSRE) -> None:
    failing = sre.get_failing_ci_runs("checkout")
    assert [r.id for r in failing] == ["RUN-501"]
    assert "NullPointerException" in sre.get_ci_logs("RUN-501")


def test_git_blame_points_at_the_change(sre: MockSRE) -> None:
    blame = sre.git_blame("checkout", "src/checkout/payment_client.py")
    assert [c.sha for c in blame] == ["c3a9"]


def test_recent_deploys_ordered_most_recent_first(sre: MockSRE) -> None:
    deploys = sre.list_recent_deploys("checkout")
    assert deploys[0].version == "v1.5.0"  # the suspect deploy is newest


def test_write_actions_are_recorded(sre: MockSRE) -> None:
    out = sre.rollback_deploy("checkout", "v1.4.3")
    assert out.ok is True
    assert out.ref == "deploy-rollback-v1.4.3"
    assert "rollback_deploy:checkout:v1.4.3" in sre.performed_actions


def test_write_to_unknown_service_fails(sre: MockSRE) -> None:
    out = sre.restart_service("totally-unknown")
    assert out.ok is False
    assert "not found" in out.message


def test_unavailable_service_is_guarded() -> None:
    sre = MockSRE.from_scenario(SCENARIO)
    sre._unavailable.add("checkout")  # simulate an outage of the control plane
    out = sre.restart_service("checkout")
    assert out.ok is False
    assert "unavailable" in out.message


def test_update_status_mutates_incident(sre: MockSRE) -> None:
    out = sre.update_status("INC-1001", IncidentStatus.INVESTIGATING)
    assert out.ok is True
    assert sre.get_incident("INC-1001").status is IncidentStatus.INVESTIGATING
