"""CLI entry point (Layer 8) — run one incident scenario through the copilot.

    python -m app.main --scenario data/scenarios/checkout-5xx-spike.yaml

This is the "UI" for now (a FastAPI HTTP API can be added as an alternate entry
point later without touching the graph). It wires observability, builds the
copilot, drives the run, and handles human-in-the-loop approvals from the terminal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from langgraph.types import Command

from app.composition import DEFAULT_RUNBOOKS, IncidentCopilot, build_copilot
from app.config import get_settings
from app.observability.tracing import runnable_config, setup_observability

DEFAULT_SCENARIO = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "scenarios"
    / "checkout-5xx-spike.yaml"
)


def _approve_from_terminal(request: dict[str, Any]) -> bool:
    """Ask the operator to approve a high-risk step (used on interrupt)."""
    print("\n=== HUMAN APPROVAL REQUIRED ===")
    print(request.get("question", "Approve high-risk action?"))
    answer = input("Approve? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def run_incident(
    copilot: IncidentCopilot,
    incident_id: str,
    *,
    prompt: str,
    config: dict[str, Any],
    auto_approve: bool = False,
) -> dict[str, Any]:
    """Drive the graph to completion, resolving HITL interrupts as they occur."""
    state: Any = {
        "incident_id": incident_id,
        "messages": [("user", prompt)],
    }
    result = copilot.graph.invoke(state, config=config)

    # Resume loop: each interrupt is a pending human approval.
    while "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        request = interrupts[0].value if interrupts else {}
        approved = True if auto_approve else _approve_from_terminal(request)
        result = copilot.graph.invoke(Command(resume={"approved": approved}), config=config)

    return result


def _print_summary(result: dict[str, Any], copilot: IncidentCopilot) -> None:
    print("\n=== INCIDENT SUMMARY ===")
    for agent in result.get("completed", []):
        finding = result.get("findings", {}).get(agent, "")
        print(f"\n[{agent}]\n{finding}")
    print("\n=== AUDIT LOG ===")
    for entry in copilot.guard.audit_log:
        status = "EXECUTED" if entry.executed else "BLOCKED"
        print(f"- {entry.tool_name} ({entry.harm_tier}) -> {entry.decision.value} [{status}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SRE Incident Copilot")
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    parser.add_argument("--runbooks", default=str(DEFAULT_RUNBOOKS))
    parser.add_argument("--incident-id", default="INC-1001")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Approve high-risk actions automatically (demo/non-interactive).",
    )
    parser.add_argument("--use-qdrant-server", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_observability(settings)

    copilot = build_copilot(
        args.scenario,
        settings=settings,
        runbooks_dir=args.runbooks,
        use_server=args.use_qdrant_server,
    )
    config = runnable_config(args.incident_id, settings=settings)
    prompt = (
        f"Incident {args.incident_id}: investigate and resolve. "
        f"Triage, find the root cause, and remediate safely."
    )
    result = run_incident(
        copilot,
        args.incident_id,
        prompt=prompt,
        config=config,
        auto_approve=args.auto_approve,
    )
    _print_summary(result, copilot)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
