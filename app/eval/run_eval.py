"""Eval loop CLI — run the copilot on a scenario, then judge its findings.

    python -m app.eval.run_eval --scenario data/scenarios/checkout-5xx-spike.yaml

Ties together the full stack: build copilot -> resolve the incident (auto-approve
so it's non-interactive) -> load the scenario's ``expected`` ground truth -> grade
with the LLM judge -> print a pass/fail report and exit non-zero on failure (so it
can gate CI). Needs the gateway (LiteLLM) running for real model calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from app.composition import DEFAULT_RUNBOOKS, build_copilot
from app.config import get_settings
from app.eval.judge import EvalReport, build_judge, evaluate_findings
from app.main import DEFAULT_SCENARIO, run_incident
from app.observability.tracing import runnable_config, setup_observability


def load_expected(scenario_path: str | Path) -> dict[str, Any]:
    """Read the ``expected`` ground-truth block from a scenario YAML."""
    data = yaml.safe_load(Path(scenario_path).read_text(encoding="utf-8"))
    return data.get("expected", {})


def _print_report(report: EvalReport) -> None:
    print("\n=== LLM-AS-JUDGE REPORT ===")
    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"[{mark}] {r.dimension}: {r.score}/5 — {r.reasoning}")
    print(f"\nMean score: {report.mean_score:.2f}  Overall: {'PASS' if report.passed else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SRE Copilot LLM-as-judge eval")
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    parser.add_argument("--runbooks", default=str(DEFAULT_RUNBOOKS))
    parser.add_argument("--incident-id", default="EVAL-1001")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_observability(settings)

    copilot = build_copilot(
        args.scenario, settings=settings, runbooks_dir=args.runbooks
    )
    config = runnable_config(args.incident_id, settings=settings)
    prompt = (
        f"Incident {args.incident_id}: investigate and resolve. "
        f"Triage, find the root cause, and remediate safely."
    )
    result = run_incident(
        copilot, args.incident_id, prompt=prompt, config=config, auto_approve=True
    )

    findings = result.get("findings", {})
    expected = load_expected(args.scenario)
    judge = build_judge(settings=settings)
    report = evaluate_findings(findings, expected, judge)

    _print_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
