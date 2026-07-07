"""Offline tests for the LLM-as-judge core.

The judge model is stubbed (any object with ``.invoke(prompt) -> JudgeVerdict``),
so these run with no network/keys. They verify the harness plumbing: prompt
construction, threshold-based pass/fail, and report aggregation — not model quality.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app.eval.judge import (
    DIMENSION_CRITERIA,
    EvalReport,
    JudgeVerdict,
    evaluate_findings,
)

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "data" / "scenarios" / "checkout-5xx-spike.yaml"


class StubJudge:
    """Returns queued verdicts and records prompts for assertions."""

    def __init__(self, verdicts: list[JudgeVerdict]) -> None:
        self._verdicts = list(verdicts)
        self.prompts: list[str] = []

    def invoke(self, prompt: str, *args, **kwargs) -> JudgeVerdict:
        self.prompts.append(prompt)
        return self._verdicts.pop(0)


def _expected() -> dict:
    data = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    return data["expected"]


FINDINGS = {
    "triage": "checkout error rate spiked to 7%, HIGH severity.",
    "diagnosis": "Deploy D-0077 (v1.5.0, c3a9) introduced a NullPointerException.",
    "remediation": "Roll back checkout to v1.4.3 (D-0076).",
}


def test_all_pass_produces_passing_report() -> None:
    judge = StubJudge([JudgeVerdict(score=5, reasoning="ok")] * 3)
    report = evaluate_findings(FINDINGS, _expected(), judge)
    assert isinstance(report, EvalReport)
    assert report.passed is True
    assert report.mean_score == 5.0
    assert len(report.results) == len(DIMENSION_CRITERIA)


def test_low_score_fails_dimension_and_overall() -> None:
    judge = StubJudge(
        [
            JudgeVerdict(score=5, reasoning="root cause correct"),
            JudgeVerdict(score=1, reasoning="proposed failover (forbidden)"),
            JudgeVerdict(score=4, reasoning="grounded"),
        ]
    )
    report = evaluate_findings(FINDINGS, _expected(), judge)
    assert report.passed is False
    safety = next(r for r in report.results if r.dimension == "remediation_safety")
    assert safety.passed is False and safety.score == 1


def test_threshold_boundary_is_inclusive() -> None:
    judge = StubJudge([JudgeVerdict(score=4, reasoning="borderline")] * 3)
    report = evaluate_findings(FINDINGS, _expected(), judge, threshold=4)
    assert report.passed is True


def test_prompt_includes_expected_and_actual() -> None:
    judge = StubJudge([JudgeVerdict(score=5, reasoning="ok")] * 3)
    evaluate_findings(FINDINGS, _expected(), judge)
    root_cause_prompt = judge.prompts[0]
    assert "root_cause_correctness" in root_cause_prompt
    assert "PaymentClient" in root_cause_prompt  # from expected ground truth
    assert "Roll back checkout to v1.4.3" in root_cause_prompt  # from actual findings


def test_as_dict_shape() -> None:
    judge = StubJudge([JudgeVerdict(score=3, reasoning="meh")] * 3)
    report = evaluate_findings(FINDINGS, _expected(), judge)
    d = report.as_dict()
    assert d["passed"] is False
    assert set(d["dimensions"]) == set(DIMENSION_CRITERIA)
    assert d["dimensions"]["groundedness"]["score"] == 3
