"""Tests for the diff engine."""

from __future__ import annotations

import pytest

from ragprobe.diff import compute_diff
from ragprobe.models import GradedResult, RunReport


def _result(qid: str, mode: str, passed: bool) -> GradedResult:
    return GradedResult(
        question_id=qid,
        question=f"question {qid}",
        failure_mode=mode,
        answer="a",
        passed=passed,
        grade_reasoning="r",
    )


def _report(run_id: str, results: list[GradedResult]) -> RunReport:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    by_mode: dict[str, dict] = {}
    for r in results:
        b = by_mode.setdefault(r.failure_mode, {"total": 0, "passed": 0, "pass_rate": 0.0})
        b["total"] += 1
        if r.passed:
            b["passed"] += 1
    for b in by_mode.values():
        b["pass_rate"] = b["passed"] / b["total"]
    return RunReport(
        run_id=run_id,
        timestamp="2026-06-21T00:00:00Z",
        pipeline_url="http://x",
        total_questions=total,
        passed=passed,
        failed=total - passed,
        pass_rate=passed / total if total else 0.0,
        by_failure_mode=by_mode,
        results=results,
    )


def test_detects_new_failures_and_passes():
    baseline = _report(
        "run_a",
        [
            _result("q1", "multi_hop", True),
            _result("q2", "multi_hop", False),
            _result("q3", "distractor", True),
        ],
    )
    current = _report(
        "run_b",
        [
            _result("q1", "multi_hop", False),  # regressed
            _result("q2", "multi_hop", True),   # improved
            _result("q3", "distractor", True),  # stable
        ],
    )
    diff = compute_diff(baseline, current)
    new_fail_ids = {r.question_id for r in diff.new_failures}
    new_pass_ids = {r.question_id for r in diff.new_passes}
    assert new_fail_ids == {"q1"}
    assert new_pass_ids == {"q2"}


def test_overall_delta():
    baseline = _report("run_a", [_result("q1", "multi_hop", True), _result("q2", "multi_hop", False)])
    current = _report("run_b", [_result("q1", "multi_hop", True), _result("q2", "multi_hop", True)])
    diff = compute_diff(baseline, current)
    assert diff.overall_delta == pytest.approx(0.5, abs=1e-4)


def test_regression_threshold_triggers():
    baseline = _report(
        "run_a",
        [_result("q1", "multi_hop", True), _result("q2", "multi_hop", True)],
    )
    current = _report(
        "run_b",
        [_result("q1", "multi_hop", False), _result("q2", "multi_hop", False)],
    )
    # multi_hop dropped 100pp; threshold 5pp -> regression
    diff = compute_diff(baseline, current, threshold=5.0)
    assert diff.regression_detected is True
    assert "multi_hop" in diff.regressed_failure_modes


def test_no_threshold_means_no_regression_flag():
    baseline = _report("run_a", [_result("q1", "multi_hop", True)])
    current = _report("run_b", [_result("q1", "multi_hop", False)])
    diff = compute_diff(baseline, current, threshold=None)
    assert diff.regression_detected is False
    assert diff.regressed_failure_modes == []
