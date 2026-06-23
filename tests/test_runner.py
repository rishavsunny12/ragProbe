"""Tests for the pipeline runner and grader (HTTP + LLM mocked)."""

from __future__ import annotations

import httpx
import pytest

from ragprobe.models import Chunk, GradedResult, PipelineAnswer, Question
from ragprobe.runner import (
    AnswerGrader,
    PipelineError,
    PipelineRunner,
    _extract_by_path,
    build_run_report,
    make_run_id,
)

from .conftest import fake_llm_response


def _q(qid: str = "q_mh_001") -> Question:
    return Question(
        id=qid,
        question='What about "quotes" and topic?',
        failure_mode="multi_hop",
        source_chunk_ids=["c0_0", "c1_0"],
        expected_difficulty=0.8,
    )


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, payload=None, raise_exc=None):
        self._payload = payload if payload is not None else {"answer": "hello"}
        self._raise = raise_exc
        self.calls = []

    async def post(self, url, json=None):
        self.calls.append(json)
        if self._raise is not None:
            raise self._raise
        return FakeResponse(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_build_body_escapes_quotes():
    runner = PipelineRunner("http://x", '{"query": "{{question}}"}', "answer")
    body = runner._build_body('say "hi"')
    assert body == {"query": 'say "hi"'}


def test_extract_by_path_nested():
    payload = {"data": {"answer": "deep"}}
    assert _extract_by_path(payload, "data.answer") == "deep"
    assert _extract_by_path(payload, "missing.key") is None
    assert _extract_by_path(payload, "") == payload


async def test_run_question_extracts_answer_and_latency():
    runner = PipelineRunner("http://x", '{"query": "{{question}}"}', "answer")
    client = FakeClient({"answer": "the answer", "retrieved_chunk_ids": ["c0_0"]})
    ans = await runner.run_question(_q(), client)
    assert isinstance(ans, PipelineAnswer)
    assert ans.answer == "the answer"
    assert ans.retrieved_chunk_ids == ["c0_0"]
    assert ans.latency_ms is not None


async def test_run_question_nested_response_path():
    runner = PipelineRunner("http://x", '{"query": "{{question}}"}', "data.answer")
    client = FakeClient({"data": {"answer": "nested"}})
    ans = await runner.run_question(_q(), client)
    assert ans.answer == "nested"


async def test_run_question_connect_error():
    runner = PipelineRunner("http://x")
    client = FakeClient(raise_exc=httpx.ConnectError("refused"))
    with pytest.raises(PipelineError):
        await runner.run_question(_q(), client)


async def test_run_all_bounded(mocker):
    questions = [_q(f"q_mh_{i:03d}") for i in range(1, 4)]
    fake = FakeClient({"answer": "ok"})
    mocker.patch("ragprobe.runner.httpx.AsyncClient", return_value=fake)
    runner = PipelineRunner("http://x")
    answers = await runner.run_all(questions, concurrency=2)
    assert len(answers) == 3
    assert all(a.answer == "ok" for a in answers)
    assert len(fake.calls) == 3


def test_grader_parses_pass(mocker):
    mocker.patch(
        "litellm.completion",
        return_value=fake_llm_response("VERDICT: PASS\nREASON: answer is correct"),
    )
    grader = AnswerGrader("openai/gpt-4o-mini")
    result = grader.grade(
        _q(),
        PipelineAnswer(question_id="q_mh_001", question="q", answer="a"),
        [Chunk(id="c0_0", text="source text")],
    )
    assert isinstance(result, GradedResult)
    assert result.passed is True
    assert "correct" in result.grade_reasoning


def test_grader_parses_fail(mocker):
    mocker.patch(
        "litellm.completion",
        return_value=fake_llm_response("VERDICT: FAIL\nREASON: missing detail"),
    )
    grader = AnswerGrader("openai/gpt-4o-mini")
    result = grader.grade(
        _q(),
        PipelineAnswer(question_id="q_mh_001", question="q", answer="a"),
        [Chunk(id="c0_0", text="source")],
    )
    assert result.passed is False


def test_build_run_report_aggregates():
    results = [
        GradedResult(
            question_id="q1", question="q", failure_mode="multi_hop",
            answer="a", passed=True, grade_reasoning="ok",
        ),
        GradedResult(
            question_id="q2", question="q", failure_mode="multi_hop",
            answer="a", passed=False, grade_reasoning="no",
        ),
        GradedResult(
            question_id="q3", question="q", failure_mode="distractor",
            answer="a", passed=True, grade_reasoning="ok",
        ),
    ]
    report = build_run_report(results, pipeline_url="http://x", run_id="run_test")
    assert report.total_questions == 3
    assert report.passed == 2
    assert report.pass_rate == pytest.approx(2 / 3, abs=1e-4)
    assert report.by_failure_mode["multi_hop"]["pass_rate"] == pytest.approx(0.5)
    assert report.by_failure_mode["distractor"]["pass_rate"] == pytest.approx(1.0)


def test_make_run_id_format():
    assert make_run_id().startswith("run_")
