"""Tests for the SQLite storage layer."""

from __future__ import annotations

import pytest

from ragprobe.models import Chunk, GradedResult, Question, RunReport
from ragprobe.storage import Storage


def test_chunk_roundtrip(tmp_path):
    db = tmp_path / "index.db"
    chunks = [
        Chunk(id="a", text="alpha", metadata={"k": 1}, embedding=[0.1, 0.2]),
        Chunk(id="b", text="beta", embedding=[0.3, 0.4]),
    ]
    with Storage(db) as store:
        store.save_chunks(chunks)
        assert store.chunk_count() == 2
        loaded = {c.id: c for c in store.load_chunks()}
    assert loaded["a"].text == "alpha"
    assert loaded["a"].metadata == {"k": 1}
    assert loaded["a"].embedding == [0.1, 0.2]


def test_chunk_without_embedding_rejected(tmp_path):
    with Storage(tmp_path / "index.db") as store:
        with pytest.raises(ValueError):
            store.save_chunks([Chunk(id="a", text="x")])


def test_question_roundtrip(tmp_path):
    q = Question(
        id="q_mh_001",
        question="why?",
        failure_mode="multi_hop",
        source_chunk_ids=["a", "b"],
        expected_difficulty=0.7,
    )
    with Storage(tmp_path / "index.db") as store:
        store.save_questions([q])
        loaded = store.load_questions()
    assert len(loaded) == 1
    assert loaded[0].id == "q_mh_001"
    assert loaded[0].source_chunk_ids == ["a", "b"]


def test_run_roundtrip(tmp_path):
    report = RunReport(
        run_id="run_x",
        timestamp="2026-06-21T00:00:00Z",
        pipeline_url="http://x",
        total_questions=1,
        passed=1,
        failed=0,
        pass_rate=1.0,
        by_failure_mode={"multi_hop": {"total": 1, "passed": 1, "pass_rate": 1.0}},
        results=[
            GradedResult(
                question_id="q1", question="q", failure_mode="multi_hop",
                answer="a", passed=True, grade_reasoning="ok",
            )
        ],
    )
    with Storage(tmp_path / "index.db") as store:
        store.save_run(report)
        loaded = store.load_run("run_x")
        assert store.load_run("missing") is None
    assert loaded is not None
    assert loaded.run_id == "run_x"
    assert loaded.pass_rate == 1.0
