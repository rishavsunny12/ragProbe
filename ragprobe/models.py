"""Pydantic v2 data models for RAGProbe.

These models define the data contracts that flow between every stage of the
pipeline (index -> generate -> run -> diff) and the on-disk file formats.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FailureMode = Literal["multi_hop", "buried_fact", "distractor", "near_miss"]

ALL_FAILURE_MODES: tuple[FailureMode, ...] = (
    "multi_hop",
    "buried_fact",
    "distractor",
    "near_miss",
)


class Chunk(BaseModel):
    """A single chunk of source text from the corpus."""

    id: str
    text: str
    metadata: dict = Field(default_factory=dict)
    embedding: list[float] | None = None


class Question(BaseModel):
    """An adversarial evaluation question generated from the chunk graph."""

    id: str  # e.g. "q_mh_001"
    question: str
    failure_mode: FailureMode
    source_chunk_ids: list[str]
    expected_difficulty: float  # 0.0-1.0
    generation_metadata: dict = Field(default_factory=dict)


class PipelineAnswer(BaseModel):
    """The raw answer returned by the user's RAG pipeline for one question."""

    question_id: str
    question: str
    answer: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: float | None = None


class GradedResult(BaseModel):
    """A pipeline answer after it has been graded PASS/FAIL by the LLM judge."""

    question_id: str
    question: str
    failure_mode: str
    answer: str
    passed: bool
    grade_reasoning: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: float | None = None


class RunReport(BaseModel):
    """A full report summarizing one run of all questions against a pipeline."""

    run_id: str
    timestamp: str  # ISO 8601 e.g. "2026-06-21T14:30:22Z"
    pipeline_url: str
    total_questions: int
    passed: int
    failed: int
    pass_rate: float
    by_failure_mode: dict[str, dict]  # {"multi_hop": {"total": 10, "passed": 4, "pass_rate": 0.4}}
    results: list[GradedResult]


class DiffReport(BaseModel):
    """The difference between a baseline run and a current run."""

    baseline_run_id: str
    current_run_id: str
    overall_delta: float
    regressed_failure_modes: list[str]
    new_failures: list[GradedResult]
    new_passes: list[GradedResult]
    regression_detected: bool
