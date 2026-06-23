"""Calibration: prove topology-aware questions are harder than random ones.

Generates three question sets from the same corpus:
  1. baseline random        - a simple question from a single random chunk
  2. random multi-question  - a question from a random (untargeted) chunk pair
  3. RAGProbe topology-hard - questions from the real graph-aware generators

Each set is run against the pipeline and graded. If the topology-hard set is
not meaningfully harder (pass-rate delta < 10 percentage points) than the
easiest random set, a calibration warning is emitted.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

from .generators import GENERATORS
from .generators.base import BaseGenerator, GenerationError
from .graph import ChunkGraph
from .models import Chunk, GradedResult, Question
from .runner import AnswerGrader, PipelineRunner

CALIBRATION_DELTA_THRESHOLD_PP = 10.0

RANDOM_BASELINE_PROMPT = """\
Write ONE simple, factual question that can be answered directly from this text.
Output only the question text, nothing else.

Text:
---
{chunk_text}
---
"""

RANDOM_MULTI_PROMPT = """\
Write ONE question related to the following two pieces of text.
Output only the question text, nothing else.

Text A:
---
{chunk_a_text}
---
Text B:
---
{chunk_b_text}
---
"""


class _RandomBaselineGenerator(BaseGenerator):
    id_prefix = "rand"
    failure_mode = "random_baseline"

    def generate(self, count: int) -> list[Question]:
        rng = random.Random(13)
        chunks = list(self.graph.chunks)
        rng.shuffle(chunks)
        out: list[Question] = []
        for chunk in chunks:
            if len(out) >= count:
                break
            try:
                text = self._call_llm(RANDOM_BASELINE_PROMPT.format(chunk_text=chunk.text))
            except GenerationError:
                continue
            out.append(
                Question(
                    id=self._make_id(len(out) + 1),
                    question=text,
                    failure_mode=self.failure_mode,
                    source_chunk_ids=[chunk.id],
                    expected_difficulty=0.2,
                )
            )
        return out


class _RandomMultiGenerator(BaseGenerator):
    id_prefix = "rmulti"
    failure_mode = "random_multi"

    def generate(self, count: int) -> list[Question]:
        rng = random.Random(29)
        chunks = list(self.graph.chunks)
        out: list[Question] = []
        attempts = 0
        while len(out) < count and attempts < count * 10 and len(chunks) >= 2:
            attempts += 1
            a, b = rng.sample(chunks, 2)
            try:
                text = self._call_llm(
                    RANDOM_MULTI_PROMPT.format(chunk_a_text=a.text, chunk_b_text=b.text)
                )
            except GenerationError:
                continue
            out.append(
                Question(
                    id=self._make_id(len(out) + 1),
                    question=text,
                    failure_mode=self.failure_mode,
                    source_chunk_ids=[a.id, b.id],
                    expected_difficulty=0.4,
                )
            )
        return out


@dataclass
class CalibrationReport:
    baseline_pass_rate: float
    random_multi_pass_rate: float
    topology_hard_pass_rate: float
    difficulty_delta_pp: float
    warning: bool
    sample_sizes: dict[str, int] = field(default_factory=dict)


def _generate_topology_hard(
    graph: ChunkGraph, llm_model: str, sample_size: int
) -> list[Question]:
    """Round-robin across the four real generators up to ``sample_size``."""
    per_mode = max(1, sample_size // len(GENERATORS))
    questions: list[Question] = []
    for mode, gen_cls in GENERATORS.items():
        gen = gen_cls(graph, llm_model)
        questions.extend(gen.generate(per_mode))
        if len(questions) >= sample_size:
            break
    return questions[:sample_size]


def _grade_set(
    questions: list[Question],
    runner: PipelineRunner,
    grader: AnswerGrader,
    id_to_chunk: dict[str, Chunk],
    concurrency: int,
) -> float:
    if not questions:
        return 0.0
    answers = asyncio.run(runner.run_all(questions, concurrency=concurrency))
    answers_by_id = {a.question_id: a for a in answers}
    graded: list[GradedResult] = []
    for q in questions:
        answer = answers_by_id[q.id]
        source_chunks = [id_to_chunk[cid] for cid in q.source_chunk_ids if cid in id_to_chunk]
        graded.append(grader.grade(q, answer, source_chunks))
    passed = sum(1 for g in graded if g.passed)
    return round(passed / len(graded), 4)


def run_calibration(
    graph: ChunkGraph,
    llm_model: str,
    pipeline_url: str,
    request_template: str = '{"query": "{{question}}"}',
    response_path: str = "answer",
    grader_llm: str = "openai/gpt-4o-mini",
    sample_size: int = 30,
    concurrency: int = 3,
) -> CalibrationReport:
    id_to_chunk = {c.id: c for c in graph.chunks}
    runner = PipelineRunner(pipeline_url, request_template, response_path)
    grader = AnswerGrader(grader_llm)

    baseline_qs = _RandomBaselineGenerator(graph, llm_model).generate(sample_size)
    multi_qs = _RandomMultiGenerator(graph, llm_model).generate(sample_size)
    hard_qs = _generate_topology_hard(graph, llm_model, sample_size)

    baseline_rate = _grade_set(baseline_qs, runner, grader, id_to_chunk, concurrency)
    multi_rate = _grade_set(multi_qs, runner, grader, id_to_chunk, concurrency)
    hard_rate = _grade_set(hard_qs, runner, grader, id_to_chunk, concurrency)

    # "Easiest" random set vs the topology-hard set, in percentage points.
    easiest_random = max(baseline_rate, multi_rate)
    difficulty_delta_pp = round((easiest_random - hard_rate) * 100.0, 2)

    return CalibrationReport(
        baseline_pass_rate=baseline_rate,
        random_multi_pass_rate=multi_rate,
        topology_hard_pass_rate=hard_rate,
        difficulty_delta_pp=difficulty_delta_pp,
        warning=difficulty_delta_pp < CALIBRATION_DELTA_THRESHOLD_PP,
        sample_sizes={
            "baseline": len(baseline_qs),
            "random_multi": len(multi_qs),
            "topology_hard": len(hard_qs),
        },
    )
