"""Pipeline caller and answer grader.

``PipelineRunner`` sends each question to the user's RAG pipeline over HTTP
(async, with bounded concurrency) and extracts the answer. ``AnswerGrader``
uses an LLM judge to decide PASS/FAIL for each answer against the source chunks
the question was generated from.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx

from datetime import datetime, timezone

from .generators.base import _extract_content
from .models import Chunk, GradedResult, PipelineAnswer, Question, RunReport


class PipelineError(RuntimeError):
    """Raised when the pipeline URL is unreachable or returns a bad response."""


class PipelineRunner:
    def __init__(
        self,
        pipeline_url: str,
        request_template: str = '{"query": "{{question}}"}',
        response_path: str = "answer",
        timeout: float = 30.0,
    ):
        self.pipeline_url = pipeline_url
        self.request_template = request_template
        self.response_path = response_path
        self.timeout = timeout

    def _build_body(self, question: str) -> dict:
        escaped = json.dumps(question)[1:-1]  # escape quotes/backslashes/newlines
        filled = self.request_template.replace("{{question}}", escaped)
        try:
            return json.loads(filled)
        except json.JSONDecodeError as e:
            raise PipelineError(
                f"Request template did not produce valid JSON after substitution: {e}"
            ) from e

    async def run_question(
        self, question: Question, client: httpx.AsyncClient
    ) -> PipelineAnswer:
        body = self._build_body(question.question)
        start = time.perf_counter()
        try:
            resp = await client.post(self.pipeline_url, json=body)
            resp.raise_for_status()
        except httpx.ConnectError as e:
            raise PipelineError(
                f"Could not connect to pipeline at {self.pipeline_url}. "
                f"Is the server running? ({e})"
            ) from e
        except httpx.HTTPStatusError as e:
            raise PipelineError(
                f"Pipeline returned HTTP {e.response.status_code} for "
                f"question {question.id}."
            ) from e
        latency_ms = (time.perf_counter() - start) * 1000.0

        payload = resp.json()
        answer = _extract_by_path(payload, self.response_path)
        retrieved = _extract_retrieved_ids(payload)

        return PipelineAnswer(
            question_id=question.id,
            question=question.question,
            answer="" if answer is None else str(answer),
            retrieved_chunk_ids=retrieved,
            latency_ms=round(latency_ms, 2),
        )

    async def run_all(
        self, questions: list[Question], concurrency: int = 3
    ) -> list[PipelineAnswer]:
        semaphore = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=5.0)
        ) as client:

            async def _bounded(q: Question) -> PipelineAnswer:
                async with semaphore:
                    return await self.run_question(q, client)

            return await asyncio.gather(*(_bounded(q) for q in questions))


class AnswerGrader:
    GRADER_PROMPT = """\
Question: {question}
Source material (correct answer comes from here):
{chunk_texts}

Given answer: {answer}

Does the given answer correctly and completely address the question based on the source material?
VERDICT: PASS or FAIL
REASON: (one sentence explaining why)
"""

    def __init__(self, grader_llm: str = "openai/gpt-4o-mini"):
        self.grader_llm = grader_llm

    def grade(
        self,
        question: Question,
        answer: PipelineAnswer,
        source_chunks: list[Chunk],
    ) -> GradedResult:
        chunk_texts = "\n---\n".join(c.text for c in source_chunks)
        prompt = self.GRADER_PROMPT.format(
            question=question.question,
            chunk_texts=chunk_texts,
            answer=answer.answer,
        )
        response = self._call_llm(prompt)
        passed, reason = _parse_verdict(response)

        return GradedResult(
            question_id=question.id,
            question=question.question,
            failure_mode=question.failure_mode,
            answer=answer.answer,
            passed=passed,
            grade_reasoning=reason,
            retrieved_chunk_ids=answer.retrieved_chunk_ids,
            latency_ms=answer.latency_ms,
        )

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        import litellm

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = litellm.completion(
                    model=self.grader_llm,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
                return _extract_content(resp).strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
        raise PipelineError(f"Grader LLM failed after {max_retries} attempts: {last_err}")


# -- helpers -------------------------------------------------------------


def _extract_by_path(payload, path: str):
    """Walk a dot-notation path (e.g. 'data.answer') into a JSON payload."""
    if not path:
        return payload
    current = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _extract_retrieved_ids(payload) -> list[str]:
    """Best-effort extraction of retrieved chunk ids from common response keys."""
    if not isinstance(payload, dict):
        return []
    for key in ("retrieved_chunk_ids", "chunk_ids", "sources", "retrieved"):
        value = payload.get(key)
        if isinstance(value, list):
            ids: list[str] = []
            for item in value:
                if isinstance(item, str):
                    ids.append(item)
                elif isinstance(item, dict) and "id" in item:
                    ids.append(str(item["id"]))
            if ids:
                return ids
    return []


def make_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"run_{now.strftime('%Y%m%d_%H%M%S')}"


def build_run_report(
    results: list[GradedResult],
    pipeline_url: str,
    run_id: str | None = None,
    timestamp: str | None = None,
) -> RunReport:
    """Aggregate graded results into a RunReport with per-failure-mode rollups."""
    now = datetime.now(timezone.utc)
    run_id = run_id or make_run_id(now)
    timestamp = timestamp or now.strftime("%Y-%m-%dT%H:%M:%SZ")

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    by_mode: dict[str, dict] = {}
    for r in results:
        bucket = by_mode.setdefault(
            r.failure_mode, {"total": 0, "passed": 0, "pass_rate": 0.0}
        )
        bucket["total"] += 1
        if r.passed:
            bucket["passed"] += 1
    for bucket in by_mode.values():
        bucket["pass_rate"] = (
            round(bucket["passed"] / bucket["total"], 4) if bucket["total"] else 0.0
        )

    return RunReport(
        run_id=run_id,
        timestamp=timestamp,
        pipeline_url=pipeline_url,
        total_questions=total,
        passed=passed,
        failed=failed,
        pass_rate=round(passed / total, 4) if total else 0.0,
        by_failure_mode=by_mode,
        results=results,
    )


def _parse_verdict(text: str) -> tuple[bool, str]:
    passed = False
    reason = ""
    for line in text.splitlines():
        stripped = line.strip().replace("**", "").strip()
        upper = stripped.upper()
        if upper.startswith("VERDICT"):
            value = stripped.split(":", 1)[-1].strip().upper()
            passed = value.startswith("PASS")
        elif upper in ("PASS", "FAIL"):
            passed = upper == "PASS"
        elif upper.startswith("REASON"):
            reason = stripped.split(":", 1)[-1].strip()
    if not reason:
        reason = text.strip()[:200]
    return passed, reason
