"""Near-miss boundary question generator.

Targets high-betweenness "chokepoint" chunks: nodes that sit on many shortest
paths between other chunks. In retrieval these chunks tend to land right at the
rank boundary (just inside or just outside the top-k), making them fragile and
easy to miss. The question is verified to be genuinely grounded in (answerable
from) the target chunk.
"""

from __future__ import annotations

from ..models import Chunk, Question
from .base import BaseGenerator, GenerationError

PROMPT = """\
You are generating evaluation questions for a RAG system.

This chunk is a retrieval "chokepoint" that connects several different topics:
---
{chunk_text}
---

Write ONE question that:
1. Is answered by a specific fact in THIS chunk
2. Uses natural phrasing that does NOT copy the chunk's exact distinctive keywords
   (so retrieval must rely on meaning, not keyword overlap)
3. A real user would naturally ask

Output only the question text, nothing else.
"""

GROUNDING_PROMPT = """\
Question: {question}

Chunk:
---
{chunk_text}
---

Can the question be fully and correctly answered using only this chunk?
ANSWERABLE: YES or NO
"""


class NearMissGenerator(BaseGenerator):
    id_prefix = "nm"
    failure_mode = "near_miss"

    def generate(self, count: int) -> list[Question]:
        candidates = self.graph.get_near_miss_candidates()
        questions: list[Question] = []
        self.discarded = 0
        cap = self._attempt_cap(count)

        for attempt, chunk in enumerate(candidates):
            if len(questions) >= count or attempt >= cap:
                break

            prompt = PROMPT.format(chunk_text=chunk.text)
            try:
                question_text = self._call_llm(prompt)
            except GenerationError:
                continue

            if not self._verify_grounded(question_text, chunk):
                self.discarded += 1
                continue

            betweenness = self.graph.betweenness.get(chunk.id, 0.0)
            n = len(questions) + 1
            questions.append(
                Question(
                    id=self._make_id(n),
                    question=question_text,
                    failure_mode=self.failure_mode,
                    source_chunk_ids=[chunk.id],
                    expected_difficulty=_difficulty(betweenness),
                    generation_metadata={"betweenness": round(betweenness, 6)},
                )
            )

        return questions

    def _verify_grounded(self, question: str, chunk: Chunk) -> bool:
        from .base import VERIFY_MAX_TOKENS, VERIFY_TEMPERATURE, _parse_yes_no

        prompt = GROUNDING_PROMPT.format(question=question, chunk_text=chunk.text)
        try:
            response = self._call_llm(
                prompt, max_tokens=VERIFY_MAX_TOKENS, temperature=VERIFY_TEMPERATURE
            )
        except GenerationError:
            return False
        return _parse_yes_no(response, "ANSWERABLE") is True


def _difficulty(betweenness: float) -> float:
    """Higher betweenness => more of a chokepoint => harder near-miss."""
    return round(min(1.0, 0.6 + betweenness), 4)
