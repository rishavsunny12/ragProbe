"""Distractor question generator.

Targets moderately similar chunk pairs (cosine ~0.65-0.85). One chunk holds
the correct answer; the other is similar enough to be retrieved as a tempting
wrong answer. The question is verified to be answerable from the target chunk
but NOT from the distractor.
"""

from __future__ import annotations

from ..models import Chunk, Question
from .base import BaseGenerator, GenerationError

PROMPT = """\
You are generating evaluation questions for a RAG system.

TARGET chunk (the correct answer comes from THIS chunk):
---
{target_text}
---

DISTRACTOR chunk (similar topic, but does NOT answer the question):
---
{distractor_text}
---

Semantic similarity between these chunks: {similarity:.2f}

Write ONE question that:
1. Is answered correctly ONLY by the TARGET chunk
2. Looks like it might be answered by the DISTRACTOR chunk (same topic area)
3. A real user would naturally ask
4. Hinges on a specific detail that differs between the two chunks

Output only the question text, nothing else.
"""


class DistractorGenerator(BaseGenerator):
    id_prefix = "ds"
    failure_mode = "distractor"

    def generate(self, count: int) -> list[Question]:
        pairs = self.graph.get_distractor_pairs(low=0.65, high=0.85)
        questions: list[Question] = []
        self.discarded = 0
        cap = self._attempt_cap(count)

        for attempt, (target, distractor) in enumerate(pairs):
            if len(questions) >= count or attempt >= cap:
                break

            similarity = self.graph.similarity(target.id, distractor.id)
            prompt = PROMPT.format(
                target_text=target.text,
                distractor_text=distractor.text,
                similarity=similarity,
            )
            try:
                question_text = self._call_llm(prompt)
            except GenerationError:
                continue

            if not self._verify_distractor(question_text, target, distractor):
                self.discarded += 1
                continue

            n = len(questions) + 1
            questions.append(
                Question(
                    id=self._make_id(n),
                    question=question_text,
                    failure_mode=self.failure_mode,
                    source_chunk_ids=[target.id],
                    expected_difficulty=_difficulty(similarity),
                    generation_metadata={
                        "distractor_chunk_id": distractor.id,
                        "chunk_similarity": round(similarity, 4),
                    },
                )
            )

        return questions

    def _verify_distractor(
        self, question: str, target: Chunk, distractor: Chunk
    ) -> bool:
        """Keep only if answerable from the target but NOT from the distractor."""
        from .base import (
            VERIFY_MAX_TOKENS,
            VERIFY_PROMPT,
            VERIFY_TEMPERATURE,
            _parse_yes_no,
        )

        prompt = VERIFY_PROMPT.format(
            question=question,
            chunk_a_text=target.text,
            chunk_b_text=distractor.text,
        )
        try:
            response = self._call_llm(
                prompt, max_tokens=VERIFY_MAX_TOKENS, temperature=VERIFY_TEMPERATURE
            )
        except GenerationError:
            return False
        target_alone = _parse_yes_no(response, "CHUNK_A_ALONE")
        distractor_alone = _parse_yes_no(response, "CHUNK_B_ALONE")
        return target_alone is True and distractor_alone is False


def _difficulty(similarity: float) -> float:
    """Higher similarity between target and distractor => more confusing."""
    return round(min(1.0, max(0.0, similarity)), 4)
