"""Multi-hop question generator.

Targets pairs of chunks that are semantically distant (low cosine similarity)
yet connected by a path in the graph: a real question can span both, but a
retriever is unlikely to surface them together. Each candidate question is
self-verified to confirm it cannot be answered from either chunk alone.
"""

from __future__ import annotations

from ..models import Chunk, Question
from .base import BaseGenerator, GenerationError

PROMPT = """\
You are generating evaluation questions for a RAG system.

Chunk A:
---
{chunk_a_text}
---

Chunk B:
---
{chunk_b_text}
---

Semantic similarity between these chunks: {similarity:.2f} (0=completely different, 1=identical)

Write ONE question that:
1. Can ONLY be fully answered by combining information from BOTH chunks
2. Cannot be answered from either chunk alone
3. A real user would naturally ask
4. Does NOT contain unique keywords that only appear in one chunk

Output only the question text, nothing else.
"""


class MultiHopGenerator(BaseGenerator):
    id_prefix = "mh"
    failure_mode = "multi_hop"

    def generate(self, count: int) -> list[Question]:
        candidates = self.graph.get_multihop_candidates(min_distance=0.4)
        questions: list[Question] = []
        self.discarded = 0
        cap = self._attempt_cap(count)

        for attempt, (chunk_a, chunk_b) in enumerate(candidates):
            if len(questions) >= count or attempt >= cap:
                break

            similarity = self.graph.similarity(chunk_a.id, chunk_b.id)
            prompt = PROMPT.format(
                chunk_a_text=chunk_a.text,
                chunk_b_text=chunk_b.text,
                similarity=similarity,
            )
            try:
                question_text = self._call_llm(prompt)
            except GenerationError:
                continue

            if not self._self_verify(question_text, chunk_a, chunk_b):
                self.discarded += 1
                continue

            n = len(questions) + 1
            questions.append(
                Question(
                    id=self._make_id(n),
                    question=question_text,
                    failure_mode=self.failure_mode,
                    source_chunk_ids=[chunk_a.id, chunk_b.id],
                    expected_difficulty=_difficulty(similarity),
                    generation_metadata={"chunk_similarity": round(similarity, 4)},
                )
            )

        return questions


def _difficulty(similarity: float) -> float:
    """Lower similarity => harder multi-hop. Map sim in [0, 0.4] to ~[1.0, 0.6]."""
    return round(min(1.0, max(0.0, 1.0 - similarity)), 4)
