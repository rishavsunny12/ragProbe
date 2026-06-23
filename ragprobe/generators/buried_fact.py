"""Buried-fact question generator.

Targets a chunk that sits inside a dense cluster of near-duplicate neighbors.
The unique fact in the target chunk is easily drowned out by look-alike
chunks during retrieval. The question is verified against the target chunk and
its most-similar distractor neighbor to ensure the answer is specific to the
target.
"""

from __future__ import annotations

import numpy as np

from ..models import Chunk, Question
from .base import BaseGenerator, GenerationError

PROMPT = """\
You are generating evaluation questions for a RAG system.

TARGET chunk (the answer must come from THIS chunk):
---
{target_text}
---

These are similar-looking DISTRACTOR chunks that a retriever may confuse with the target:
---
{distractor_texts}
---

Write ONE question that:
1. Can be answered ONLY from the TARGET chunk, using a specific fact unique to it
2. Would be tempting to (incorrectly) answer from one of the distractor chunks
3. A real user would naturally ask

Output only the question text, nothing else.
"""


class BuriedFactGenerator(BaseGenerator):
    id_prefix = "bf"
    failure_mode = "buried_fact"

    def generate(self, count: int) -> list[Question]:
        candidates = self.graph.get_buried_fact_candidates(distractor_threshold=0.75)
        questions: list[Question] = []
        self.discarded = 0
        cap = self._attempt_cap(count)

        for attempt, target in enumerate(candidates):
            if len(questions) >= count or attempt >= cap:
                break

            neighbors = self._top_neighbors(target, n=3)
            if not neighbors:
                continue

            distractor_texts = "\n---\n".join(c.text for c in neighbors)
            prompt = PROMPT.format(
                target_text=target.text,
                distractor_texts=distractor_texts,
            )
            try:
                question_text = self._call_llm(prompt)
            except GenerationError:
                continue

            # Verify the question is specific to the target vs its top distractor:
            # answerable from target (chunk A) but NOT from the distractor (chunk B).
            top_distractor = neighbors[0]
            if not self._verify_buried(question_text, target, top_distractor):
                self.discarded += 1
                continue

            sim = self.graph.similarity(target.id, top_distractor.id)
            n = len(questions) + 1
            questions.append(
                Question(
                    id=self._make_id(n),
                    question=question_text,
                    failure_mode=self.failure_mode,
                    source_chunk_ids=[target.id],
                    expected_difficulty=_difficulty(len(neighbors)),
                    generation_metadata={
                        "distractor_chunk_ids": [c.id for c in neighbors],
                        "top_distractor_similarity": round(sim, 4),
                    },
                )
            )

        return questions

    def _top_neighbors(self, target: Chunk, n: int) -> list[Chunk]:
        """Return the ``n`` most similar other chunks to ``target``."""
        assert self.graph.sim_matrix is not None
        idx = self.graph.id_to_index[target.id]
        row = self.graph.sim_matrix[idx].copy()
        row[idx] = -np.inf
        order = np.argsort(row)[::-1][:n]
        return [self.graph.chunks[j] for j in order]

    def _verify_buried(self, question: str, target: Chunk, distractor: Chunk) -> bool:
        """Keep only if answerable from the target but NOT the distractor."""
        # _self_verify returns True when NEITHER chunk alone answers it; here we
        # need target-alone == YES and distractor-alone == NO, so we parse directly.
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


def _difficulty(neighbor_count: int) -> float:
    """More near-duplicate neighbors => more buried => harder."""
    return round(min(1.0, 0.5 + 0.1 * neighbor_count), 4)
