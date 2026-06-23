"""BaseGenerator: shared LLM plumbing for all question generators.

Provides:
  * ``_call_llm``     - litellm completion with retry + exponential backoff
  * ``_self_verify``  - LLM self-check that a question genuinely needs both chunks
  * id / difficulty helpers
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from ..graph import ChunkGraph
from ..models import Chunk, Question

VERIFY_PROMPT = """\
Question: {question}

Chunk A: {chunk_a_text}
Chunk B: {chunk_b_text}

Answer these three questions exactly:
CHUNK_A_ALONE: YES or NO (can the question be fully answered from Chunk A alone?)
CHUNK_B_ALONE: YES or NO (can the question be fully answered from Chunk B alone?)
REQUIRES_BOTH: YES or NO (does answering require combining both chunks?)
"""

# Verification/grounding answers are just a few labelled YES/NO lines, so keep
# the output tiny and deterministic -- this is the single biggest lever on
# generation speed (an unbounded verify response can take tens of seconds).
VERIFY_MAX_TOKENS = 96
VERIFY_TEMPERATURE = 0.0


class GenerationError(RuntimeError):
    """Raised when the LLM cannot be reached after all retries."""


class BaseGenerator(ABC):
    #: short id fragment, e.g. "mh" -> question ids like "q_mh_001"
    id_prefix: str = "gen"
    failure_mode: str = "generic"

    def __init__(self, graph: ChunkGraph, llm_model: str):
        self.graph = graph
        self.llm_model = llm_model
        #: per-generator count of questions discarded by self-verification
        self.discarded = 0

    @abstractmethod
    def generate(self, count: int) -> list[Question]:
        """Produce up to ``count`` verified questions."""

    # -- LLM plumbing ----------------------------------------------------

    def _call_llm(
        self,
        prompt: str,
        max_retries: int = 3,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Call the configured LLM, retrying with exponential backoff.

        ``max_tokens`` caps the output so a verbose local model cannot ramble
        toward the context limit (which would make a single call take tens of
        seconds). Questions and verification verdicts are short, so this bound
        is generous.

        Raises GenerationError after ``max_retries`` consecutive failures.
        """
        import litellm

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = litellm.completion(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return _extract_content(resp).strip()
            except Exception as e:  # noqa: BLE001 - retry on any provider error
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # 1s, 2s, 4s ...
        raise GenerationError(
            f"LLM call failed after {max_retries} attempts: {last_err}"
        )

    def _self_verify(self, question: str, chunk_a: Chunk, chunk_b: Chunk) -> bool:
        """Return True only if the question requires BOTH chunks.

        Asks the LLM whether the question can be answered from either chunk
        alone; returns False (discard) if either answer is YES.
        """
        prompt = VERIFY_PROMPT.format(
            question=question,
            chunk_a_text=chunk_a.text,
            chunk_b_text=chunk_b.text,
        )
        try:
            response = self._call_llm(
                prompt, max_tokens=VERIFY_MAX_TOKENS, temperature=VERIFY_TEMPERATURE
            )
        except GenerationError:
            return False

        a_alone = _parse_yes_no(response, "CHUNK_A_ALONE")
        b_alone = _parse_yes_no(response, "CHUNK_B_ALONE")
        return a_alone is False and b_alone is False

    # -- helpers ---------------------------------------------------------

    def _make_id(self, n: int) -> str:
        return f"q_{self.id_prefix}_{n:03d}"

    @staticmethod
    def _attempt_cap(count: int) -> int:
        """Max candidates to try before giving up.

        On highly repetitive corpora, self-verification can discard most
        candidates (e.g. when neighbors are near-duplicates). This bounds the
        worst-case number of LLM calls so generation can't run unbounded.
        """
        return count * 8 + 10


def _extract_content(resp) -> str:
    """Pull the text content out of a litellm/openai-style response object."""
    try:
        return resp.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        pass
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:  # pragma: no cover
        raise GenerationError(f"Unexpected LLM response shape: {resp!r}") from e


def _parse_yes_no(text: str, label: str) -> bool | None:
    """Find ``label: YES/NO`` in ``text``; return True/False/None.

    Local models (e.g. Gemma via Ollama) often wrap labels in markdown bold
    (**CHUNK_A_ALONE:** NO). Strip decoration before matching.

    When no labelled line is found, accept a leading bare ``YES``/``NO`` line
    (common for single-field prompts like ANSWERABLE).
    """
    label_upper = label.upper()
    for line in text.splitlines():
        stripped = line.strip().replace("**", "").strip()
        if not stripped.upper().startswith(label_upper):
            continue
        value = stripped.split(":", 1)[-1].strip().upper()
        if value.startswith("YES"):
            return True
        if value.startswith("NO"):
            return False
    for line in text.splitlines():
        stripped = line.strip().replace("**", "").strip().upper()
        if stripped in ("YES", "NO"):
            return stripped == "YES"
    return None
