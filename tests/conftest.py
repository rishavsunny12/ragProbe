"""Shared pytest fixtures and helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ragprobe.graph import ChunkGraph

from .fixtures.synthetic import make_synthetic_chunks


@pytest.fixture
def synthetic_chunks():
    return make_synthetic_chunks()


@pytest.fixture
def built_graph(synthetic_chunks):
    # small k so the test corpus forms meaningful sparse structure
    graph = ChunkGraph(synthetic_chunks, k=3)
    graph.build()
    return graph


def fake_llm_response(content: str):
    """Build an object shaped like a litellm/openai completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def make_llm_side_effect(question_text: str, verify_text: str):
    """Return a side_effect for litellm.completion.

    Returns ``verify_text`` for self-verification/grounding prompts and
    ``question_text`` for generation prompts, based on prompt content.
    """

    def _side_effect(*args, **kwargs):
        messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
        prompt = messages[0]["content"] if messages else ""
        if "CHUNK_A_ALONE" in prompt or "ANSWERABLE" in prompt:
            return fake_llm_response(verify_text)
        return fake_llm_response(question_text)

    return _side_effect
