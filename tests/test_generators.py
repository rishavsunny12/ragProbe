"""Tests for the question generators (all LLM calls mocked)."""

from __future__ import annotations

from ragprobe.generators import (
    BuriedFactGenerator,
    DistractorGenerator,
    MultiHopGenerator,
    NearMissGenerator,
)
from ragprobe.generators.base import _parse_yes_no

from .conftest import fake_llm_response, make_llm_side_effect


def test_multihop_generates_verified_questions(built_graph, mocker):
    side_effect = make_llm_side_effect(
        question_text="What links topic 0 and topic 3?",
        verify_text="CHUNK_A_ALONE: NO\nCHUNK_B_ALONE: NO\nREQUIRES_BOTH: YES",
    )
    mocker.patch("litellm.completion", side_effect=side_effect)

    gen = MultiHopGenerator(built_graph, "openai/gpt-4o-mini")
    questions = gen.generate(2)

    assert len(questions) == 2
    assert all(q.failure_mode == "multi_hop" for q in questions)
    assert all(len(q.source_chunk_ids) == 2 for q in questions)
    assert questions[0].id == "q_mh_001"
    assert gen.discarded == 0
    assert "chunk_similarity" in questions[0].generation_metadata


def test_multihop_discards_when_single_chunk_answers(built_graph, mocker):
    side_effect = make_llm_side_effect(
        question_text="A leaky question.",
        verify_text="CHUNK_A_ALONE: YES\nCHUNK_B_ALONE: NO\nREQUIRES_BOTH: NO",
    )
    mocker.patch("litellm.completion", side_effect=side_effect)

    gen = MultiHopGenerator(built_graph, "openai/gpt-4o-mini")
    questions = gen.generate(2)

    assert questions == []
    assert gen.discarded > 0


def test_multihop_skips_on_llm_failure(built_graph, mocker):
    mocker.patch("litellm.completion", side_effect=RuntimeError("boom"))
    mocker.patch("time.sleep")  # don't actually back off
    gen = MultiHopGenerator(built_graph, "openai/gpt-4o-mini")
    questions = gen.generate(2)
    assert questions == []


def test_buried_fact_requires_target_specificity(built_graph, mocker):
    side_effect = make_llm_side_effect(
        question_text="What specific fact is unique to this chunk?",
        verify_text="CHUNK_A_ALONE: YES\nCHUNK_B_ALONE: NO\nREQUIRES_BOTH: NO",
    )
    mocker.patch("litellm.completion", side_effect=side_effect)

    gen = BuriedFactGenerator(built_graph, "openai/gpt-4o-mini")
    questions = gen.generate(2)

    assert len(questions) >= 1
    assert all(q.failure_mode == "buried_fact" for q in questions)
    assert all(len(q.source_chunk_ids) == 1 for q in questions)
    assert "distractor_chunk_ids" in questions[0].generation_metadata


def test_distractor_keeps_target_only_questions(built_graph, mocker):
    side_effect = make_llm_side_effect(
        question_text="Which detail belongs to the target?",
        verify_text="CHUNK_A_ALONE: YES\nCHUNK_B_ALONE: NO",
    )
    mocker.patch("litellm.completion", side_effect=side_effect)

    gen = DistractorGenerator(built_graph, "openai/gpt-4o-mini")
    questions = gen.generate(2)

    assert len(questions) >= 1
    assert all(q.failure_mode == "distractor" for q in questions)
    assert "distractor_chunk_id" in questions[0].generation_metadata


def test_near_miss_grounding(built_graph, mocker):
    side_effect = make_llm_side_effect(
        question_text="What does this chokepoint chunk state?",
        verify_text="ANSWERABLE: YES",
    )
    mocker.patch("litellm.completion", side_effect=side_effect)

    gen = NearMissGenerator(built_graph, "openai/gpt-4o-mini")
    questions = gen.generate(2)

    assert len(questions) >= 1
    assert all(q.failure_mode == "near_miss" for q in questions)
    assert "betweenness" in questions[0].generation_metadata


def test_parse_yes_no():
    text = "CHUNK_A_ALONE: NO\nCHUNK_B_ALONE: YES"
    assert _parse_yes_no(text, "CHUNK_A_ALONE") is False
    assert _parse_yes_no(text, "CHUNK_B_ALONE") is True
    assert _parse_yes_no(text, "MISSING") is None

    # Gemma / Ollama often bolds the labels
    gemma = "**CHUNK_A_ALONE:** NO\n\n**CHUNK_B_ALONE:** NO\n\n**REQUIRES_BOTH:** YES"
    assert _parse_yes_no(gemma, "CHUNK_A_ALONE") is False
    assert _parse_yes_no(gemma, "CHUNK_B_ALONE") is False

    assert _parse_yes_no("NO\n\nThe chunk does not contain enough info.", "ANSWERABLE") is False
    assert _parse_yes_no("YES\n\nGrounded in the chunk.", "ANSWERABLE") is True


def test_retry_backoff_then_success(built_graph, mocker):
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return fake_llm_response("ok")

    mocker.patch("litellm.completion", side_effect=flaky)
    mocker.patch("time.sleep")
    gen = MultiHopGenerator(built_graph, "openai/gpt-4o-mini")
    assert gen._call_llm("hello") == "ok"
    assert calls["n"] == 2
