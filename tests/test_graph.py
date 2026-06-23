"""Tests for the ChunkGraph topology queries."""

from __future__ import annotations

import numpy as np
import pytest

from ragprobe.graph import ChunkGraph

from .fixtures.synthetic import BRIDGE_IDS, make_synthetic_chunks


def test_build_produces_similarity_matrix(built_graph):
    n = len(built_graph.chunks)
    assert built_graph.sim_matrix is not None
    assert built_graph.sim_matrix.shape == (n, n)
    # diagonal is self-similarity ~ 1.0
    assert np.allclose(np.diag(built_graph.sim_matrix), 1.0, atol=1e-6)
    # symmetric
    assert np.allclose(built_graph.sim_matrix, built_graph.sim_matrix.T, atol=1e-6)


def test_empty_chunks_raises():
    with pytest.raises(ValueError):
        ChunkGraph([])


def test_similarity_symmetric(built_graph):
    a, b = built_graph.chunks[0].id, built_graph.chunks[7].id
    assert built_graph.similarity(a, b) == pytest.approx(
        built_graph.similarity(b, a), abs=1e-9
    )


def test_buried_fact_candidates_have_dense_neighborhoods(built_graph):
    candidates = built_graph.get_buried_fact_candidates(distractor_threshold=0.75)
    assert candidates, "expected at least one buried-fact candidate"
    sim = built_graph.sim_matrix
    for chunk in candidates:
        idx = built_graph.id_to_index[chunk.id]
        high = int(np.sum(sim[idx] > 0.75)) - 1  # exclude self
        assert high >= 3


def test_distractor_pairs_within_band(built_graph):
    pairs = built_graph.get_distractor_pairs(low=0.65, high=0.85)
    assert pairs, "expected at least one distractor pair"
    sims = [built_graph.similarity(a.id, b.id) for a, b in pairs]
    assert all(0.65 <= s <= 0.85 for s in sims)
    # sorted descending (most confusing first)
    assert sims == sorted(sims, reverse=True)


def test_multihop_candidates_are_distant_and_connected(built_graph):
    pairs = built_graph.get_multihop_candidates(min_distance=0.4)
    assert pairs, "expected at least one multi-hop candidate"
    sims = [built_graph.similarity(a.id, b.id) for a, b in pairs]
    assert all(s < 0.4 for s in sims)
    # sorted ascending (most distant first)
    assert sims == sorted(sims)


def test_near_miss_top_candidate_is_a_bridge(built_graph):
    candidates = built_graph.get_near_miss_candidates()
    assert candidates
    # the single highest-betweenness chunk should be one of the chain bridges
    assert candidates[0].id in BRIDGE_IDS


def test_summary_reports_topology(built_graph):
    summary = built_graph.summary()
    assert summary["chunk_count"] == len(built_graph.chunks)
    assert summary["connected_components"] >= 1
    assert "isolated_chunks" in summary


def test_build_without_precomputed_embeddings_uses_model(monkeypatch):
    """When embeddings are missing, build() must call the model encoder."""
    chunks = make_synthetic_chunks()
    for c in chunks:
        c.embedding = None

    captured = {}

    class FakeModel:
        def encode(self, texts, **kwargs):
            captured["called"] = True
            return np.random.default_rng(1).normal(0, 1, (len(texts), 8))

    graph = ChunkGraph(chunks, k=3)
    monkeypatch.setattr(ChunkGraph, "model", property(lambda self: FakeModel()))
    graph.build()
    assert captured.get("called") is True
    assert graph.sim_matrix is not None
