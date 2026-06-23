"""Chunk topology graph.

Builds a k-nearest-neighbor graph over chunk embeddings using cosine
similarity, then exposes structural queries that map directly onto the four
adversarial failure modes:

    multi_hop    -> distant pairs that still have a path between them
    buried_fact  -> chunks surrounded by many near-duplicate neighbors
    distractor   -> moderately similar pairs (tempting wrong answers)
    near_miss    -> high-betweenness "chokepoint" chunks

The SentenceTransformer model is loaded lazily, so a graph can be built from
chunks that already carry embeddings (e.g. loaded from SQLite or supplied by
tests) without importing/downloading any model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from .models import Chunk

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_K = 20


class ChunkGraph:
    def __init__(
        self,
        chunks: list[Chunk],
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        k: int = DEFAULT_K,
        batch_size: int = 32,
    ):
        if not chunks:
            raise ValueError("Cannot build a ChunkGraph from zero chunks.")
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.k = k
        self.batch_size = batch_size

        self.graph = nx.DiGraph()
        self.id_to_index: dict[str, int] = {c.id: i for i, c in enumerate(chunks)}
        self.embeddings: np.ndarray | None = None
        self.sim_matrix: np.ndarray | None = None
        self.betweenness: dict[str, float] = {}

        self._model: "SentenceTransformer | None" = None

    # -- model (lazy) ----------------------------------------------------

    @property
    def model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    # -- build -----------------------------------------------------------

    def build(self) -> None:
        """Embed (if needed), build the k-NN graph, and compute centrality."""
        self._ensure_embeddings()
        self._compute_similarity()
        self._build_knn_graph()
        self._compute_betweenness()

    def _ensure_embeddings(self) -> None:
        missing = [c for c in self.chunks if c.embedding is None]
        if missing:
            texts = [c.text for c in missing]
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for chunk, vec in zip(missing, vectors):
                chunk.embedding = [float(x) for x in vec]

        self.embeddings = np.array([c.embedding for c in self.chunks], dtype=np.float64)

    def _compute_similarity(self) -> None:
        assert self.embeddings is not None
        emb = self.embeddings
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12  # avoid divide-by-zero for degenerate vectors
        normalized = emb / norms
        self.sim_matrix = normalized @ normalized.T
        np.clip(self.sim_matrix, -1.0, 1.0, out=self.sim_matrix)

    def _build_knn_graph(self) -> None:
        assert self.sim_matrix is not None
        n = len(self.chunks)
        for c in self.chunks:
            self.graph.add_node(c.id)

        if n < 2:
            return

        k = min(self.k, n - 1)
        sim = self.sim_matrix.copy()
        np.fill_diagonal(sim, -np.inf)  # never link a node to itself

        for i in range(n):
            # indices of the k most similar other chunks
            neighbor_idx = np.argpartition(sim[i], -k)[-k:]
            for j in neighbor_idx:
                weight = float(self.sim_matrix[i, j])
                self.graph.add_edge(
                    self.chunks[i].id, self.chunks[j].id, weight=weight
                )

    def _compute_betweenness(self) -> None:
        # Chokepoints are an undirected notion: a bridge node "lies between"
        # two regions regardless of edge direction. Computing betweenness on the
        # directed k-NN graph would miss bridges, because clusters only point
        # inward (cluster -> bridge edges do not exist), so we project to
        # undirected first. The score is still stored on the directed graph.
        undirected = self.graph.to_undirected()
        self.betweenness = nx.betweenness_centrality(undirected)
        nx.set_node_attributes(self.graph, self.betweenness, name="betweenness")

    # -- helpers ---------------------------------------------------------

    def _chunk(self, idx: int) -> Chunk:
        return self.chunks[idx]

    def similarity(self, id_a: str, id_b: str) -> float:
        assert self.sim_matrix is not None
        return float(self.sim_matrix[self.id_to_index[id_a], self.id_to_index[id_b]])

    def _component_map(self) -> dict[str, int]:
        undirected = self.graph.to_undirected()
        comp_map: dict[str, int] = {}
        for comp_id, component in enumerate(nx.connected_components(undirected)):
            for node in component:
                comp_map[node] = comp_id
        return comp_map

    # -- candidate queries (one per failure mode) ------------------------

    def get_multihop_candidates(
        self, min_distance: float = 0.4
    ) -> list[tuple[Chunk, Chunk]]:
        """Pairs with cosine similarity < ``min_distance`` that have a path
        between them, sorted by ascending similarity (most distant first)."""
        assert self.sim_matrix is not None
        n = len(self.chunks)
        comp_map = self._component_map()

        candidates: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(self.sim_matrix[i, j])
                if sim >= min_distance:
                    continue
                a_id, b_id = self.chunks[i].id, self.chunks[j].id
                # same connected component => a path exists
                if comp_map.get(a_id) != comp_map.get(b_id):
                    continue
                candidates.append((sim, i, j))

        candidates.sort(key=lambda t: t[0])  # ascending similarity
        return [(self._chunk(i), self._chunk(j)) for _, i, j in candidates]

    def get_buried_fact_candidates(
        self, distractor_threshold: float = 0.75
    ) -> list[Chunk]:
        """Chunks with 3+ neighbors above ``distractor_threshold``, sorted by
        descending neighbor count (most buried first)."""
        assert self.sim_matrix is not None
        n = len(self.chunks)
        scored: list[tuple[int, int]] = []
        for i in range(n):
            row = self.sim_matrix[i]
            # count neighbors above threshold, excluding self
            count = int(np.sum(row > distractor_threshold)) - 1
            if count >= 3:
                scored.append((count, i))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [self._chunk(i) for _, i in scored]

    def get_distractor_pairs(
        self, low: float = 0.65, high: float = 0.85
    ) -> list[tuple[Chunk, Chunk]]:
        """Pairs with cosine similarity in [low, high] (similar but not
        identical), sorted by descending similarity (most confusing first)."""
        assert self.sim_matrix is not None
        n = len(self.chunks)
        pairs: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(self.sim_matrix[i, j])
                if low <= sim <= high:
                    pairs.append((sim, i, j))
        pairs.sort(key=lambda t: t[0], reverse=True)
        return [(self._chunk(i), self._chunk(j)) for _, i, j in pairs]

    def get_near_miss_candidates(self) -> list[Chunk]:
        """Chunks with high betweenness centrality (retrieval chokepoints),
        sorted by descending betweenness."""
        ranked = sorted(
            self.betweenness.items(), key=lambda kv: kv[1], reverse=True
        )
        return [
            self.chunks[self.id_to_index[node_id]]
            for node_id, score in ranked
            if score > 0.0
        ] or [
            # fall back to all chunks by betweenness order if none are > 0
            self.chunks[self.id_to_index[node_id]]
            for node_id, _ in ranked
        ]

    # -- summary ---------------------------------------------------------

    def summary(self) -> dict:
        """Topology summary for the ``index`` command output."""
        undirected = self.graph.to_undirected()
        components = list(nx.connected_components(undirected))
        isolated = [n for n in self.graph.nodes if undirected.degree(n) == 0]
        return {
            "chunk_count": len(self.chunks),
            "connected_components": len(components),
            "isolated_chunks": len(isolated),
            "isolated_chunk_ids": isolated,
        }
