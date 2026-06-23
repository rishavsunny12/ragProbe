"""Deterministic synthetic chunk corpus with precomputed embeddings.

Building embeddings in code (instead of running sentence-transformers) keeps
unit tests fast and offline. The geometry is engineered so every failure-mode
query has something to return:

  * 4 tight clusters (6 chunks each) -> buried_fact + intra-cluster structure
  * 3 satellites near cluster 0 at cosine ~0.70/0.78/0.82 -> distractor band
  * 3 bridge chunks chaining the clusters -> high betweenness (near_miss) and
    low-similarity-but-connected pairs (multi_hop)
"""

from __future__ import annotations

import numpy as np

from ragprobe.models import Chunk

D = 12
NOISE = 0.02


def _basis(i: int) -> np.ndarray:
    v = np.zeros(D)
    v[i] = 1.0
    return v


def make_synthetic_chunks(seed: int = 0) -> list[Chunk]:
    rng = np.random.default_rng(seed)
    chunks: list[Chunk] = []

    # 4 tight clusters along orthogonal basis directions e0..e3
    for c in range(4):
        center = _basis(c)
        for i in range(6):
            v = center + rng.normal(0, NOISE, D)
            chunks.append(
                Chunk(
                    id=f"c{c}_{i}",
                    text=f"Cluster {c} document {i} discussing topic {c}.",
                    metadata={"cluster": c},
                    embedding=[float(x) for x in v],
                )
            )

    # satellites near cluster 0 placed at target cosines (distractor band)
    for name, b, dim in [("sat0", 1.020, 4), ("sat1", 0.802, 5), ("sat2", 0.698, 6)]:
        v = _basis(0) + b * _basis(dim) + rng.normal(0, NOISE, D)
        chunks.append(
            Chunk(
                id=name,
                text=f"Satellite {name} closely related to topic 0.",
                metadata={"role": "satellite"},
                embedding=[float(x) for x in v],
            )
        )

    # bridges chaining clusters 0-1, 1-2, 2-3 (retrieval chokepoints)
    for name, a, bdim in [("br01", 0, 1), ("br12", 1, 2), ("br23", 2, 3)]:
        v = _basis(a) + _basis(bdim) + rng.normal(0, NOISE, D)
        chunks.append(
            Chunk(
                id=name,
                text=f"Bridge {name} connecting two topic areas.",
                metadata={"role": "bridge"},
                embedding=[float(x) for x in v],
            )
        )

    return chunks


BRIDGE_IDS = {"br01", "br12", "br23"}
