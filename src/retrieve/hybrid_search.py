from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from src.index.chunk_index import ChunkIndex
from src.index.embedder import Embedder
from src.index.schema import CodeChunk


@dataclass
class ScoredChunk:
    """A candidate chunk with its per-signal scores, for inspecting the fusion step.

    vector: the chunk's own dense embedding, already fetched from Qdrant for dense
    scoring above — carried through (not recomputed) so a later MMR diversity pass can
    compare candidates to each other without a second embedding call."""

    chunk: CodeChunk
    dense_score: float
    bm25_score: float
    fused_score: float
    vector: list[float]


class RankFuser:
    """Combines two independently-scaled score lists via weighted min-max fusion."""

    def __init__(self, dense_weight: float, bm25_weight: float):
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def fuse(self, dense_scores: list[float], bm25_scores: list[float]) -> list[float]:
        dense_norm = self._normalize(dense_scores)
        bm25_norm = self._normalize(bm25_scores)
        return [self.dense_weight * d + self.bm25_weight * b for d, b in zip(dense_norm, bm25_norm)]

    def _normalize(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return [0.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]


class HybridSearch:
    """Fuses dense-vector and BM25 retrieval over a router-scoped candidate pool."""

    def __init__(self, embedder: Embedder, chunk_index: ChunkIndex, fuser: RankFuser):
        self.embedder = embedder
        self.chunk_index = chunk_index
        self.fuser = fuser

    def search(
        self, question: str, space_id: str, file_paths: list[str], k: int
    ) -> list[CodeChunk]:
        return [sc.chunk for sc in self.search_scored(question, space_id, file_paths, k)]

    def search_scored(
        self, question: str, space_id: str, file_paths: list[str], k: int,
        query_vector: list[float] | None = None,
    ) -> list[ScoredChunk]:
        """Pass query_vector when the caller already embedded the question, to avoid a
        redundant embedding forward pass. question is still needed either way, for BM25."""
        candidates = self.chunk_index.fetch_by_files(space_id, file_paths)
        if not candidates:
            return []
        chunks, vectors = zip(*candidates)

        vector = query_vector if query_vector is not None else self.embedder.embed_one(question)
        dense_scores = self._cosine_scores(vector, vectors)
        bm25 = BM25Okapi([chunk.code.split() for chunk in chunks])
        bm25_scores = list(bm25.get_scores(question.split()))

        fused = self.fuser.fuse(dense_scores, bm25_scores)
        ranked = sorted(
            zip(fused, dense_scores, bm25_scores, chunks, vectors), key=lambda t: t[0], reverse=True
        )
        return [
            ScoredChunk(chunk=chunk, dense_score=dense, bm25_score=bm25_, fused_score=fused_, vector=vector)
            for fused_, dense, bm25_, chunk, vector in ranked[:k]
        ]

    def _cosine_scores(
        self, query_vector: list[float], vectors: tuple[list[float], ...]
    ) -> list[float]:
        query = np.array(query_vector)
        matrix = np.array(vectors)
        query_unit = query / (np.linalg.norm(query) + 1e-9)
        matrix_unit = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        return (matrix_unit @ query_unit).tolist()
