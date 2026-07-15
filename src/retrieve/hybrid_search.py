from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from src.index.chunk_index import ChunkIndex
from src.index.embedder import Embedder
from src.index.schema import CodeChunk


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

    def search(self, question: str, repo: str, file_paths: list[str], k: int) -> list[CodeChunk]:
        candidates = self.chunk_index.fetch_by_files(repo, file_paths)
        if not candidates:
            return []
        chunks, vectors = zip(*candidates)

        dense_scores = self._cosine_scores(self.embedder.embed_one(question), vectors)
        bm25 = BM25Okapi([chunk.code.split() for chunk in chunks])
        bm25_scores = list(bm25.get_scores(question.split()))

        fused = self.fuser.fuse(dense_scores, bm25_scores)
        ranked = sorted(zip(fused, chunks), key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in ranked[:k]]

    def _cosine_scores(
        self, query_vector: list[float], vectors: tuple[list[float], ...]
    ) -> list[float]:
        query = np.array(query_vector)
        matrix = np.array(vectors)
        query_unit = query / (np.linalg.norm(query) + 1e-9)
        matrix_unit = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        return (matrix_unit @ query_unit).tolist()
