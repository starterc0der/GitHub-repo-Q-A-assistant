from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

from src.index.schema import CodeChunk


class CrossReranker:
    """Cross-encoder reranking — scores each (question, chunk) pair jointly for precision.

    Lazy-loads the model on first use, mirroring Embedder's pattern.
    """

    def __init__(self, model_name: str, min_top_score: float = 0.0, mmr_lambda: float = 0.7):
        self.model_name = model_name
        # Gates the whole list on the best score, not each chunk: supporting context
        # legitimately scores low, so a per-chunk floor threw away useful chunks.
        self.min_top_score = min_top_score
        # MMR relevance/diversity balance (see _mmr_select): 1.0 is pure relevance
        # (identical to plain top-k), 0.0 is pure diversity. 0.7 favors relevance but
        # still lets a near-duplicate lose to a genuinely different, slightly
        # lower-scoring chunk.
        self.mmr_lambda = mmr_lambda
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, question: str, chunks: list[CodeChunk], top_k: int) -> list[CodeChunk]:
        return [chunk for chunk, _ in self.rerank_scored(question, chunks, top_k)]

    def rerank_scored(
        self, question: str, chunks: list[CodeChunk], top_k: int,
        vectors: list[list[float]] | None = None,
    ) -> list[tuple[CodeChunk, float]]:
        """vectors, when given (one per chunk, same order), enables MMR: top_k is chosen
        to balance relevance against diversity from what's already picked, instead of
        being the top_k highest-scoring chunks outright — which can all be overlapping
        windows of the same paragraph. Without vectors, behaves exactly as before."""
        results, _diversity_picks = self.rerank_scored_with_diversity(question, chunks, top_k, vectors)
        return results

    def rerank_scored_with_diversity(
        self, question: str, chunks: list[CodeChunk], top_k: int,
        vectors: list[list[float]] | None = None,
    ) -> tuple[list[tuple[CodeChunk, float]], set[str]]:
        """Same as rerank_scored, but also reports which returned chunk IDs MMR picked
        FOR diversity — i.e. chunks that would NOT be in a plain top-k of these same
        scores, computed once and reused for both selections rather than re-scoring.
        Empty set when vectors weren't given (nothing to compare against) or MMR's
        selection happened to match plain top-k anyway."""
        if not chunks:
            return [], set()
        pairs = [(question, chunk.embeddable_text) for chunk in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)
        if vectors is not None:
            ranked_mmr = sorted(
                zip(scores, chunks, vectors), key=lambda t: t[0], reverse=True
            )
            if float(ranked_mmr[0][0]) < self.min_top_score:
                return [], set()
            selected = self._mmr_select(ranked_mmr, top_k)
            plain_top_k_ids = {chunk.id for _score, chunk, _vector in ranked_mmr[:top_k]}
            diversity_picks = {chunk.id for chunk, _score in selected if chunk.id not in plain_top_k_ids}
            return selected, diversity_picks
        ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
        if float(ranked[0][0]) < self.min_top_score:
            return [], set()  # nothing here answers the question — reject the whole list
        return [(chunk, float(score)) for score, chunk in ranked[:top_k]], set()

    def _mmr_select(
        self, ranked: list[tuple[float, CodeChunk, list[float]]], top_k: int
    ) -> list[tuple[CodeChunk, float]]:
        """Greedy Maximal Marginal Relevance: picks the highest-scoring chunk first, then
        each next pick maximizes (mmr_lambda * relevance - (1 - mmr_lambda) * similarity
        to the closest chunk already picked) — so a second near-duplicate of an
        already-strong pick loses to a lower-scoring but genuinely different chunk."""
        pool = list(enumerate(ranked))
        selected: list[tuple[float, CodeChunk, list[float]]] = []
        while pool and len(selected) < top_k:
            if not selected:
                idx, best = pool[0]
            else:
                idx, best = max(
                    pool,
                    key=lambda p: self.mmr_lambda * p[1][0]
                    - (1 - self.mmr_lambda) * max(self._cosine(p[1][2], s[2]) for s in selected),
                )
            selected.append(best)
            pool = [p for p in pool if p[0] != idx]
        return [(chunk, float(score)) for score, chunk, _vector in selected]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9
        return float(np.dot(va, vb) / denom)
