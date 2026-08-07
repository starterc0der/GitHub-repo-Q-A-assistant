from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.index.schema import CodeChunk


class CrossReranker:
    """Cross-encoder reranking — scores each (question, chunk) pair jointly for precision.

    Lazy-loads the model on first use, mirroring Embedder's pattern.
    """

    def __init__(self, model_name: str, min_top_score: float = 0.0):
        self.model_name = model_name
        # Gates the whole list on the best score, not each chunk: supporting context
        # legitimately scores low, so a per-chunk floor threw away useful chunks.
        self.min_top_score = min_top_score
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, question: str, chunks: list[CodeChunk], top_k: int) -> list[CodeChunk]:
        return [chunk for chunk, _ in self.rerank_scored(question, chunks, top_k)]

    def rerank_scored(
        self, question: str, chunks: list[CodeChunk], top_k: int
    ) -> list[tuple[CodeChunk, float]]:
        if not chunks:
            return []
        pairs = [(question, chunk.embeddable_text) for chunk in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
        if float(ranked[0][0]) < self.min_top_score:
            return []  # nothing here answers the question — reject the whole list
        return [(chunk, float(score)) for score, chunk in ranked[:top_k]]
