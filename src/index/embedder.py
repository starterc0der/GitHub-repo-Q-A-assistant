from __future__ import annotations

from sentence_transformers import SentenceTransformer


class Embedder:
    """Wraps a local sentence-transformers model, lazy-loaded on first use."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
