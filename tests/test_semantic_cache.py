from __future__ import annotations

from src.config import Settings
from src.index.qa_cache_index import QaCacheIndex
from src.index.vector_store import VectorStore
from src.pipeline import Pipeline


class FakeEmbedder:
    """Deterministic stand-in: each call returns whatever vector was registered for that
    exact text, so the similarity between two questions can be controlled directly
    instead of depending on what a real embedding model happens to produce."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.dim = 2

    def embed_one(self, text: str) -> list[float]:
        return self.vectors[text]


def _pipeline_stub(vectors: dict[str, list[float]], min_score: float = 0.92) -> Pipeline:
    """semantic_cache_lookup/_put only touch settings, embedder, and qa_cache_index, so
    this skips Pipeline.__init__ entirely rather than loading real embedding models."""
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = Settings(semantic_cache_min_score=min_score)
    pipeline.embedder = FakeEmbedder(vectors)
    pipeline.qa_cache_index = QaCacheIndex(VectorStore(":memory:"))
    return pipeline


def test_semantic_cache_lookup_hits_on_a_close_paraphrase() -> None:
    vectors = {
        "who is Bhishma": [1.0, 0.0],
        "tell me about Bhishma": [0.99, 0.01],  # near-identical direction -> high cosine
    }
    pipeline = _pipeline_stub(vectors)
    pipeline.semantic_cache_put("demo", "hash-a", "who is Bhishma", "msg-a")

    result = pipeline.semantic_cache_lookup("demo", "tell me about Bhishma")

    assert result is not None
    message_id, matched_question, score = result
    assert message_id == "msg-a"
    assert matched_question == "who is Bhishma"
    assert score >= pipeline.settings.semantic_cache_min_score


def test_semantic_cache_lookup_misses_on_a_related_but_different_question() -> None:
    """Regression guard for the exact risk semantic cache introduces: a topically close
    but substantively different question ("...'s father") must NOT be served the
    original answer just because it shares most of its words."""
    vectors = {
        "who is Bhishma": [1.0, 0.0],
        "who is Bhishma's father": [0.6, 0.8],  # related, but a clearly different vector
    }
    pipeline = _pipeline_stub(vectors)
    pipeline.semantic_cache_put("demo", "hash-a", "who is Bhishma", "msg-a")

    assert pipeline.semantic_cache_lookup("demo", "who is Bhishma's father") is None


def test_semantic_cache_lookup_returns_none_when_nothing_cached_yet() -> None:
    pipeline = _pipeline_stub({"who is Bhishma": [1.0, 0.0]})

    assert pipeline.semantic_cache_lookup("demo", "who is Bhishma") is None


def test_invalidate_qa_cache_clears_only_that_space() -> None:
    vectors = {"who is Bhishma": [1.0, 0.0]}
    pipeline = _pipeline_stub(vectors)
    pipeline.semantic_cache_put("space-a", "hash-a", "who is Bhishma", "msg-a")

    pipeline.invalidate_qa_cache("space-a")

    assert pipeline.semantic_cache_lookup("space-a", "who is Bhishma") is None
