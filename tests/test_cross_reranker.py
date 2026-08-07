from __future__ import annotations

from src.index.schema import CodeChunk
from src.retrieve.cross_reranker import CrossReranker


def _chunk(file_path: str) -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}::1-1",
        repo="demo",
        file_path=file_path,
        language="python",
        symbol_name=None,
        start_line=1,
        end_line=1,
        code="def run(): pass",
    )


class FakeCrossEncoder:
    def __init__(self, scores: list[float]):
        self.scores = scores

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
        return self.scores


def test_rerank_scored_orders_by_score_and_keeps_top_k() -> None:
    reranker = CrossReranker("unused")
    reranker._model = FakeCrossEncoder([0.1, 0.9])
    chunks = [_chunk("a.py"), _chunk("b.py")]

    results = reranker.rerank_scored("question", chunks, top_k=1)

    assert [c.file_path for c, _ in results] == ["b.py"]
    assert results[0][1] == 0.9


def test_rerank_derives_from_rerank_scored() -> None:
    reranker = CrossReranker("unused")
    reranker._model = FakeCrossEncoder([0.1, 0.9])
    chunks = [_chunk("a.py"), _chunk("b.py")]

    results = reranker.rerank("question", chunks, top_k=2)

    assert [c.file_path for c in results] == ["b.py", "a.py"]


def test_rerank_keeps_low_scoring_chunks_when_the_top_one_is_strong() -> None:
    """The model scores "does this answer the question alone", so supporting context
    legitimately scores low. A strong top score means the question IS answerable, and
    those weak chunks are the context that makes the answer complete."""
    reranker = CrossReranker("unused", min_top_score=0.01)
    reranker._model = FakeCrossEncoder([0.884, 0.046, 0.003])
    chunks = [_chunk("a.py"), _chunk("b.py"), _chunk("c.py")]

    kept = reranker.rerank_scored("how does routing work", chunks, top_k=6)

    assert [c.file_path for c, _ in kept] == ["a.py", "b.py", "c.py"]


def test_rerank_rejects_everything_when_even_the_best_chunk_is_irrelevant() -> None:
    reranker = CrossReranker("unused", min_top_score=0.01)
    reranker._model = FakeCrossEncoder([0.0, 0.0])

    assert reranker.rerank_scored("who is the prime minister", [_chunk("a.py"), _chunk("b.py")], 6) == []


def test_rerank_still_honours_top_k_above_the_gate() -> None:
    reranker = CrossReranker("unused", min_top_score=0.01)
    reranker._model = FakeCrossEncoder([0.9, 0.8, 0.7, 0.6])
    chunks = [_chunk(f"{n}.py") for n in "abcd"]

    assert len(reranker.rerank_scored("q", chunks, top_k=2)) == 2
