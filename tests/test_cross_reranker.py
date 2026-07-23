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

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
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
