from __future__ import annotations

from src.index.doc_index import DocIndex
from src.index.schema import FileSummary
from src.index.vector_store import VectorStore
from src.retrieve.router import Router


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed_one(self, text: str) -> list[float]:
        return self.vectors[text]


def test_route_to_files_returns_nearest_summary_scoped_to_repo() -> None:
    doc_index = DocIndex(VectorStore(":memory:"))
    doc_index.ensure(dim=2)
    doc_index.upsert(
        [
            FileSummary(repo="demo", file_path="a.py", language="python", summary="parses input"),
            FileSummary(repo="demo", file_path="b.py", language="python", summary="writes output"),
            FileSummary(repo="other", file_path="c.py", language="python", summary="parses input"),
        ],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    )
    router = Router(FakeEmbedder({"how do we parse input?": [1.0, 0.0]}), doc_index)

    files = router.route_to_files("how do we parse input?", repo="demo", top_files=1)

    assert files == ["a.py"]
