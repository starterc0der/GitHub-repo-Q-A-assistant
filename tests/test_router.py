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


def _summary(space_id: str, file_path: str, summary: str) -> FileSummary:
    return FileSummary(
        space_id=space_id, source_id="src1", file_path=file_path,
        language="python", summary=summary,
    )


def test_route_to_files_returns_nearest_summary_scoped_to_space() -> None:
    doc_index = DocIndex(VectorStore(":memory:"))
    doc_index.ensure(dim=2)
    doc_index.upsert(
        [
            _summary("demo", "a.py", "parses input"),
            _summary("demo", "b.py", "writes output"),
            _summary("other", "c.py", "parses input"),
        ],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    )
    router = Router(FakeEmbedder({"how do we parse input?": [1.0, 0.0]}), doc_index)

    files = router.route_to_files("how do we parse input?", space_id="demo", top_files=1)

    assert files == ["a.py"]


def test_route_to_files_scored_returns_file_path_and_score_pairs() -> None:
    doc_index = DocIndex(VectorStore(":memory:"))
    doc_index.ensure(dim=2)
    doc_index.upsert([_summary("demo", "a.py", "parses input")], [[1.0, 0.0]])
    router = Router(FakeEmbedder({"how do we parse input?": [1.0, 0.0]}), doc_index)

    results = router.route_to_files_scored("how do we parse input?", space_id="demo", top_files=1)

    assert results[0][0] == "a.py"
    assert results[0][1] > 0
