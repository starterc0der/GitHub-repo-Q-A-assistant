from __future__ import annotations

from src.index.doc_index import DocIndex
from src.index.schema import FileSummary
from src.index.vector_store import VectorStore


def test_search_scopes_results_to_repo() -> None:
    index = DocIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert(
        [
            FileSummary(repo="demo", file_path="a.py", language="python", summary="parses input"),
            FileSummary(repo="other", file_path="c.py", language="python", summary="parses input"),
        ],
        [[1.0, 0.0], [1.0, 0.0]],
    )

    results = index.search([1.0, 0.0], limit=5, repo="demo")

    assert [s.file_path for s in results] == ["a.py"]
