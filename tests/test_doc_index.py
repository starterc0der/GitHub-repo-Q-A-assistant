from __future__ import annotations

import pytest

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


def test_search_scored_returns_similarity_scores_alongside_summaries() -> None:
    index = DocIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert(
        [FileSummary(repo="demo", file_path="a.py", language="python", summary="parses input")],
        [[1.0, 0.0]],
    )

    results = index.search_scored([1.0, 0.0], limit=5, repo="demo")

    assert len(results) == 1
    summary, score = results[0]
    assert summary.file_path == "a.py"
    assert score == pytest.approx(1.0)


def test_all_vectors_returns_every_summary_in_repo_with_its_vector() -> None:
    index = DocIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert(
        [
            FileSummary(repo="demo", file_path="a.py", language="python", summary="parses input"),
            FileSummary(repo="demo", file_path="b.py", language="python", summary="writes output"),
            FileSummary(repo="other", file_path="c.py", language="python", summary="parses input"),
        ],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    )

    results = index.all_vectors("demo")

    assert {summary.file_path for summary, _ in results} == {"a.py", "b.py"}
    assert all(vector for _, vector in results)
