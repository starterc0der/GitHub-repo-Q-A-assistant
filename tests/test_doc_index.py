from __future__ import annotations

import pytest

from src.index.doc_index import DocIndex
from src.index.schema import FileSummary
from src.index.vector_store import VectorStore


def _summary(space_id: str, file_path: str, summary: str, source_id: str = "src1") -> FileSummary:
    return FileSummary(
        space_id=space_id, source_id=source_id, file_path=file_path,
        language="python", summary=summary,
    )


def test_search_scopes_results_to_space() -> None:
    index = DocIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert(
        [
            _summary("demo", "a.py", "parses input"),
            _summary("other", "c.py", "parses input"),
        ],
        [[1.0, 0.0], [1.0, 0.0]],
    )

    results = index.search([1.0, 0.0], limit=5, space_id="demo")

    assert [s.file_path for s in results] == ["a.py"]


def test_search_scored_returns_similarity_scores_alongside_summaries() -> None:
    index = DocIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert([_summary("demo", "a.py", "parses input")], [[1.0, 0.0]])

    results = index.search_scored([1.0, 0.0], limit=5, space_id="demo")

    assert len(results) == 1
    summary, score = results[0]
    assert summary.file_path == "a.py"
    assert score == pytest.approx(1.0)


def test_all_vectors_returns_every_summary_in_space_with_its_vector() -> None:
    index = DocIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert(
        [
            _summary("demo", "a.py", "parses input"),
            _summary("demo", "b.py", "writes output"),
            _summary("other", "c.py", "parses input"),
        ],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    )

    results = index.all_vectors("demo")

    assert {summary.file_path for summary, _ in results} == {"a.py", "b.py"}
    assert all(vector for _, vector in results)


def test_delete_source_removes_only_that_sources_summaries() -> None:
    index = DocIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert(
        [
            _summary("demo", "a.py", "parses input", source_id="src1"),
            _summary("demo", "b.py", "writes output", source_id="src2"),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    index.delete_source("demo", "src1")

    results = index.all_vectors("demo")
    assert [s.file_path for s, _ in results] == ["b.py"]
