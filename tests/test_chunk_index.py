from __future__ import annotations

from src.index.chunk_index import ChunkIndex
from src.index.schema import CodeChunk
from src.index.vector_store import VectorStore


def _chunk(file_path: str) -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}::1-1",
        space_id="demo",
        source_id="src1",
        file_path=file_path,
        language="python",
        symbol_name="run",
        start_line=1,
        end_line=1,
        code="def run(): pass",
    )


def test_search_filters_by_space_and_file_paths() -> None:
    index = ChunkIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert([_chunk("a.py"), _chunk("b.py")], [[1.0, 0.0], [0.0, 1.0]])

    results = index.search([1.0, 0.0], limit=5, space_id="demo", file_paths=["a.py"])

    assert [c.file_path for c in results] == ["a.py"]


def test_fetch_by_files_with_empty_list_returns_whole_space() -> None:
    index = ChunkIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert([_chunk("a.py"), _chunk("b.py")], [[1.0, 0.0], [0.0, 1.0]])

    candidates = index.fetch_by_files(space_id="demo", file_paths=[])

    assert {chunk.file_path for chunk, _ in candidates} == {"a.py", "b.py"}
    assert all(vector for _, vector in candidates)


def test_fetch_by_sources_returns_every_chunk_for_the_given_sources() -> None:
    index = ChunkIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    a, b, c = _chunk("a.py"), _chunk("b.py"), _chunk("c.py")
    b.source_id = "src2"
    c.source_id = "src3"
    index.upsert([a, b, c], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    results = index.fetch_by_sources("demo", ["src1", "src2"])

    assert {chunk.file_path for chunk in results} == {"a.py", "b.py"}


def test_fetch_by_sources_with_empty_list_returns_nothing() -> None:
    index = ChunkIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert([_chunk("a.py")], [[1.0, 0.0]])

    assert index.fetch_by_sources("demo", []) == []


def test_delete_source_removes_only_that_sources_chunks() -> None:
    index = ChunkIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    a, b = _chunk("a.py"), _chunk("b.py")
    b.source_id = "src2"
    index.upsert([a, b], [[1.0, 0.0], [0.0, 1.0]])

    index.delete_source("demo", "src1")

    remaining = index.fetch_by_files(space_id="demo", file_paths=[])
    assert [chunk.file_path for chunk, _ in remaining] == ["b.py"]
