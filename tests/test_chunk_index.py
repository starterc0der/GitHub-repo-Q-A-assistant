from __future__ import annotations

from src.index.chunk_index import ChunkIndex
from src.index.schema import CodeChunk
from src.index.vector_store import VectorStore


def _chunk(file_path: str) -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}::1-1",
        repo="demo",
        file_path=file_path,
        language="python",
        symbol_name="run",
        start_line=1,
        end_line=1,
        code="def run(): pass",
    )


def test_search_filters_by_repo_and_file_paths() -> None:
    index = ChunkIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert([_chunk("a.py"), _chunk("b.py")], [[1.0, 0.0], [0.0, 1.0]])

    results = index.search([1.0, 0.0], limit=5, repo="demo", file_paths=["a.py"])

    assert [c.file_path for c in results] == ["a.py"]


def test_fetch_by_files_with_empty_list_returns_whole_repo() -> None:
    index = ChunkIndex(VectorStore(":memory:"))
    index.ensure(dim=2)
    index.upsert([_chunk("a.py"), _chunk("b.py")], [[1.0, 0.0], [0.0, 1.0]])

    candidates = index.fetch_by_files(repo="demo", file_paths=[])

    assert {chunk.file_path for chunk, _ in candidates} == {"a.py", "b.py"}
    assert all(vector for _, vector in candidates)
