from __future__ import annotations

from src.index.chunk_index import ChunkIndex
from src.index.doc_index import DocIndex
from src.index.schema import CodeChunk, FileSummary
from src.index.vector_store import VectorStore
from src.pipeline import Pipeline


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed_one(self, text: str) -> list[float]:
        return self.vectors[text]


def _summary(file_path: str) -> FileSummary:
    return FileSummary(
        space_id="demo", source_id="src1", file_path=file_path,
        language="python", summary=f"summary of {file_path}",
    )


def _chunk(file_path: str, symbol_name: str | None = None) -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}::1-1", space_id="demo", source_id="src1", file_path=file_path,
        language="python", symbol_name=symbol_name, start_line=1, end_line=1, code="pass",
    )


def _pipeline_stub(doc_index: DocIndex, chunk_index: ChunkIndex, embedder: FakeEmbedder) -> Pipeline:
    """vectors_trace only touches embedder/doc_index/chunk_index — skip Pipeline.__init__
    entirely rather than loading real embedding/reranker models, same as
    test_pipeline_wide_fallback.py's _pipeline_stub."""
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.embedder = embedder
    pipeline.doc_index = doc_index
    pipeline.chunk_index = chunk_index
    return pipeline


def test_vectors_trace_projects_files_and_chunks_with_labels() -> None:
    doc_index = DocIndex(VectorStore(":memory:"))
    doc_index.ensure(dim=2)
    doc_index.upsert([_summary("a.py"), _summary("b.py")], [[1.0, 0.0], [0.0, 1.0]])

    chunk_index = ChunkIndex(VectorStore(":memory:"))
    chunk_index.ensure(dim=2)
    chunk_index.upsert(
        [_chunk("a.py", "foo"), _chunk("b.py")], [[1.0, 0.0], [0.0, 1.0]]
    )

    embedder = FakeEmbedder({"how do we parse input?": [1.0, 0.0]})
    pipeline = _pipeline_stub(doc_index, chunk_index, embedder)

    result = pipeline.vectors_trace("how do we parse input?", space_id="demo")

    assert set(result.file_xyz.keys()) == {"a.py", "b.py"}
    assert set(result.whole_chunk_xyz.keys()) == {"a.py::1-1", "b.py::1-1"}
    assert result.chunk_labels["a.py::1-1"] == "a.py · foo"
    assert result.chunk_labels["b.py::1-1"] == "b.py · block"


def test_vectors_trace_reembeds_the_given_question() -> None:
    """Re-embedding must be deterministic for the given question — not the caller's job to
    pass a pre-computed vector, since the whole point is this runs after the fact."""
    doc_index = DocIndex(VectorStore(":memory:"))
    doc_index.ensure(dim=2)
    doc_index.upsert([_summary("a.py")], [[1.0, 0.0]])

    chunk_index = ChunkIndex(VectorStore(":memory:"))
    chunk_index.ensure(dim=2)

    embedder = FakeEmbedder({"a specific question": [0.0, 1.0]})
    pipeline = _pipeline_stub(doc_index, chunk_index, embedder)

    result = pipeline.vectors_trace("a specific question", space_id="demo")

    assert "a.py" in result.file_xyz
