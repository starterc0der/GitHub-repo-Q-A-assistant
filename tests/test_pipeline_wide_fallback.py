from __future__ import annotations

from src.config import Settings
from src.index.chunk_index import ChunkIndex
from src.index.schema import CodeChunk
from src.index.vector_store import VectorStore
from src.pipeline import Pipeline


def _chunk(file_path: str, code: str, source_id: str = "src1") -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}::1-1", space_id="demo", source_id=source_id, file_path=file_path,
        language="text", symbol_name=None, start_line=1, end_line=1, code=code,
    )


def _pipeline_stub(chunk_index: ChunkIndex, wide_answer_max_tokens: int = 150_000) -> Pipeline:
    """_wide_fallback_chunks/_too_large_message only touch settings + chunk_index, so this
    skips Pipeline.__init__ entirely rather than loading real embedding/reranker models."""
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = Settings(wide_answer_max_tokens=wide_answer_max_tokens)
    pipeline.chunk_index = chunk_index
    return pipeline


def test_wide_fallback_chunks_returns_every_chunk_in_the_routed_sources() -> None:
    chunk_index = ChunkIndex(VectorStore(":memory:"))
    chunk_index.ensure(dim=2)
    chunk_index.upsert(
        [_chunk("a.py", "aaa"), _chunk("b.py", "bbb")], [[1.0, 0.0], [0.0, 1.0]]
    )
    pipeline = _pipeline_stub(chunk_index)

    chunks = pipeline._wide_fallback_chunks("demo", ["src1"])

    assert {c.file_path for c in chunks} == {"a.py", "b.py"}


def test_wide_fallback_is_not_capped_by_how_many_pages_routing_picked() -> None:
    """Regression test: routing's top_files cap can pick only 1 of a 3-page document's
    pages, but the wide fallback must still read the *whole* source, not just that page —
    otherwise a summary silently drops pages it never looked at."""
    chunk_index = ChunkIndex(VectorStore(":memory:"))
    chunk_index.ensure(dim=2)
    chunk_index.upsert(
        [
            _chunk("story · p.1", "page one", source_id="story"),
            _chunk("story · p.2", "page two", source_id="story"),
            _chunk("story · p.3", "page three", source_id="story"),
            _chunk("other.py", "unrelated", source_id="other-source"),
        ],
        [[1.0, 0.0]] * 4,
    )
    pipeline = _pipeline_stub(chunk_index)

    # Only "story · p.1" was routed (simulates top_files capping mid-document), but the
    # fallback is keyed on source_id, so it must still pull all 3 pages of "story".
    chunks = pipeline._wide_fallback_chunks("demo", ["story"])

    assert {c.file_path for c in chunks} == {"story · p.1", "story · p.2", "story · p.3"}


def test_wide_fallback_chunks_returns_empty_for_no_routed_sources() -> None:
    pipeline = _pipeline_stub(ChunkIndex(VectorStore(":memory:")))

    assert pipeline._wide_fallback_chunks("demo", []) == []


def test_too_large_message_reports_file_count_and_token_estimate() -> None:
    pipeline = _pipeline_stub(ChunkIndex(VectorStore(":memory:")), wide_answer_max_tokens=1000)

    message = pipeline._too_large_message(3, 5000)

    assert "3 file(s)" in message
    assert "5,000 tokens" in message
    assert "1,000 tokens" in message
