from __future__ import annotations

import time

from src.config import Settings
from src.index.schema import CodeChunk
from src.pipeline import Pipeline, _CandidateState
from src.retrieve.compressor import Compressor


class _FakeCompressLLM:
    """Chunk 1 keeps line 1 only (drops the rest); chunk 2 is judged NONE (dropped
    entirely) — exercises both a real trim and a full drop in one call, matching how
    Compressor.compress_batch actually replies."""

    last_usage: dict | None = None

    def complete(self, prompt: str, system: str | None = None) -> str:
        return "1: 1\n2: NONE"


class _FakeChunkIndex:
    def __init__(self, wide_chunks: list[CodeChunk]):
        self._wide_chunks = wide_chunks

    def fetch_by_sources(self, space_id: str, source_ids: list[str]) -> list[CodeChunk]:
        return self._wide_chunks


class _FakeIntentClassifier:
    """Non-empty reranked_scored reaches this gate tier (see BroadIntentClassifier) —
    always says "not broad" so the compress path (not wide fallback) is what's tested."""

    def is_broad(self, question: str) -> bool:
        return False


def _chunk(file_path: str, code: str) -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}::1-4", space_id="demo", source_id="src1", file_path=file_path,
        language="text", symbol_name=None, start_line=1, end_line=4, code=code,
    )


def _pipeline_stub(wide_chunks: list[CodeChunk] | None = None) -> Pipeline:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = Settings()
    pipeline.compressor = Compressor(_FakeCompressLLM())
    pipeline.chunk_index = _FakeChunkIndex(wide_chunks or [])
    pipeline.intent_classifier = _FakeIntentClassifier()
    pipeline.llm = _FakeCompressLLM()
    pipeline.bulk_llm = _FakeCompressLLM()
    return pipeline


def test_compress_path_records_original_and_compressed_tokens() -> None:
    kept = _chunk("a.py", "aaaa\nbbbb\ncccc\ndddd")  # 20 chars -> 5 tokens original
    dropped = _chunk("b.py", "eeee\nffff\ngggg\nhhhh")
    pipeline = _pipeline_stub()
    candidate = _CandidateState(
        query_vector=[0.0], routed=[("a.py", "src1", 0.9)], scored_candidates=[],
        reranked_scored=[(kept, 0.9), (dropped, 0.8)],
    )

    state = pipeline._finish_retrieval(
        "question", "demo", candidate, [], {}, {}, {}, time.monotonic()
    )

    kept_trace = next(t for t in state.final_chunk_traces if t.chunk.file_path == "a.py")
    dropped_trace = next(t for t in state.final_chunk_traces if t.chunk.file_path == "b.py")

    assert kept_trace.dropped is False
    assert kept_trace.original_tokens == len("aaaa\nbbbb\ncccc\ndddd") // 4
    assert kept_trace.compressed_tokens == len("aaaa") // 4  # only line 1 survived
    assert kept_trace.compressed_tokens < kept_trace.original_tokens

    assert dropped_trace.dropped is True
    assert dropped_trace.original_tokens == len("eeee\nffff\ngggg\nhhhh") // 4
    assert dropped_trace.compressed_tokens == 0  # nothing survives a full drop


def test_wide_fallback_path_reports_equal_original_and_compressed_tokens() -> None:
    """Wide fallback never compresses — original_tokens and compressed_tokens must match
    exactly, not just both be non-zero, since nothing was actually trimmed."""
    whole_file = _chunk("story.pdf", "aaaa\nbbbb\ncccc\ndddd\neeee\nffff")
    pipeline = _pipeline_stub(wide_chunks=[whole_file])
    candidate = _CandidateState(
        query_vector=[0.0], routed=[("story.pdf", "src1", 0.9)], scored_candidates=[],
        reranked_scored=[],  # nothing reranked -> eligible for wide fallback
    )

    state = pipeline._finish_retrieval(
        "summarize the whole story", "demo", candidate, [], {}, {}, {}, time.monotonic()
    )

    assert state.wide_fallback is True
    trace = state.final_chunk_traces[0]
    assert trace.original_tokens == trace.compressed_tokens
    assert trace.original_tokens == len(whole_file.code) // 4
