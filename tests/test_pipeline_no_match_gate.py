from __future__ import annotations

import time

from src.config import Settings
from src.pipeline import Pipeline, _CandidateState
from src.retrieve.compressor import Compressor


class _FakeChunkIndex:
    def fetch_by_sources(self, space_id: str, source_ids: list[str]) -> list:
        return []


class _FakeLLM:
    """Neither test below has non-empty reranked_chunks, so Compressor.compress_batch's
    own early-return means .complete() is never actually called — only last_usage's
    reset-before-call needs a real attribute to write to."""

    last_usage: dict | None = None


def _pipeline_stub(route_min_top_score: float = 0.48) -> Pipeline:
    """_finish_retrieval only touches settings/compressor/chunk_index for the branch
    under test, so this skips Pipeline.__init__ entirely rather than loading real models."""
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = Settings(route_min_top_score=route_min_top_score)
    pipeline.compressor = Compressor(None)
    pipeline.chunk_index = _FakeChunkIndex()
    pipeline.llm = _FakeLLM()
    pipeline.bulk_llm = _FakeLLM()
    return pipeline


def _candidate(route_top_score: float) -> _CandidateState:
    return _CandidateState(
        query_vector=[0.0],
        routed=[("f.py", "src1", route_top_score)],
        scored_candidates=[],
        reranked_scored=[],  # reranking found nothing, in both scenarios below
    )


def test_off_topic_question_gates_to_no_match_instead_of_wide_fallback() -> None:
    """Zero reranked chunks AND a weak route score means nothing here relates to the
    question — must NOT attempt the wide (whole-space) path, which would just end in a
    confusing 'too large, be more specific' refusal for a question that isn't too broad,
    it's just not covered."""
    pipeline = _pipeline_stub(route_min_top_score=0.48)
    candidate = _candidate(route_top_score=0.40)

    state = pipeline._finish_retrieval(
        "what is the boiling point of water?", "demo", candidate, [], {}, {}, {}, time.monotonic()
    )

    assert state.wide_fallback is False
    assert state.final_chunks == []


def test_genuinely_broad_question_still_goes_wide() -> None:
    """Zero reranked chunks with a STRONG route score means the topic is covered, just
    not by any single chunk — the wide path must still fire here."""
    pipeline = _pipeline_stub(route_min_top_score=0.48)
    candidate = _candidate(route_top_score=0.60)

    state = pipeline._finish_retrieval(
        "summarize the whole story", "demo", candidate, [], {}, {}, {}, time.monotonic()
    )

    assert state.wide_fallback is True
    assert state.wide_fallback_reason == "no chunk answered this on its own"
