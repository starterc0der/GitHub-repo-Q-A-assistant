from __future__ import annotations

from src.config import Settings
from src.generate.decomposer import RETRY_REWRITE_SYSTEM_PROMPT, QueryDecomposer
from src.index.schema import CodeChunk
from src.pipeline import Pipeline, _CandidateState, _RetrievalState, _add_usage


def test_add_usage_accumulates_prompt_and_completion_tokens() -> None:
    totals: dict[str, int] = {}
    _add_usage(totals, {"prompt_tokens": 10, "completion_tokens": 4})
    _add_usage(totals, {"prompt_tokens": 6, "completion_tokens": 2})

    assert totals == {"prompt_tokens": 16, "completion_tokens": 6}


def test_add_usage_is_a_noop_for_none() -> None:
    """A provider that reports no usage must not be treated as zero cost merged in —
    just nothing added, so a genuine total stays accurate."""
    totals = {"prompt_tokens": 5, "completion_tokens": 1}
    _add_usage(totals, None)

    assert totals == {"prompt_tokens": 5, "completion_tokens": 1}


class _FakeLLM:
    def __init__(self, reply: str = "unused"):
        self.reply = reply
        self.last_usage: dict | None = None
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 50}
        # A retry-rewrite call must not be picked up by these decompose-focused tests —
        # "" makes QueryDecomposer.rewrite_for_retry return the sub-question unchanged,
        # i.e. no retry.
        return "" if system == RETRY_REWRITE_SYSTEM_PROMPT else self.reply


def _pipeline_stub(bulk_llm: _FakeLLM) -> Pipeline:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = Settings()
    pipeline.bulk_llm = bulk_llm
    pipeline.decomposer = QueryDecomposer(bulk_llm, max_subquestions=3)
    return pipeline


def test_decompose_skip_does_not_pick_up_a_stale_previous_usage() -> None:
    """Regression guard: a simple (non-compound) question skips the LLM call entirely
    via the keyword pre-filter, so last_usage must not still hold whatever an EARLIER,
    unrelated call left behind — that would misattribute someone else's cost to this
    question's decompose stage."""
    bulk_llm = _FakeLLM()
    bulk_llm.last_usage = {"prompt_tokens": 999, "completion_tokens": 999}  # stale leftover
    pipeline = _pipeline_stub(bulk_llm)
    candidate = _CandidateState(query_vector=[0.0], routed=[], scored_candidates=[], reranked_scored=[])
    pipeline._retrieve_candidates = lambda question, space_id: candidate
    pipeline._finish_retrieval = lambda *a, **k: _RetrievalState(
        query_vector=[0.0], routed=[], scored_candidates=[], reranked_scored=[],
        final_chunks=[], final_chunk_traces=[], wide_fallback=False,
        wide_fallback_reason="", too_large_message=None,
    )

    state = pipeline._retrieve("what does the Router class do?", "demo")

    assert bulk_llm.calls == 0  # keyword pre-filter never thought this was compound
    assert state.tokens == {}
    assert "decompose" in state.timings


def test_decompose_that_actually_fires_records_its_own_usage() -> None:
    bulk_llm = _FakeLLM(reply="What does X do?\nWhat does Y do?")
    pipeline = _pipeline_stub(bulk_llm)
    # Non-empty reranked_scored: both sub-questions found evidence, so sufficiency stays
    # "sufficient" and the retry pass never fires — this test is only about decompose's
    # own usage tracking, not retry (see test_pipeline_iterative_retrieval.py).
    chunk = CodeChunk(
        id="c1", space_id="demo", source_id="src1", file_path="a.py",
        language="text", symbol_name=None, start_line=1, end_line=1, code="x",
    )
    candidate = _CandidateState(
        query_vector=[0.0], routed=[], scored_candidates=[], reranked_scored=[(chunk, 0.9)]
    )
    pipeline._retrieve_candidates = lambda question, space_id: candidate
    pipeline._merge_candidates = lambda sub_qs, states: (candidate, {}, {}, {})
    pipeline._finish_retrieval = lambda *a, **k: _RetrievalState(
        query_vector=[0.0], routed=[], scored_candidates=[], reranked_scored=[],
        final_chunks=[], final_chunk_traces=[], wide_fallback=False,
        wide_fallback_reason="", too_large_message=None,
    )

    state = pipeline._retrieve("what does X do, and what does Y do?", "demo")

    assert bulk_llm.calls == 1
    assert state.tokens == {"prompt_tokens": 100, "completion_tokens": 50}
