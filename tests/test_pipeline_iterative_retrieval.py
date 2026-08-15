from __future__ import annotations

from src.config import Settings
from src.generate.decomposer import DECOMPOSE_SYSTEM_PROMPT, RETRY_REWRITE_SYSTEM_PROMPT, QueryDecomposer
from src.index.schema import CodeChunk
from src.pipeline import Pipeline, _CandidateState, _RetrievalState


class _FakeLLM:
    """Routes by system prompt: the decompose call gets `decompose_reply`; a retry-rewrite
    call gets whatever `retry_replies` has queued for it (or "" — no rewrite — once
    exhausted)."""

    def __init__(self, decompose_reply: str, retry_replies: list[str] | None = None):
        self.decompose_reply = decompose_reply
        self.retry_replies = list(retry_replies or [])
        self.last_usage: dict | None = None
        self.retry_calls = 0

    def complete(self, prompt: str, system: str | None = None) -> str:
        if system == RETRY_REWRITE_SYSTEM_PROMPT:
            self.retry_calls += 1
            return self.retry_replies.pop(0) if self.retry_replies else ""
        assert system == DECOMPOSE_SYSTEM_PROMPT
        return self.decompose_reply


def _chunk(chunk_id: str) -> CodeChunk:
    return CodeChunk(
        id=chunk_id, space_id="demo", source_id="src1", file_path="a.py",
        language="text", symbol_name=None, start_line=1, end_line=1, code="x",
    )


def _pipeline_stub(llm: _FakeLLM) -> Pipeline:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = Settings()
    pipeline.bulk_llm = llm
    pipeline.decomposer = QueryDecomposer(llm, max_subquestions=3)
    return pipeline


def _stub_finish(pipeline: Pipeline) -> None:
    pipeline._finish_retrieval = lambda *a, **k: _RetrievalState(
        query_vector=[0.0], routed=[], scored_candidates=[], reranked_scored=[],
        final_chunks=[], final_chunk_traces=[], wide_fallback=False,
        wide_fallback_reason="", too_large_message=None,
    )
    pipeline._merge_candidates = lambda sub_qs, states: (states[0], {}, {}, {})


def test_retry_that_finds_evidence_upgrades_partial_to_sufficient() -> None:
    """The real Drona/Kripa case: "how is Drona different from Kripa?" finds nothing, but
    rewriting it to directly ask about Kripa (who IS in the corpus, per the "who is
    Kripa?" case verified live) finds evidence on retry."""
    llm = _FakeLLM(
        "NONE\nPARALLEL\nWho is Drona?\nHow is Drona different from Kripa?",
        retry_replies=["Who is Kripa?"],
    )
    pipeline = _pipeline_stub(llm)
    states_by_q = {
        "Who is Drona?": _CandidateState([0.0], [], [], [(_chunk("d1"), 0.9)]),
        "How is Drona different from Kripa?": _CandidateState([0.0], [], [], []),
        "Who is Kripa?": _CandidateState([0.0], [], [], [(_chunk("k1"), 0.85)]),
    }
    pipeline._retrieve_candidates = lambda q, space_id, **_kw: states_by_q[q]
    _stub_finish(pipeline)

    state = pipeline._retrieve("who is Drona, and how is he different from Kripa?", "demo")

    assert state.sufficiency == "sufficient"
    assert state.insufficient_sub_questions == []
    assert state.retried_sub_questions == {"How is Drona different from Kripa?": "Who is Kripa?"}
    assert "retry" in state.timings


def test_retry_that_still_finds_nothing_stays_partial() -> None:
    """The information genuinely isn't in the corpus — retry gets one honest attempt,
    then the gap is still reported, not silently retried forever."""
    llm = _FakeLLM(
        "NONE\nPARALLEL\nWho is Drona?\nWhat is Drona's favorite color?",
        retry_replies=["What color does Drona prefer?"],
    )
    pipeline = _pipeline_stub(llm)
    states_by_q = {
        "Who is Drona?": _CandidateState([0.0], [], [], [(_chunk("d1"), 0.9)]),
        "What is Drona's favorite color?": _CandidateState([0.0], [], [], []),
        "What color does Drona prefer?": _CandidateState([0.0], [], [], []),
    }
    pipeline._retrieve_candidates = lambda q, space_id, **_kw: states_by_q[q]
    _stub_finish(pipeline)

    state = pipeline._retrieve("who is Drona, and what is his favorite color?", "demo")

    assert state.sufficiency == "partial"
    assert state.insufficient_sub_questions == ["What is Drona's favorite color?"]
    assert state.retried_sub_questions == {
        "What is Drona's favorite color?": "What color does Drona prefer?"
    }
    assert llm.retry_calls == 1  # exactly one retry attempt, never looped


def test_retry_rewrite_returning_the_same_text_is_not_retried_again() -> None:
    """A rewrite failure (LLMClient falls back to the original text) must not be treated
    as a distinct retry attempt — no new retrieval call, no phantom timing key."""
    llm = _FakeLLM("NONE\nPARALLEL\nWho is Drona?\nHow is Drona different from Kripa?", retry_replies=[])
    pipeline = _pipeline_stub(llm)
    calls: list[str] = []

    def fake_retrieve_candidates(q: str, space_id: str, **_kw) -> _CandidateState:
        calls.append(q)
        if q == "Who is Drona?":
            return _CandidateState([0.0], [], [], [(_chunk("d1"), 0.9)])
        return _CandidateState([0.0], [], [], [])

    pipeline._retrieve_candidates = fake_retrieve_candidates
    _stub_finish(pipeline)

    state = pipeline._retrieve("who is Drona, and how is he different from Kripa?", "demo")

    assert calls == ["Who is Drona?", "How is Drona different from Kripa?"]  # no retry call
    assert state.sufficiency == "partial"
    assert state.retried_sub_questions == {}
    assert "retry" not in state.timings


def test_single_question_never_retries() -> None:
    """No decomposition, no sub-question-level insufficiency to retry — matches the
    existing sufficiency-detection scope."""
    llm = _FakeLLM("SINGLE")
    pipeline = _pipeline_stub(llm)
    pipeline._retrieve_candidates = lambda q, space_id, **_kw: _CandidateState([0.0], [], [], [])
    _stub_finish(pipeline)

    state = pipeline._retrieve("what does the Router class do?", "demo")

    assert state.retried_sub_questions == {}
    assert llm.retry_calls == 0
