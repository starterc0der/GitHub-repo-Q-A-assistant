from __future__ import annotations

from src.config import Settings
from src.generate.decomposer import RETRY_REWRITE_SYSTEM_PROMPT, QueryDecomposer
from src.index.schema import CodeChunk
from src.pipeline import Pipeline, _CandidateState, _RetrievalState


class _FakeLLM:
    """Answers the decompose call with `reply`; answers a retry-rewrite call with "" (so
    QueryDecomposer.rewrite_for_retry falls back to the sub-question unchanged, meaning
    "nothing to retry") — these tests are about sufficiency detection itself, not the
    retry pass, which has its own test file."""

    def __init__(self, reply: str):
        self.reply = reply
        self.last_usage: dict | None = None
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        if system == RETRY_REWRITE_SYSTEM_PROMPT:
            return ""
        return self.reply


def _chunk(chunk_id: str) -> CodeChunk:
    return CodeChunk(
        id=chunk_id, space_id="demo", source_id="src1", file_path="a.py",
        language="text", symbol_name=None, start_line=1, end_line=1, code="x",
    )


def _pipeline_stub(decompose_reply: str) -> Pipeline:
    """_retrieve only needs decomposer + the three sub-methods it calls, so this skips
    Pipeline.__init__ entirely rather than loading real embedding/reranker models."""
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = Settings()
    bulk_llm = _FakeLLM(decompose_reply)
    pipeline.bulk_llm = bulk_llm
    pipeline.decomposer = QueryDecomposer(bulk_llm, max_subquestions=3)
    return pipeline


def _stub_finish(pipeline: Pipeline) -> None:
    pipeline._finish_retrieval = lambda *a, **k: _RetrievalState(
        query_vector=[0.0], routed=[], scored_candidates=[], reranked_scored=[],
        final_chunks=[], final_chunk_traces=[], wide_fallback=False,
        wide_fallback_reason="", too_large_message=None,
    )
    pipeline._merge_candidates = lambda sub_qs, states: (states[0], {}, {}, {})


def test_single_question_is_always_sufficient() -> None:
    """No decomposition happened, so there's no "some parts missing" to report — the
    existing NO_MATCH/wide_fallback gate already fully covers this question's outcome."""
    pipeline = _pipeline_stub("NONE\nSINGLE")
    pipeline._retrieve_candidates = lambda q, space_id, **_kw: _CandidateState(
        query_vector=[0.0], routed=[], scored_candidates=[], reranked_scored=[],
    )
    _stub_finish(pipeline)

    state = pipeline._retrieve("what does the Router class do?", "demo")

    assert state.sufficiency == "sufficient"
    assert state.insufficient_sub_questions == []


def test_decomposed_question_with_both_sub_questions_covered_is_sufficient() -> None:
    pipeline = _pipeline_stub("NONE\nPARALLEL\nWhat does X do?\nWhat does Y do?")
    states_by_q = {
        "What does X do?": _CandidateState([0.0], [], [], [(_chunk("x1"), 0.9)]),
        "What does Y do?": _CandidateState([0.0], [], [], [(_chunk("y1"), 0.8)]),
    }
    pipeline._retrieve_candidates = lambda q, space_id, **_kw: states_by_q[q]
    _stub_finish(pipeline)

    state = pipeline._retrieve("what does X do and what does Y do?", "demo")

    assert state.sufficiency == "sufficient"
    assert state.insufficient_sub_questions == []


def test_decomposed_question_with_one_sub_question_uncovered_is_partial() -> None:
    """The exact real-world case this feature exists for: "who is Drona, and how is he
    different from Kripa?" — Drona has evidence, the comparison doesn't."""
    pipeline = _pipeline_stub("NONE\nPARALLEL\nWho is Drona?\nHow is Drona different from Kripa?")
    states_by_q = {
        "Who is Drona?": _CandidateState([0.0], [], [], [(_chunk("d1"), 0.9)]),
        "How is Drona different from Kripa?": _CandidateState([0.0], [], [], []),  # rerank gate: nothing cleared it
    }
    pipeline._retrieve_candidates = lambda q, space_id, **_kw: states_by_q[q]
    _stub_finish(pipeline)

    state = pipeline._retrieve("who is Drona, and how is he different from Kripa?", "demo")

    assert state.sufficiency == "partial"
    assert state.insufficient_sub_questions == ["How is Drona different from Kripa?"]


def test_decomposed_question_with_no_sub_questions_covered_is_insufficient() -> None:
    pipeline = _pipeline_stub("NONE\nPARALLEL\nWhat is X?\nWhat is Y?")
    states_by_q = {
        "What is X?": _CandidateState([0.0], [], [], []),
        "What is Y?": _CandidateState([0.0], [], [], []),
    }
    pipeline._retrieve_candidates = lambda q, space_id, **_kw: states_by_q[q]
    _stub_finish(pipeline)

    state = pipeline._retrieve("what is X and what is Y?", "demo")

    assert state.sufficiency == "insufficient"
    assert set(state.insufficient_sub_questions) == {"What is X?", "What is Y?"}
