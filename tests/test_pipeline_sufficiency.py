from __future__ import annotations

from src.config import Settings
from src.generate.decomposer import RETRY_REWRITE_SYSTEM_PROMPT, DecomposeResult, QueryDecomposer
from src.generate.sufficiency import SYSTEM_PROMPT as SUFFICIENCY_SYSTEM_PROMPT, SufficiencyChecker
from src.index.schema import CodeChunk
from src.pipeline import Pipeline, _CandidateState, _RetrievalState


class _FakeLLM:
    """Answers the decompose call with `reply`; answers a retry-rewrite call with "" (so
    QueryDecomposer.rewrite_for_retry falls back to the sub-question unchanged, meaning
    "nothing to retry"); answers a sufficiency-check call with `sufficiency_reply` (default
    "SUFFICIENT" — most of these tests are about the zero-chunk case, not the ambiguous
    middle band, which has its own tests below)."""

    def __init__(self, reply: str, sufficiency_reply: str = "SUFFICIENT"):
        self.reply = reply
        self.sufficiency_reply = sufficiency_reply
        self.last_usage: dict | None = None
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        if system == RETRY_REWRITE_SYSTEM_PROMPT:
            return ""
        if system == SUFFICIENCY_SYSTEM_PROMPT:
            return self.sufficiency_reply
        return self.reply


def _chunk(chunk_id: str) -> CodeChunk:
    return CodeChunk(
        id=chunk_id, space_id="demo", source_id="src1", file_path="a.py",
        language="text", symbol_name=None, start_line=1, end_line=1, code="x",
    )


def _pipeline_stub(decompose_reply: str, sufficiency_reply: str = "SUFFICIENT") -> Pipeline:
    """_retrieve only needs decomposer + the sub-methods it calls, so this skips
    Pipeline.__init__ entirely rather than loading real embedding/reranker models."""
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = Settings()
    bulk_llm = _FakeLLM(decompose_reply, sufficiency_reply)
    pipeline.bulk_llm = bulk_llm
    pipeline.decomposer = QueryDecomposer(bulk_llm, max_subquestions=3)
    pipeline.sufficiency_checker = SufficiencyChecker(bulk_llm)
    return pipeline


def _stub_finish(pipeline: Pipeline) -> None:
    pipeline._finish_retrieval = lambda *a, **k: _RetrievalState(
        query_vector=[0.0], routed=[], scored_candidates=[], reranked_scored=[],
        final_chunks=[], final_chunk_traces=[], wide_fallback=False,
        wide_fallback_reason="", too_large_message=None,
    )
    pipeline._merge_candidates = lambda sub_qs, states: (states[0], {}, {}, {})


def test_single_question_with_no_evidence_is_insufficient() -> None:
    """A single, non-decomposed question is checked for sufficiency too now, not just
    decomposed sub-questions — zero chunks means insufficient regardless of mode. The
    retry pass still gets its one attempt (this fake's retry-rewrite always returns ""
    — see class docstring — so it stays insufficient rather than finding new evidence)."""
    pipeline = _pipeline_stub("NONE\nSINGLE")
    pipeline._retrieve_candidates = lambda q, space_id, **_kw: _CandidateState(
        query_vector=[0.0], routed=[], scored_candidates=[], reranked_scored=[],
    )
    _stub_finish(pipeline)

    state = pipeline._retrieve("what does the Router class do?", "demo")

    assert state.sufficiency == "insufficient"
    assert state.insufficient_sub_questions == ["what does the Router class do?"]


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


def test_high_scoring_chunk_skips_the_sufficiency_checker_call() -> None:
    """Above sufficiency_check_max_score, the score alone is trusted — no extra call."""
    pipeline = _pipeline_stub("NONE\nSINGLE", sufficiency_reply="this should never be read")
    pipeline._retrieve_candidates = lambda q, space_id, **_kw: _CandidateState(
        [0.0], [], [], [(_chunk("r1"), 0.9)],
    )
    _stub_finish(pipeline)

    state = pipeline._retrieve("what does the Router class do?", "demo")

    assert state.sufficiency == "sufficient"
    assert pipeline.bulk_llm.calls == 1  # just decompose — not even the sufficiency call fired


def test_ambiguous_score_calls_sufficiency_checker_and_trusts_a_sufficient_verdict() -> None:
    """Below the ceiling, on-topic-but-maybe-incomplete evidence gets read, not just
    scored — a SUFFICIENT verdict means no retry, same as a confidently high score."""
    pipeline = _pipeline_stub("NONE\nSINGLE", sufficiency_reply="SUFFICIENT")
    pipeline._retrieve_candidates = lambda q, space_id, **_kw: _CandidateState(
        [0.0], [], [], [(_chunk("r1"), 0.1)],
    )
    _stub_finish(pipeline)

    state = pipeline._retrieve("what does the Router class do?", "demo")

    assert state.sufficiency == "sufficient"
    assert state.insufficient_sub_questions == []
    assert pipeline.bulk_llm.calls == 2  # decompose + the sufficiency call, no retry needed


def test_ambiguous_score_with_a_flagged_gap_marks_insufficient_and_retries() -> None:
    """The real Krishna/Arjuna case: a chunk scores above the pass floor (on-topic) but
    doesn't name the specific weapon — the checker catches what the score alone missed,
    and that feeds the same retry pass a zero-chunk sub-question would trigger."""
    pipeline = _pipeline_stub("NONE\nSINGLE", sufficiency_reply="the specific weapon given")
    pipeline._retrieve_candidates = lambda q, space_id, **_kw: _CandidateState(
        [0.0], [], [], [(_chunk("r1"), 0.1)],
    )
    _stub_finish(pipeline)

    state = pipeline._retrieve("What weapon did Krishna give Arjuna?", "demo")

    assert state.sufficiency == "insufficient"
    assert state.insufficient_sub_questions == ["What weapon did Krishna give Arjuna?"]
    # decompose + the sufficiency call + 1 retry-rewrite call (this fake's retry always
    # returns "", i.e. no better query found, so it stays insufficient rather than
    # silently being cleared just because it still holds its original, non-empty chunks).
    assert pipeline.bulk_llm.calls == 3


def test_broad_single_question_skips_sufficiency_check_and_retry() -> None:
    """Regression: a single question the router already classified as broad gets its
    hybrid+rerank skipped ON PURPOSE (see Pipeline._retrieve_candidates'
    skip_hybrid_rerank), leaving reranked_scored empty by design — meant to fall through
    to _finish_retrieval's wide-fallback path. Treating that empty result as
    "insufficient, retry narrowly" (as a real bug briefly did) would silently run a full
    search the router deliberately skipped, and never let wide-fallback happen at all."""
    pipeline = _pipeline_stub("unused — decompose_result is pre-routed, not derived here")
    question = "What happened during the game of dice?"
    route = DecomposeResult("single", [question], is_broad=True)
    pipeline._retrieve_candidates = lambda q, space_id, **_kw: _CandidateState(
        [0.0], [], [], [],  # empty on purpose — this is what skip_hybrid_rerank leaves behind
    )
    _stub_finish(pipeline)

    state = pipeline._retrieve(question, "demo", decompose_result=route)

    assert state.insufficient_sub_questions == []
    assert state.retried_sub_questions == {}
    assert pipeline.bulk_llm.calls == 0  # no sufficiency call, no retry call, no decompose call


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
