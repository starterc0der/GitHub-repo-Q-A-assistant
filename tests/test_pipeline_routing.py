from __future__ import annotations

from src.config import Settings
from src.generate.decomposer import DecomposeResult, QueryDecomposer
from src.generate.rewriter import StandaloneRewriter
from src.index.schema import CodeChunk
from src.pipeline import Pipeline, _CandidateState, _RetrievalState


class _FakeLLM:
    def __init__(self, reply: str = "NONE\nSINGLE"):
        self.reply = reply
        self.last_usage: dict | None = None
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 50}
        return self.reply


def _routing_pipeline_stub(reply: str = "NONE\nSINGLE") -> tuple[Pipeline, _FakeLLM]:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = Settings()
    bulk_llm = _FakeLLM(reply)
    pipeline.bulk_llm = bulk_llm
    pipeline.decomposer = QueryDecomposer(bulk_llm, max_subquestions=3)
    pipeline.rewriter = StandaloneRewriter(bulk_llm, max_subquestions=3)
    return pipeline, bulk_llm


def test_route_question_uses_decomposer_on_turn_one() -> None:
    pipeline, llm = _routing_pipeline_stub("NONE\nSINGLE")

    result, ms, tokens = pipeline.route_question("what does the Router class do?", [])

    assert result.mode == "single"
    assert llm.calls == 1
    assert ms >= 0
    assert tokens == {"prompt_tokens": 100, "completion_tokens": 50}


def test_route_question_uses_rewriter_when_history_present() -> None:
    pipeline, llm = _routing_pipeline_stub("NONE\nSINGLE\nDoes sick leave roll over?")
    history = [("user", "does PTO roll over?"), ("assistant", "up to 5 days")]

    result, ms, tokens = pipeline.route_question("what about sick leave?", history)

    assert result.sub_questions == ["Does sick leave roll over?"]
    assert llm.calls == 1
    assert tokens == {"prompt_tokens": 100, "completion_tokens": 50}


def test_route_question_classifies_identically_on_turn_one_and_turn_two() -> None:
    """The whole point of unifying decompose/rewrite behind one grammar: a greeting is a
    greeting regardless of which turn it's asked on."""
    pipeline, _llm = _routing_pipeline_stub("META")
    turn_one, _, _ = pipeline.route_question("hi there", [])

    pipeline2, _llm2 = _routing_pipeline_stub("META")
    turn_two, _, _ = pipeline2.route_question("hi there", [("user", "earlier"), ("assistant", "answer")])

    assert turn_one.is_meta is True
    assert turn_two.is_meta is True


def _retrieve_pipeline_stub(bulk_llm: _FakeLLM) -> Pipeline:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = Settings()
    pipeline.bulk_llm = bulk_llm
    pipeline.decomposer = QueryDecomposer(bulk_llm, max_subquestions=3)
    return pipeline


def _stub_finish(pipeline: Pipeline) -> None:
    pipeline._finish_retrieval = lambda *a, **k: _RetrievalState(
        query_vector=[0.0], routed=[], scored_candidates=[], reranked_scored=[],
        final_chunks=[], final_chunk_traces=[], wide_fallback=True,
        wide_fallback_reason="no chunk answered this on its own", too_large_message=None,
    )
    pipeline._merge_candidates = lambda sub_qs, states: (states[0], {}, {}, {})


def test_retrieve_skips_hybrid_and_rerank_for_a_broad_question() -> None:
    """The whole point of upfront broad classification: don't pay for hybrid search +
    cross-rerank when the router already knows the answer needs the wide-source path,
    not a narrow best-match."""
    llm = _FakeLLM()
    pipeline = _retrieve_pipeline_stub(llm)
    seen_kwargs = {}

    def fake_retrieve_candidates(question, space_id, **kwargs):
        seen_kwargs.update(kwargs)
        return _CandidateState(query_vector=[0.0], routed=[("f.py", "src1", 0.9)], scored_candidates=[], reranked_scored=[])

    pipeline._retrieve_candidates = fake_retrieve_candidates
    _stub_finish(pipeline)
    route = DecomposeResult("single", ["summarize this repo"], is_broad=True)

    pipeline._retrieve("summarize this repo", "demo", decompose_result=route)

    assert seen_kwargs.get("skip_hybrid_rerank") is True


def test_retrieve_does_not_skip_hybrid_and_rerank_for_a_narrow_question() -> None:
    llm = _FakeLLM()
    pipeline = _retrieve_pipeline_stub(llm)
    seen_kwargs = {}

    def fake_retrieve_candidates(question, space_id, **kwargs):
        seen_kwargs.update(kwargs)
        return _CandidateState(query_vector=[0.0], routed=[], scored_candidates=[], reranked_scored=[])

    pipeline._retrieve_candidates = fake_retrieve_candidates
    _stub_finish(pipeline)
    route = DecomposeResult("single", ["what does the Router class do?"], is_broad=False)

    pipeline._retrieve("what does the Router class do?", "demo", decompose_result=route)

    assert seen_kwargs.get("skip_hybrid_rerank") is False


def test_retrieve_carries_wants_chart_and_is_broad_onto_the_retrieval_state() -> None:
    llm = _FakeLLM()
    pipeline = _retrieve_pipeline_stub(llm)
    pipeline._retrieve_candidates = lambda q, space_id, **kw: _CandidateState(
        query_vector=[0.0], routed=[], scored_candidates=[], reranked_scored=[]
    )
    _stub_finish(pipeline)
    route = DecomposeResult("single", ["compare Q1 and Q2"], is_broad=True, wants_chart=True)

    state = pipeline._retrieve("compare Q1 and Q2", "demo", decompose_result=route)

    assert state.is_broad is True
    assert state.wants_chart is True
