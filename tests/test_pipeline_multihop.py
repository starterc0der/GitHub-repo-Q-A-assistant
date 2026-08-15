from __future__ import annotations

from src.config import Settings
from src.generate.decomposer import HOP_RESOLVE_SYSTEM_PROMPT, RETRY_REWRITE_SYSTEM_PROMPT, QueryDecomposer
from src.index.schema import CodeChunk
from src.pipeline import Pipeline, _CandidateState, _RetrievalState


class _FakeLLM:
    """Answers the decompose call with the queued reply, a hop-resolve call with the
    queued resolve reply (or "UNRESOLVED" — forcing the raw-concatenation fallback — when
    none was supplied), and a retry-rewrite call (fired when a hop is insufficient) with
    "" — unchanged, meaning "nothing to retry" — since these tests are about the
    sequential hop-resolution behavior itself, not the (separately tested) retry pass."""

    def __init__(self, decompose_reply: str, resolve_replies: list[str] | None = None):
        self.decompose_reply = decompose_reply
        self.resolve_replies = list(resolve_replies or [])
        self.last_usage: dict | None = None
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        if system == RETRY_REWRITE_SYSTEM_PROMPT:
            return ""
        if system == HOP_RESOLVE_SYSTEM_PROMPT:
            return self.resolve_replies.pop(0) if self.resolve_replies else "UNRESOLVED"
        return self.decompose_reply


def _chunk(chunk_id: str, code: str) -> CodeChunk:
    return CodeChunk(
        id=chunk_id, space_id="demo", source_id="src1", file_path="a.py",
        language="text", symbol_name=None, start_line=1, end_line=1, code=code,
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


def test_sequential_hop2_resolves_placeholder_from_hop1_top_chunk() -> None:
    llm = _FakeLLM(
        "SEQUENTIAL\nWhich department had the most failures?\n"
        "What policy caused failures in {hop1}?",
        resolve_replies=["What policy caused failures in Engineering?"],
    )
    pipeline = _pipeline_stub(llm)
    states_by_q = {
        "Which department had the most failures?": _CandidateState(
            [0.0], [], [], [(_chunk("d1", "Engineering"), 0.9)]
        ),
        "What policy caused failures in Engineering?": _CandidateState(
            [0.0], [], [], [(_chunk("p1", "Hiring freeze policy"), 0.8)]
        ),
    }
    pipeline._retrieve_candidates = lambda q, space_id: states_by_q[q]
    _stub_finish(pipeline)

    state = pipeline._retrieve(
        "which department had the most failures, and what policy caused them?", "demo"
    )

    assert state.decompose_mode == "sequential"
    assert state.sufficiency == "sufficient"
    assert llm.calls == 2  # decompose + one resolve call for hop 2


def test_sequential_hop1_found_nothing_falls_back_to_hop1_question_text() -> None:
    """When hop 1 itself is insufficient, there's no chunk to build hop 2's context
    from — fall back to substituting hop 1's own question text rather than sending a
    literal unresolved "{hop1}" string to the embedder."""
    llm = _FakeLLM(
        "SEQUENTIAL\nWhich department had the most failures?\n"
        "What policy caused failures in {hop1}?"
    )
    pipeline = _pipeline_stub(llm)
    states_by_q = {
        "Which department had the most failures?": _CandidateState([0.0], [], [], []),
        "What policy caused failures in Which department had the most failures??": _CandidateState(
            [0.0], [], [], []
        ),
    }
    pipeline._retrieve_candidates = lambda q, space_id: states_by_q[q]
    _stub_finish(pipeline)

    state = pipeline._retrieve(
        "which department had the most failures, and what policy caused them?", "demo"
    )

    assert state.decompose_mode == "sequential"
    assert state.sufficiency == "insufficient"


def test_sequential_hop2_without_placeholder_gets_hop1_context_prefixed() -> None:
    llm = _FakeLLM(
        "SEQUENTIAL\nWho won the award?\nWhat team do they play for?"
    )
    pipeline = _pipeline_stub(llm)
    states_by_q = {
        "Who won the award?": _CandidateState([0.0], [], [], [(_chunk("a1", "Alex Ray"), 0.9)]),
        "Alex Ray. What team do they play for?": _CandidateState(
            [0.0], [], [], [(_chunk("t1", "Riverside FC"), 0.9)]
        ),
    }
    pipeline._retrieve_candidates = lambda q, space_id: states_by_q[q]
    _stub_finish(pipeline)

    state = pipeline._retrieve("who won the award, and what team do they play for?", "demo")

    assert state.decompose_mode == "sequential"
    assert state.sufficiency == "sufficient"


def test_sequential_reranked_chunk_code_is_truncated_and_whitespace_collapsed() -> None:
    long_code = ("line one\n" * 40) + "the answer is Engineering"
    llm = _FakeLLM(
        "SEQUENTIAL\nWhich department had the most failures?\n"
        "What policy caused failures in {hop1}?"
    )
    pipeline = _pipeline_stub(llm)
    expected_context = " ".join(long_code.split())[:150]
    states_by_q = {
        "Which department had the most failures?": _CandidateState(
            [0.0], [], [], [(_chunk("d1", long_code), 0.9)]
        ),
        f"What policy caused failures in {expected_context}?": _CandidateState(
            [0.0], [], [], [(_chunk("p1", "Hiring freeze"), 0.8)]
        ),
    }
    pipeline._retrieve_candidates = lambda q, space_id: states_by_q[q]
    _stub_finish(pipeline)

    state = pipeline._retrieve(
        "which department had the most failures, and what policy caused them?", "demo"
    )

    assert state.sufficiency == "sufficient"


def test_sequential_hop_context_concatenates_top_three_reranked_chunks() -> None:
    """resolve_hop is only as good as the raw material it's given — the top-1 chunk
    alone often doesn't contain the specific answer even when a lower-ranked chunk in
    hop 1's own top-k does, so _hop_context pools the top 3."""
    llm = _FakeLLM(
        "SEQUENTIAL\nWho is Arjuna's charioteer?\nWhat advice did {hop1} give?"
    )
    pipeline = _pipeline_stub(llm)
    expected_context = "The chariot thundered onward. Krishna, Arjuna's charioteer, spoke. Kaurava forces massed nearby."
    states_by_q = {
        "Who is Arjuna's charioteer?": _CandidateState(
            [0.0], [], [], [
                (_chunk("c1", "The chariot thundered onward."), 0.9),
                (_chunk("c2", "Krishna, Arjuna's charioteer, spoke."), 0.85),
                (_chunk("c3", "Kaurava forces massed nearby."), 0.8),
                (_chunk("c4", "not included — beyond top 3"), 0.7),
            ]
        ),
        f"What advice did {expected_context} give?": _CandidateState(
            [0.0], [], [], [(_chunk("a1", "advice"), 0.8)]
        ),
    }
    pipeline._retrieve_candidates = lambda q, space_id: states_by_q[q]
    _stub_finish(pipeline)

    state = pipeline._retrieve("who is Arjuna's charioteer, and what advice did they give?", "demo")

    assert state.sufficiency == "sufficient"


def test_sequential_three_hop_chain_resolves_each_placeholder_from_the_previous_hop() -> None:
    llm = _FakeLLM(
        "SEQUENTIAL\nWhich department had the most failures?\n"
        "What policy caused failures in {hop1}?\nWho approved {hop2}?",
        resolve_replies=[
            "What policy caused failures in Engineering?",
            "Who approved the hiring freeze policy?",
        ],
    )
    pipeline = _pipeline_stub(llm)
    states_by_q = {
        "Which department had the most failures?": _CandidateState(
            [0.0], [], [], [(_chunk("d1", "Engineering"), 0.9)]
        ),
        "What policy caused failures in Engineering?": _CandidateState(
            [0.0], [], [], [(_chunk("p1", "the hiring freeze policy"), 0.8)]
        ),
        "Who approved the hiring freeze policy?": _CandidateState(
            [0.0], [], [], [(_chunk("a1", "The CFO approved it"), 0.7)]
        ),
    }
    pipeline._retrieve_candidates = lambda q, space_id: states_by_q[q]
    _stub_finish(pipeline)

    state = pipeline._retrieve(
        "which department had the most failures, what policy caused them, and who approved it?",
        "demo",
    )

    assert state.decompose_mode == "sequential"
    assert state.sufficiency == "sufficient"
    assert llm.calls == 3  # decompose + one resolve call per later hop
