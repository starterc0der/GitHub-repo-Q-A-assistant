from __future__ import annotations

from src.config import Settings
from src.generate.provenance import ClaimAttributor
from src.index.schema import CodeChunk
from src.pipeline import Pipeline, _RetrievalState
from src.trace import AnswerTrace


def _chunk(file_path: str) -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}::1-1", space_id="demo", source_id="src1", file_path=file_path,
        language="text", symbol_name=None, start_line=1, end_line=1, code="pass",
    )


def _retrieval_state(final_chunks: list[CodeChunk], too_large_message: str | None = None) -> _RetrievalState:
    return _RetrievalState(
        query_vector=[1.0, 0.0], routed=[], scored_candidates=[], reranked_scored=[],
        final_chunks=final_chunks, final_chunk_traces=[], wide_fallback=False,
        wide_fallback_reason="", too_large_message=too_large_message,
    )


class FakeAnswerGenerator:
    """Yields the given deltas and, if fail=True, raises right after — mirrors a real
    mid-stream drop where some text already arrived before the connection broke."""

    def __init__(self, deltas: list[str], fail: bool = False):
        self.deltas = deltas
        self.fail = fail
        self.build_prompt_calls = 0

    def answer_stream(self, question, chunks, history=None, insufficient=None, wants_chart=False):
        yield from self.deltas
        if self.fail:
            raise RuntimeError("stream dropped")

    def finalize(self, text):
        from src.generate.answer import Answer
        return Answer(text=text)

    def build_prompt(self, question, chunks, insufficient=None):
        self.build_prompt_calls += 1
        return "prompt"


class _FakeLLM:
    last_usage: dict | None = None

    def complete(self, prompt: str, system: str | None = None) -> str:
        return ""


def _pipeline_stub(answer_generator, retrieval_state: _RetrievalState) -> Pipeline:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = Settings()
    pipeline.answer_generator = answer_generator
    pipeline.llm = _FakeLLM()
    pipeline.bulk_llm = _FakeLLM()
    pipeline.claim_attributor = ClaimAttributor(pipeline.bulk_llm)
    pipeline._retrieve = lambda question, space_id, *a, **k: retrieval_state
    return pipeline


def _drain(gen):
    deltas = []
    try:
        while True:
            deltas.append(next(gen))
    except StopIteration as stop:
        return deltas, stop.value


def test_query_trace_stream_yields_deltas_then_returns_the_full_trace() -> None:
    state = _retrieval_state([_chunk("a.py")])
    pipeline = _pipeline_stub(FakeAnswerGenerator(["Hello", ", ", "world"]), state)

    deltas, trace = _drain(pipeline.query_trace_stream("who is this", "demo"))

    assert deltas == ["Hello", ", ", "world"]
    assert trace.answer.text == "Hello, world"
    assert trace.answer.error == ""


def test_query_trace_stream_persists_partial_text_on_mid_stream_failure() -> None:
    """A dropped stream shouldn't discard what already streamed to the user — the
    persisted message should match what they actually saw, plus the error."""
    state = _retrieval_state([_chunk("a.py")])
    pipeline = _pipeline_stub(FakeAnswerGenerator(["Hello", ", wo"], fail=True), state)

    deltas, trace = _drain(pipeline.query_trace_stream("who is this", "demo"))

    assert deltas == ["Hello", ", wo"]
    assert trace.answer.text == "Hello, wo"
    assert "stream dropped" in trace.answer.error


def test_query_trace_stream_skips_generation_when_no_chunks() -> None:
    """No chunks (NO_MATCH / too-large refusal) shouldn't call the LLM at all — one
    yielded message, no generator call."""
    state = _retrieval_state([], too_large_message="too large to answer")
    generator = FakeAnswerGenerator(["should never be used"])
    pipeline = _pipeline_stub(generator, state)

    deltas, trace = _drain(pipeline.query_trace_stream("who is this", "demo"))

    assert deltas == ["too large to answer"]
    assert trace.answer.text == "too large to answer"


def test_query_trace_stream_builds_the_same_trace_shape_as_history_field() -> None:
    state = _retrieval_state([_chunk("a.py")])
    pipeline = _pipeline_stub(FakeAnswerGenerator(["hi"]), state)
    history = [("user", "earlier question"), ("assistant", "earlier answer")]

    _, trace = _drain(pipeline.query_trace_stream("q", "demo", history=history))

    assert trace.history == [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
