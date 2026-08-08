from __future__ import annotations

from src.index.schema import CodeChunk, FileSummary
from src.ingest.contextualizer import Contextualizer


class FakeLLM:
    def __init__(self, response: str | None = None, fail: bool = False):
        self.response = response
        self.fail = fail

    def complete(self, prompt: str, system: str | None = None) -> str:
        if self.fail:
            raise RuntimeError("ollama unreachable")
        return self.response


def _chunk(language: str = "python") -> CodeChunk:
    return CodeChunk(
        id="a.py::1-1",
        space_id="demo",
        source_id="src1",
        file_path="a.py",
        language=language,
        symbol_name="run",
        start_line=1,
        end_line=1,
        code="def run(): pass",
    )


def test_add_context_header_uses_llm_response() -> None:
    contextualizer = Contextualizer(FakeLLM(response="1: From a.py, runs the thing."))
    summary = FileSummary(space_id="demo",
        source_id="src1", file_path="a.py", language="python", summary="s")

    result = contextualizer.add_context_header(_chunk(), summary, "repo summary")

    assert result.context_header == "From a.py, runs the thing."
    assert result.code == "def run(): pass"


def test_add_context_headers_batch_covers_every_chunk_in_one_call() -> None:
    calls = []

    class RecordingLLM(FakeLLM):
        def complete(self, prompt, system=None):
            calls.append(prompt)
            return "1: From a.py, first.\n2: From a.py, second."

    contextualizer = Contextualizer(RecordingLLM())
    summary = FileSummary(space_id="demo",
        source_id="src1", file_path="a.py", language="python", summary="s")
    chunks = [_chunk(), _chunk()]

    result = contextualizer.add_context_headers_batch(chunks, summary, "repo summary")

    assert len(calls) == 1
    assert [c.context_header for c in result] == [
        "From a.py, first.",
        "From a.py, second.",
    ]


def test_add_context_header_falls_back_to_template_on_llm_failure() -> None:
    contextualizer = Contextualizer(FakeLLM(fail=True))
    summary = FileSummary(space_id="demo",
        source_id="src1", file_path="a.py", language="python", summary="does things")

    result = contextualizer.add_context_header(_chunk(), summary, "repo summary")

    assert result.context_header == "From a.py (does things). Defines run."


def test_prose_chunks_skip_the_llm_entirely() -> None:
    class ExplodingLLM(FakeLLM):
        def complete(self, prompt, system=None):
            raise AssertionError("prose headers must not call the LLM")

    contextualizer = Contextualizer(ExplodingLLM())
    summary = FileSummary(space_id="demo",
        source_id="src1", file_path="notes.txt", language="text", summary="notes")

    result = contextualizer.add_context_headers_batch(
        [_chunk(language="text")], summary, "repo summary"
    )

    assert result[0].context_header == "From a.py (notes). Defines run."
