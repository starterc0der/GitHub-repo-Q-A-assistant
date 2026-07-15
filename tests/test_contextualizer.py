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


def _chunk() -> CodeChunk:
    return CodeChunk(
        id="a.py::1-1",
        repo="demo",
        file_path="a.py",
        language="python",
        symbol_name="run",
        start_line=1,
        end_line=1,
        code="def run(): pass",
    )


def test_add_context_header_uses_llm_response() -> None:
    contextualizer = Contextualizer(FakeLLM(response=" From a.py, runs the thing. "))
    summary = FileSummary(repo="demo", file_path="a.py", language="python", summary="s")

    result = contextualizer.add_context_header(_chunk(), summary, "repo summary")

    assert result.context_header == "From a.py, runs the thing."
    assert result.code == "def run(): pass"


def test_add_context_header_falls_back_to_template_on_llm_failure() -> None:
    contextualizer = Contextualizer(FakeLLM(fail=True))
    summary = FileSummary(repo="demo", file_path="a.py", language="python", summary="does things")

    result = contextualizer.add_context_header(_chunk(), summary, "repo summary")

    assert result.context_header == "From a.py (does things). Defines run."
