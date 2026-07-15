from __future__ import annotations

from src.index.schema import CodeChunk
from src.retrieve.compressor import Compressor


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
        id="a.py::10-13",
        repo="demo",
        file_path="a.py",
        language="python",
        symbol_name="run",
        start_line=10,
        end_line=13,
        code="line one\nline two\nline three\nline four",
    )


def test_compress_narrows_to_contiguous_relevant_window() -> None:
    compressor = Compressor(FakeLLM(response="2,3"))

    result = compressor.compress("question", _chunk())

    assert result.code == "line two\nline three"
    assert (result.start_line, result.end_line) == (11, 12)


def test_compress_returns_none_when_llm_says_none() -> None:
    compressor = Compressor(FakeLLM(response="NONE"))

    assert compressor.compress("question", _chunk()) is None


def test_compress_keeps_whole_chunk_on_llm_failure() -> None:
    compressor = Compressor(FakeLLM(fail=True))
    chunk = _chunk()

    assert compressor.compress("question", chunk) == chunk
