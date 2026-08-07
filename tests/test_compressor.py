from __future__ import annotations

from src.index.schema import CodeChunk
from src.retrieve.compressor import Compressor


class FakeLLM:
    def __init__(self, response: str | None = None, fail: bool = False):
        self.response = response
        self.fail = fail

    def complete(self, prompt: str, system: str | None = None) -> str:
        if self.fail:
            raise RuntimeError("llm unreachable")
        return self.response


def _chunk(file_path: str = "a.py", start: int = 10) -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}::10-13",
        repo="demo",
        file_path=file_path,
        language="python",
        symbol_name="run",
        start_line=start,
        end_line=start + 3,
        code="line one\nline two\nline three\nline four",
    )


def test_compress_narrows_to_contiguous_relevant_window() -> None:
    compressor = Compressor(FakeLLM(response="1: 2,3"))

    result = compressor.compress("question", _chunk())

    assert result.code == "line two\nline three"
    assert (result.start_line, result.end_line) == (11, 12)


def test_compress_returns_none_when_llm_says_none() -> None:
    compressor = Compressor(FakeLLM(response="1: NONE"))

    assert compressor.compress("question", _chunk()) is None


def test_compress_keeps_whole_chunk_on_llm_failure() -> None:
    compressor = Compressor(FakeLLM(fail=True))
    chunk = _chunk()

    assert compressor.compress("question", chunk) == chunk


def test_compress_batch_maps_verdicts_back_to_the_right_chunks() -> None:
    """One call, several chunks — the index in each reply line is what keeps them aligned."""
    compressor = Compressor(FakeLLM(response="1: 2,3\n2: NONE\n3: 1"))
    chunks = [_chunk("a.py", 10), _chunk("b.py", 20), _chunk("c.py", 30)]

    a, b, c = compressor.compress_batch("question", chunks)

    assert (a.code, a.start_line, a.end_line) == ("line two\nline three", 11, 12)
    assert b is None
    assert (c.code, c.start_line, c.end_line) == ("line one", 30, 30)


def test_compress_batch_keeps_chunks_the_model_never_mentioned() -> None:
    """A truncated reply must not silently drop the tail — unjudged is not irrelevant."""
    compressor = Compressor(FakeLLM(response="1: 2,3"))
    chunks = [_chunk("a.py", 10), _chunk("b.py", 20)]

    a, b = compressor.compress_batch("question", chunks)

    assert a.code == "line two\nline three"
    assert b == chunks[1]  # untouched, not dropped


def test_compress_batch_tolerates_separator_drift() -> None:
    compressor = Compressor(FakeLLM(response="1 - 2,3\n2) NONE"))
    a, b = compressor.compress_batch("question", [_chunk("a.py", 10), _chunk("b.py", 20)])
    assert a.code == "line two\nline three"
    assert b is None


def test_compress_batch_ignores_out_of_range_chunk_indexes() -> None:
    compressor = Compressor(FakeLLM(response="1: 2\n9: 1,2,3"))
    (a,) = compressor.compress_batch("question", [_chunk("a.py", 10)])
    assert a.code == "line two"


def test_compress_batch_keeps_everything_whole_on_llm_failure() -> None:
    compressor = Compressor(FakeLLM(fail=True))
    chunks = [_chunk("a.py", 10), _chunk("b.py", 20)]

    assert compressor.compress_batch("question", chunks) == chunks


def test_compress_batch_on_empty_input() -> None:
    assert Compressor(FakeLLM(response="")).compress_batch("question", []) == []
