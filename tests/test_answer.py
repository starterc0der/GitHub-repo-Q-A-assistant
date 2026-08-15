from __future__ import annotations

from src.generate.answer import AnswerGenerator
from src.index.schema import CodeChunk


class _FakeLLM:
    def complete(self, prompt: str, system: str | None = None, history=None) -> str:
        return "unused"


def _chunk() -> CodeChunk:
    return CodeChunk(
        id="c1", space_id="demo", source_id="src1", file_path="a.py",
        language="text", symbol_name=None, start_line=1, end_line=1, code="x = 1",
    )


def test_build_prompt_adds_chart_hint_when_wants_chart() -> None:
    gen = AnswerGenerator(_FakeLLM())

    prompt = gen.build_prompt("compare Q1 and Q2", [_chunk()], wants_chart=True)

    assert "chart" in prompt.lower()
    assert "classified as asking for a comparison or graph" in prompt


def test_build_prompt_omits_chart_hint_by_default() -> None:
    gen = AnswerGenerator(_FakeLLM())

    prompt = gen.build_prompt("what does x do?", [_chunk()])

    assert "classified as asking for a comparison" not in prompt
