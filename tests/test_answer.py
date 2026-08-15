from __future__ import annotations

from src.generate.answer import AnswerGenerator
from src.index.schema import CodeChunk


class _FakeLLM:
    def complete(self, prompt: str, system: str | None = None, history=None) -> str:
        return "unused"


def _chunk(
    id: str = "c1", file_path: str = "a.py", start_line: int = 1, end_line: int = 1,
    code: str = "x = 1",
) -> CodeChunk:
    return CodeChunk(
        id=id, space_id="demo", source_id="src1", file_path=file_path,
        language="text", symbol_name=None, start_line=start_line, end_line=end_line, code=code,
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


def test_build_prompt_groups_by_source_and_orders_by_line() -> None:
    # Rerank-score order interleaves A/B/A; each source's block should read as one
    # coherent pass instead, internally ordered by start_line.
    gen = AnswerGenerator(_FakeLLM())
    chunks = [
        _chunk(id="a2", file_path="a.py", start_line=20, code="A_LATE"),
        _chunk(id="b1", file_path="b.py", start_line=1, code="B_ONLY"),
        _chunk(id="a1", file_path="a.py", start_line=1, code="A_EARLY"),
    ]

    prompt = gen.build_prompt("what happens?", chunks)

    a_block = prompt.split("[a.py]")[1].split("[b.py]")[0]
    assert a_block.index("A_EARLY") < a_block.index("A_LATE")
    # a.py's group comes first since its best-ranked chunk (a2) led the input.
    assert prompt.index("[a.py]") < prompt.index("[b.py]")


def test_build_prompt_drops_lowest_ranked_chunks_over_budget() -> None:
    # 1 token ~= 4 chars (Tokenizer.CHARS_PER_TOKEN); budget of 2 tokens fits one
    # 8-char chunk but not two — the tail (lowest-ranked) chunk should be dropped.
    gen = AnswerGenerator(_FakeLLM(), max_context_tokens=2)
    chunks = [_chunk(id="c1", code="AAAAAAAA"), _chunk(id="c2", code="BBBBBBBB")]

    prompt = gen.build_prompt("what happens?", chunks)

    assert "AAAAAAAA" in prompt
    assert "BBBBBBBB" not in prompt


def test_build_prompt_always_keeps_at_least_one_chunk_over_budget() -> None:
    # A single oversized chunk must never be dropped entirely just for exceeding
    # the budget alone — the guard only stops adding MORE chunks after the first.
    gen = AnswerGenerator(_FakeLLM(), max_context_tokens=1)
    chunks = [_chunk(id="c1", code="AAAAAAAAAAAAAAAAAAAA")]

    prompt = gen.build_prompt("what happens?", chunks)

    assert "AAAAAAAAAAAAAAAAAAAA" in prompt
