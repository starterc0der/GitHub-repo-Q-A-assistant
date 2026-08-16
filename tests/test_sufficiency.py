from __future__ import annotations

from src.generate.sufficiency import SufficiencyChecker
from src.index.schema import CodeChunk


class _FakeLLM:
    def __init__(self, reply: str = "SUFFICIENT") -> None:
        self.reply = reply
        self.last_prompt: str | None = None

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.last_prompt = prompt
        return self.reply


class _FailingLLM:
    def complete(self, prompt: str, system: str | None = None) -> str:
        raise RuntimeError("quota exhausted")


def _chunk(chunk_id: str = "c1") -> CodeChunk:
    return CodeChunk(
        id=chunk_id, space_id="demo", source_id="src1", file_path="a.py",
        language="text", symbol_name=None, start_line=1, end_line=1, code="Krishna and Arjuna spoke.",
    )


def test_check_returns_none_when_no_chunks() -> None:
    checker = SufficiencyChecker(_FakeLLM("this should never be read"))

    assert checker.check("What weapon?", []) is None


def test_check_returns_none_when_llm_says_sufficient() -> None:
    checker = SufficiencyChecker(_FakeLLM("SUFFICIENT"))

    assert checker.check("What weapon?", [_chunk()]) is None


def test_check_returns_missing_description_when_llm_flags_a_gap() -> None:
    checker = SufficiencyChecker(_FakeLLM("the specific weapon given"))

    assert checker.check("What weapon?", [_chunk()]) == "the specific weapon given"


def test_check_fails_open_on_llm_error() -> None:
    checker = SufficiencyChecker(_FailingLLM())

    assert checker.check("What weapon?", [_chunk()]) is None
