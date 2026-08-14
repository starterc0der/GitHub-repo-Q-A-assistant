from __future__ import annotations

from src.generate.faithfulness import SUPPORTED_SENTINEL, FaithfulnessChecker
from src.index.schema import CodeChunk


class FakeLLM:
    def __init__(self, response: str | None = None, fail: bool = False):
        self.response = response
        self.fail = fail
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("llm unreachable")
        return self.response


def _chunk(code: str) -> CodeChunk:
    return CodeChunk(
        id="c1", space_id="demo", source_id="src1", file_path="a.py",
        language="text", symbol_name=None, start_line=1, end_line=1, code=code,
    )


def test_check_passes_when_llm_reports_supported() -> None:
    llm = FakeLLM(response=SUPPORTED_SENTINEL)
    checker = FaithfulnessChecker(llm)

    result = checker.check("who is Kripa?", "Kripa is skilled in single combat.", [_chunk("Kripa was well-versed in the rules of single combat.")])

    assert result.supported is True
    assert result.unsupported_claims == []
    assert result.checked is True


def test_check_flags_unsupported_claims() -> None:
    llm = FakeLLM(response="Kripa was a king of Hastinapura.")
    checker = FaithfulnessChecker(llm)

    result = checker.check("who is Kripa?", "Kripa was a king of Hastinapura.", [_chunk("Kripa was well-versed in single combat.")])

    assert result.supported is False
    assert result.unsupported_claims == ["Kripa was a king of Hastinapura."]


def test_check_skips_llm_call_with_no_chunks_or_empty_answer() -> None:
    llm = FakeLLM(response="unused")
    checker = FaithfulnessChecker(llm)

    assert checker.check("q", "an answer", []).supported is True
    assert checker.check("q", "", [_chunk("x")]).supported is True
    assert llm.calls == 0


def test_check_fails_open_on_llm_failure() -> None:
    """An LLM outage must not get reported as "this answer is unfaithful" — that's a
    false accusation, worse than just not checking at all."""
    llm = FakeLLM(fail=True)
    checker = FaithfulnessChecker(llm)

    result = checker.check("q", "an answer", [_chunk("x")])

    assert result.supported is True
    assert result.checked is False


def test_check_treats_explicit_missing_info_as_not_unsupported() -> None:
    """An answer honestly saying 'the chunks don't mention this' should never itself be
    flagged as an unsupported claim — that would punish honesty about a gap."""
    llm = FakeLLM(response=SUPPORTED_SENTINEL)
    checker = FaithfulnessChecker(llm)

    result = checker.check(
        "who is Kripa's father?", "The chunks don't mention who Kripa's father is.",
        [_chunk("Kripa was well-versed in single combat.")],
    )

    assert result.supported is True
