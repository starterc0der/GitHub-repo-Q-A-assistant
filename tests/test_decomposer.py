from __future__ import annotations

from src.generate.decomposer import SINGLE_SENTINEL, QueryDecomposer, looks_compound


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


def test_looks_compound_flags_coordinating_language() -> None:
    assert looks_compound("what does X do and how is that different from Y?")
    assert looks_compound("who wrote it? when was it published?")
    assert not looks_compound("what does the Router class do?")


def test_decompose_skips_llm_call_for_simple_questions() -> None:
    llm = FakeLLM(response="unused")
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    result = decomposer.decompose("what does the Router class do?")

    assert result == ["what does the Router class do?"]
    assert llm.calls == 0


def test_decompose_splits_a_compound_question() -> None:
    llm = FakeLLM(response="What does the Router class do?\nHow is that different from HybridSearch?")
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    result = decomposer.decompose("what does Router do, and how is that different from HybridSearch?")

    assert result == [
        "What does the Router class do?",
        "How is that different from HybridSearch?",
    ]


def test_decompose_caps_at_max_subquestions() -> None:
    llm = FakeLLM(response="\n".join(f"sub-question {i}" for i in range(5)))
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    result = decomposer.decompose("a and b and c and d and e?")

    assert len(result) == 3


def test_decompose_returns_original_on_single_sentinel() -> None:
    llm = FakeLLM(response=SINGLE_SENTINEL)
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    question = "what does the Router class do, and is it fast?"
    assert decomposer.decompose(question) == [question]


def test_decompose_falls_back_to_original_on_llm_failure() -> None:
    llm = FakeLLM(fail=True)
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    question = "what does X do and how is that different from Y?"
    assert decomposer.decompose(question) == [question]
