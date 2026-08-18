from __future__ import annotations

from src.generate.decomposer import HOP_RESOLVE_SYSTEM_PROMPT, META_MARKER, QueryDecomposer


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


def test_decompose_always_calls_the_llm_even_for_a_simple_question() -> None:
    """No keyword pre-filter — is_broad/wants_chart/is_meta are semantic judgments a
    keyword list can't make, so every question pays for this one cheap call."""
    llm = FakeLLM(response="NONE\nSINGLE")
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    result = decomposer.decompose("what does the Router class do?")

    assert result.mode == "single"
    assert result.sub_questions == ["what does the Router class do?"]
    assert llm.calls == 1


def test_decompose_classifies_meta_questions() -> None:
    llm = FakeLLM(response=META_MARKER)
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    result = decomposer.decompose("hi, what can you help with?")

    assert result.is_meta is True
    assert result.sub_questions == ["hi, what can you help with?"]


def test_decompose_classifies_broad_flag() -> None:
    llm = FakeLLM(response="BROAD\nSINGLE")
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    result = decomposer.decompose("give me an overview of this repo")

    assert result.is_broad is True
    assert result.wants_chart is False
    assert result.mode == "single"


def test_decompose_classifies_chart_flag() -> None:
    llm = FakeLLM(response="CHART\nSINGLE")
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    result = decomposer.decompose("compare Q1 and Q2 revenue as a chart")

    assert result.wants_chart is True
    assert result.is_broad is False


def test_decompose_classifies_both_broad_and_chart_flags_together() -> None:
    llm = FakeLLM(response="BROAD CHART\nSINGLE")
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    result = decomposer.decompose("chart the trend across the whole document")

    assert result.is_broad is True
    assert result.wants_chart is True


def test_decompose_classifies_live_flag() -> None:
    llm = FakeLLM(response="LIVE\nSINGLE")
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    result = decomposer.decompose("what is the pressure at Zone_11 right now")

    assert result.wants_live_data is True
    assert result.is_broad is False
    assert result.wants_chart is False


def test_decompose_parses_explicit_parallel_marker() -> None:
    llm = FakeLLM(
        response="NONE\nPARALLEL\nWhat does the Router class do?\nHow is that different from HybridSearch?"
    )
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    result = decomposer.decompose("what does Router do, and how is that different from HybridSearch?")

    assert result.mode == "parallel"
    assert result.sub_questions == [
        "What does the Router class do?",
        "How is that different from HybridSearch?",
    ]


def test_decompose_parses_sequential_marker_with_hop_placeholder() -> None:
    llm = FakeLLM(
        response="NONE\nSEQUENTIAL\nWhich department had the most failures?\n"
        "What policy caused failures in {hop1}?"
    )
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    result = decomposer.decompose("which department had the most failures, and what policy caused them?")

    assert result.mode == "sequential"
    assert result.sub_questions == [
        "Which department had the most failures?",
        "What policy caused failures in {hop1}?",
    ]


def test_decompose_sequential_reply_with_only_one_hop_falls_back_to_single() -> None:
    llm = FakeLLM(response="NONE\nSEQUENTIAL\nWhich department had the most failures?")
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    question = "which department had the most failures, and what policy caused them?"
    result = decomposer.decompose(question)

    assert result.mode == "single"
    assert result.sub_questions == [question]


def test_decompose_parses_sequential_marker_with_three_hops() -> None:
    llm = FakeLLM(
        response="NONE\nSEQUENTIAL\nWhich department had the most failures?\n"
        "What policy caused failures in {hop1}?\nWho approved {hop2}?"
    )
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    result = decomposer.decompose(
        "which department had the most failures, what policy caused them, and who approved it?"
    )

    assert result.mode == "sequential"
    assert result.sub_questions == [
        "Which department had the most failures?",
        "What policy caused failures in {hop1}?",
        "Who approved {hop2}?",
    ]


def test_decompose_caps_at_max_subquestions() -> None:
    llm = FakeLLM(response="NONE\nPARALLEL\n" + "\n".join(f"sub-question {i}" for i in range(5)))
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    result = decomposer.decompose("a and b and c and d and e?")

    assert len(result.sub_questions) == 3


def test_decompose_returns_original_on_single_mode() -> None:
    llm = FakeLLM(response="NONE\nSINGLE")
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    question = "what does the Router class do, and is it fast?"
    result = decomposer.decompose(question)

    assert result.mode == "single"
    assert result.sub_questions == [question]


def test_decompose_falls_back_to_original_on_llm_failure() -> None:
    llm = FakeLLM(fail=True)
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    question = "what does X do and how is that different from Y?"
    result = decomposer.decompose(question)

    assert result.mode == "single"
    assert result.sub_questions == [question]


def test_decompose_falls_back_to_original_on_empty_reply() -> None:
    llm = FakeLLM(response="")
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    question = "what does the Router class do?"
    result = decomposer.decompose(question)

    assert result.mode == "single"
    assert result.sub_questions == [question]


def test_resolve_hop_substitutes_llm_extracted_answer() -> None:
    llm = FakeLLM(response="What policy caused failures in Engineering?")
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    resolved = decomposer.resolve_hop(
        "Which department had the most failures?",
        "Engineering had the most failures this quarter.",
        "What policy caused failures in {hop1}?",
        "{hop1}",
    )

    assert resolved == "What policy caused failures in Engineering?"
    assert llm.calls == 1


def test_resolve_hop_skips_llm_call_when_context_is_empty() -> None:
    llm = FakeLLM(response="unused")
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    resolved = decomposer.resolve_hop(
        "Which department had the most failures?", "", "What policy caused failures in {hop1}?", "{hop1}"
    )

    assert resolved == "What policy caused failures in Which department had the most failures??"
    assert llm.calls == 0


def test_resolve_hop_falls_back_to_concatenation_on_unresolved_reply() -> None:
    llm = FakeLLM(response="UNRESOLVED")
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    resolved = decomposer.resolve_hop(
        "Which department had the most failures?", "some noisy passage", "What policy caused failures in {hop1}?", "{hop1}"
    )

    assert resolved == "What policy caused failures in some noisy passage?"


def test_resolve_hop_falls_back_to_concatenation_on_llm_failure() -> None:
    llm = FakeLLM(fail=True)
    decomposer = QueryDecomposer(llm, max_subquestions=3)

    resolved = decomposer.resolve_hop(
        "Which department had the most failures?", "Engineering", "What policy caused failures in {hop1}?", "{hop1}"
    )

    assert resolved == "What policy caused failures in Engineering?"


def test_resolve_hop_sends_the_documented_system_prompt() -> None:
    seen = {}

    class RecordingLLM:
        def complete(self, prompt: str, system: str | None = None) -> str:
            seen["system"] = system
            return "resolved question"

    decomposer = QueryDecomposer(RecordingLLM(), max_subquestions=3)
    decomposer.resolve_hop("q1", "context", "template {hop1}", "{hop1}")

    assert seen["system"] == HOP_RESOLVE_SYSTEM_PROMPT
