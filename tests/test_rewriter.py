from __future__ import annotations

from src.generate.decomposer import META_MARKER
from src.generate.rewriter import StandaloneRewriter


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


def test_rewrite_skips_llm_call_on_turn_one() -> None:
    llm = FakeLLM(response="unused")
    rewriter = StandaloneRewriter(llm, max_subquestions=3)

    result = rewriter.rewrite("does PTO roll over?", [])

    assert result.mode == "single"
    assert result.sub_questions == ["does PTO roll over?"]
    assert llm.calls == 0


def test_rewrite_returns_standalone_question() -> None:
    llm = FakeLLM(response="NONE\nSINGLE\nDoes unused sick leave roll over into next year?")
    rewriter = StandaloneRewriter(llm, max_subquestions=3)
    history = [("user", "does PTO roll over?"), ("assistant", "up to 5 days")]

    result = rewriter.rewrite("what about sick leave?", history)

    assert result.mode == "single"
    assert result.sub_questions == ["Does unused sick leave roll over into next year?"]
    assert result.is_meta is False


def test_rewrite_classifies_meta_conversation_questions() -> None:
    """The LLM, not a keyword list, decides — so any phrasing of "summarize this chat"
    is caught, not just the ones someone thought to hardcode."""
    llm = FakeLLM(response=META_MARKER)
    rewriter = StandaloneRewriter(llm, max_subquestions=3)
    history = [("user", "who is Gandhari?"), ("assistant", "Dhritarashtra's wife")]

    result = rewriter.rewrite("what's the gist of what we've covered so far?", history)

    assert result.is_meta is True
    assert result.sub_questions == ["what's the gist of what we've covered so far?"]


def test_rewrite_falls_back_to_raw_question_on_llm_failure() -> None:
    llm = FakeLLM(fail=True)
    rewriter = StandaloneRewriter(llm, max_subquestions=3)

    result = rewriter.rewrite("what about sick leave?", [("user", "hi"), ("assistant", "hi")])

    assert result.mode == "single"
    assert result.sub_questions == ["what about sick leave?"]


def test_rewrite_falls_back_to_raw_question_on_empty_reply() -> None:
    llm = FakeLLM(response="")
    rewriter = StandaloneRewriter(llm, max_subquestions=3)

    result = rewriter.rewrite("what about sick leave?", [("user", "hi"), ("assistant", "hi")])

    assert result.mode == "single"
    assert result.sub_questions == ["what about sick leave?"]


def test_rewrite_classifies_broad_flag() -> None:
    llm = FakeLLM(response="BROAD\nSINGLE\nGive an overview of the whole handbook.")
    rewriter = StandaloneRewriter(llm, max_subquestions=3)

    result = rewriter.rewrite("give me an overview", [("user", "hi"), ("assistant", "hi")])

    assert result.is_broad is True
    assert result.wants_chart is False


def test_rewrite_classifies_chart_flag() -> None:
    llm = FakeLLM(response="CHART\nSINGLE\nCompare Q1 and Q2 revenue.")
    rewriter = StandaloneRewriter(llm, max_subquestions=3)

    result = rewriter.rewrite("compare them", [("user", "what was Q1 revenue?"), ("assistant", "$1M")])

    assert result.wants_chart is True
    assert result.is_broad is False


def test_rewrite_parses_parallel_sub_questions_resolved_against_history() -> None:
    llm = FakeLLM(
        response="NONE\nPARALLEL\nWhat does the Router class do?\nHow is that different from HybridSearch?"
    )
    rewriter = StandaloneRewriter(llm, max_subquestions=3)

    result = rewriter.rewrite("and how is it different from HybridSearch?", [("user", "hi"), ("assistant", "hi")])

    assert result.mode == "parallel"
    assert result.sub_questions == [
        "What does the Router class do?",
        "How is that different from HybridSearch?",
    ]


def test_rewrite_parses_sequential_hops() -> None:
    llm = FakeLLM(
        response="NONE\nSEQUENTIAL\nWhich department had the most failures?\n"
        "What policy caused failures in {hop1}?"
    )
    rewriter = StandaloneRewriter(llm, max_subquestions=3)

    result = rewriter.rewrite("what caused it", [("user", "hi"), ("assistant", "hi")])

    assert result.mode == "sequential"
    assert result.sub_questions == [
        "Which department had the most failures?",
        "What policy caused failures in {hop1}?",
    ]


def test_rewrite_single_mode_falls_back_to_raw_question_when_llm_omits_rewritten_line() -> None:
    llm = FakeLLM(response="NONE\nSINGLE")
    rewriter = StandaloneRewriter(llm, max_subquestions=3)

    result = rewriter.rewrite("what about sick leave?", [("user", "hi"), ("assistant", "hi")])

    assert result.sub_questions == ["what about sick leave?"]
