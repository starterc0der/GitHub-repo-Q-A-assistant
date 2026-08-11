from __future__ import annotations

from src.generate.rewriter import META_SENTINEL, StandaloneRewriter


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
    rewriter = StandaloneRewriter(llm)
    is_meta, text = rewriter.rewrite("does PTO roll over?", [])
    assert (is_meta, text) == (False, "does PTO roll over?")
    assert llm.calls == 0


def test_rewrite_returns_standalone_question() -> None:
    llm = FakeLLM(response="Does unused sick leave roll over into next year?")
    rewriter = StandaloneRewriter(llm)
    history = [("user", "does PTO roll over?"), ("assistant", "up to 5 days")]
    is_meta, text = rewriter.rewrite("what about sick leave?", history)
    assert is_meta is False
    assert text == "Does unused sick leave roll over into next year?"


def test_rewrite_classifies_meta_conversation_questions() -> None:
    """The LLM, not a keyword list, decides — so any phrasing of "summarize this chat"
    is caught, not just the ones someone thought to hardcode."""
    llm = FakeLLM(response=META_SENTINEL)
    rewriter = StandaloneRewriter(llm)
    history = [("user", "who is Gandhari?"), ("assistant", "Dhritarashtra's wife")]
    is_meta, text = rewriter.rewrite("what's the gist of what we've covered so far?", history)
    assert is_meta is True
    assert text == "what's the gist of what we've covered so far?"


def test_rewrite_falls_back_to_raw_question_on_llm_failure() -> None:
    llm = FakeLLM(fail=True)
    rewriter = StandaloneRewriter(llm)
    is_meta, text = rewriter.rewrite("what about sick leave?", [("user", "hi"), ("assistant", "hi")])
    assert (is_meta, text) == (False, "what about sick leave?")


def test_rewrite_falls_back_to_raw_question_on_empty_reply() -> None:
    llm = FakeLLM(response="")
    rewriter = StandaloneRewriter(llm)
    is_meta, text = rewriter.rewrite("what about sick leave?", [("user", "hi"), ("assistant", "hi")])
    assert (is_meta, text) == (False, "what about sick leave?")
