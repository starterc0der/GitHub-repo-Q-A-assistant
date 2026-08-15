from __future__ import annotations

from src.generate.memory import ConversationMemory


class _FakeLLM:
    def __init__(self, reply: str = "summary") -> None:
        self.reply = reply
        self.last_prompt: str | None = None

    def complete(self, prompt: str, system: str | None = None, history=None) -> str:
        self.last_prompt = prompt
        return self.reply


class _FailingLLM:
    def complete(self, prompt: str, system: str | None = None, history=None) -> str:
        raise RuntimeError("quota exhausted")


def test_fold_returns_existing_summary_unchanged_when_no_new_turns() -> None:
    memory = ConversationMemory(_FakeLLM())

    result = memory.fold("old summary", [])

    assert result == "old summary"


def test_fold_calls_llm_with_existing_summary_and_new_turns() -> None:
    llm = _FakeLLM(reply="updated summary")
    memory = ConversationMemory(llm)

    result = memory.fold("earlier stuff", [("user", "hi"), ("assistant", "hello")])

    assert result == "updated summary"
    assert "earlier stuff" in llm.last_prompt
    assert "user: hi" in llm.last_prompt
    assert "assistant: hello" in llm.last_prompt


def test_fold_keeps_old_summary_on_llm_failure() -> None:
    memory = ConversationMemory(_FailingLLM())

    result = memory.fold("old summary", [("user", "hi"), ("assistant", "hello")])

    assert result == "old summary"
