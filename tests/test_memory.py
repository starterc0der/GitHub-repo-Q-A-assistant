from __future__ import annotations

from src.generate.memory import ConversationMemory


class _FakeLLM:
    def __init__(self, reply: str = "a fact") -> None:
        self.reply = reply
        self.last_prompt: str | None = None

    def complete(self, prompt: str, system: str | None = None, history=None) -> str:
        self.last_prompt = prompt
        return self.reply


class _FailingLLM:
    def complete(self, prompt: str, system: str | None = None, history=None) -> str:
        raise RuntimeError("quota exhausted")


def test_extract_returns_the_fact() -> None:
    llm = _FakeLLM(reply="Bhishma took a vow of celibacy.")
    memory = ConversationMemory(llm)

    result = memory.extract("What vow did Bhishma take?", "Bhishma took a vow of celibacy.")

    assert result == "Bhishma took a vow of celibacy."
    assert "What vow did Bhishma take?" in llm.last_prompt
    assert "Bhishma took a vow of celibacy." in llm.last_prompt


def test_extract_returns_none_for_the_none_sentinel() -> None:
    memory = ConversationMemory(_FakeLLM(reply="NONE"))

    result = memory.extract("What is the capital of Mars?", "The source material does not cover this.")

    assert result is None


def test_extract_returns_none_on_llm_failure() -> None:
    memory = ConversationMemory(_FailingLLM())

    result = memory.extract("What vow did Bhishma take?", "Bhishma took a vow of celibacy.")

    assert result is None
