from __future__ import annotations

from src.llm_client import LLMClient

NONE_SENTINEL = "NONE"

SYSTEM_PROMPT = (
    "You extract one atomic fact from a single question-and-answer turn in a "
    "conversation, so it can be remembered later without re-reading the whole exchange. "
    "Write one short, self-contained sentence stating what was established — concrete "
    "facts, names, and decisions, not the question itself or how it was phrased.\n"
    "Answer only from what THIS TURN'S ANSWER actually states — never use outside "
    "knowledge to fill in what it doesn't say, even if you recognize the topic. If the "
    f"answer is a refusal, says the source material doesn't cover it, or establishes "
    f"nothing concrete, reply with exactly: {NONE_SENTINEL}"
)


class ConversationMemory:
    """Extracts one atomic fact per aged-out turn and keeps it permanently, instead of
    repeatedly re-summarizing the whole running memory into fresh prose. A fact rewritten
    on every fold erodes over many turns — an early fact can get rewritten dozens of
    times by turn 50, and testing showed this genuinely happening: two of the earliest
    facts in a 15-turn conversation were completely gone by the end, question order was
    lost entirely, and the model started blending in outside knowledge the conversation
    never actually stated. An atomic fact extracted once and never touched again can't
    erode or drift — see src/api/chat_routes.py's _windowed_history for how facts
    accumulate turn by turn."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract(self, question: str, answer: str) -> str | None:
        """Returns a short fact string, or None when this turn established nothing
        concrete (a refusal, NO_MATCH, or an LLM call failure — never guessed)."""
        prompt = f"Question: {question}\nAnswer: {answer}"
        try:
            reply = self.llm.complete(prompt, system=SYSTEM_PROMPT).strip()
        except RuntimeError:
            return None
        if not reply or reply == NONE_SENTINEL:
            return None
        return reply
