from __future__ import annotations

from src.llm_client import LLMClient

SYSTEM_PROMPT = (
    "Rewrite the latest user question into a standalone question that can be understood "
    "without the earlier conversation, by resolving pronouns and implicit references using "
    "the conversation history. Preserve the user's intent exactly — do not answer the "
    "question, expand its scope, or add assumptions. Reply with only the rewritten "
    "question, no preamble."
)


class StandaloneRewriter:
    """Collapses (history, follow-up question) into one self-contained question — the
    only thing retrieval ever embeds. Skipped on turn 1: there's nothing to resolve."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def rewrite(self, question: str, history: list[tuple[str, str]]) -> str:
        if not history:
            return question
        transcript = "\n".join(f"{role}: {content}" for role, content in history)
        prompt = f"Conversation so far:\n{transcript}\n\nLatest question: {question}"
        try:
            rewritten = self.llm.complete(prompt, system=SYSTEM_PROMPT).strip()
        except RuntimeError:
            return question
        return rewritten or question
