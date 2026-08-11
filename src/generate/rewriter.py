from __future__ import annotations

from src.llm_client import LLMClient

# A fixed sentinel, not a keyword list: users phrase "summarize this chat" in unbounded
# ways ("what's the gist of our talk", "recap what we've covered"...), and no literal
# match list keeps up. Folded into the rewrite call instead of a separate classifier
# call — it already runs on every follow-up turn, so this adds zero extra LLM calls.
META_SENTINEL = "META_CONVERSATION_QUESTION"

SYSTEM_PROMPT = (
    "First decide: is the latest question asking about THIS CONVERSATION itself — what "
    "was discussed, who was mentioned, a summary/recap of the chat — rather than about "
    f"the source document? If so, reply with exactly: {META_SENTINEL}\n"
    "Otherwise, rewrite the latest question into a standalone question that can be "
    "understood without the earlier conversation, by resolving pronouns and implicit "
    "references using the conversation history. Preserve the user's intent exactly — do "
    "not answer the question, expand its scope, or add assumptions. Reply with only the "
    "rewritten question, no preamble."
)


class StandaloneRewriter:
    """Collapses (history, follow-up question) into one self-contained question — the
    only thing retrieval ever embeds. Skipped on turn 1: there's nothing to resolve.

    Also classifies meta/about-the-conversation questions as a side effect of this same
    call (see META_SENTINEL) — no separate LLM call for that."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def rewrite(self, question: str, history: list[tuple[str, str]]) -> tuple[bool, str]:
        """Returns (is_meta, text). text is the standalone question, or the original
        question unchanged when is_meta is True, on turn 1, or on LLM failure."""
        if not history:
            return False, question
        transcript = "\n".join(f"{role}: {content}" for role, content in history)
        prompt = f"Conversation so far:\n{transcript}\n\nLatest question: {question}"
        try:
            reply = self.llm.complete(prompt, system=SYSTEM_PROMPT).strip()
        except RuntimeError:
            return False, question
        if reply == META_SENTINEL:
            return True, question
        return False, reply or question
