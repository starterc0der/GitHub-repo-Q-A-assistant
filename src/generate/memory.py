from __future__ import annotations

from src.llm_client import LLMClient

SYSTEM_PROMPT = (
    "You maintain a compact running summary of an earlier part of a conversation, so a "
    "model that only sees the summary plus the most recent turns still has continuity "
    "with what came before. Given the EXISTING SUMMARY (empty on the first fold) and the "
    "NEW TURNS to fold in, write an updated summary covering both — a few sentences of "
    "plain prose, no preamble. Preserve concrete facts, names, and decisions; drop small "
    "talk and phrasing detail."
)


class ConversationMemory:
    """Keeps a chat's un-windowed history from either disappearing (a fixed-N recency
    window forgets everything older) or being replayed in full forever (sending the whole
    chat on every meta/recap question). fold() runs once per turn, only for the turn(s)
    that just aged out of the window — never re-summarizes turns already folded in."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def fold(self, existing_summary: str, new_turns: list[tuple[str, str]]) -> str:
        if not new_turns:
            return existing_summary
        transcript = "\n".join(f"{role}: {content}" for role, content in new_turns)
        prompt = f"EXISTING SUMMARY: {existing_summary or '(none yet)'}\n\nNEW TURNS:\n{transcript}"
        try:
            return self.llm.complete(prompt, system=SYSTEM_PROMPT).strip() or existing_summary
        except RuntimeError:
            return existing_summary  # keep the old summary rather than losing it on a flaky call
