from __future__ import annotations

from src.index.schema import CodeChunk
from src.llm_client import LLMClient

SUFFICIENT_SENTINEL = "SUFFICIENT"

SYSTEM_PROMPT = (
    "You judge whether retrieved evidence actually answers a question — not just whether "
    "it's topically related. Given the question and its top retrieved passages, decide: "
    "can a complete, confident answer be written from this evidence alone?\n"
    f"If yes, reply with exactly: {SUFFICIENT_SENTINEL}\n"
    "If no — the evidence is on-topic but doesn't contain the specific fact/answer needed "
    "— reply with one short phrase naming what's missing (e.g. \"the specific weapon "
    "given\", \"the root cause of the failure\"), nothing else."
)


class SufficiencyChecker:
    """Catches what a rerank-score threshold structurally can't: evidence that scored
    above the pass floor (so it's topically relevant) but still doesn't contain the
    specific fact the question needs — e.g. a passage that mentions Krishna, Arjuna, and
    weapons without ever naming which weapon was given. A score only measures relevance;
    this reads the actual text against the actual question.

    Only worth calling in the ambiguous middle band — see Pipeline's use of
    sufficiency_check_max_score. A very low score is already known-insufficient (caught
    upstream, before this ever runs); a score comfortably above that ceiling is
    empirically almost always a real, complete match, so spending a call to confirm
    near-certainty would be wasted cost."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def check(self, question: str, chunks: list[CodeChunk]) -> str | None:
        """Returns None when the evidence is sufficient, or a short string naming what's
        missing. Fails open (None) on any LLM error — same convention as every other
        LLM-backed check in this codebase, so a flaky call never blocks an answer that
        would otherwise have gone through."""
        if not chunks:
            return None
        context = "\n\n".join(f"[{c.file_path}]\n{c.code}" for c in chunks)
        prompt = f"Question: {question}\n\nRetrieved evidence:\n{context}"
        try:
            reply = self.llm.complete(prompt, system=SYSTEM_PROMPT).strip()
        except RuntimeError:
            return None
        if not reply or reply == SUFFICIENT_SENTINEL:
            return None
        return reply
