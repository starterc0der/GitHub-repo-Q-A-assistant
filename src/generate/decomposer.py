from __future__ import annotations

from src.llm_client import LLMClient

SINGLE_SENTINEL = "SINGLE"

# Cheap, free, catches the common phrasings before anything pays for an LLM call — mirrors
# BroadIntentClassifier's keyword-first gate in intent.py.
COMPOUND_HINTS = (" and ", " & ", "; ")

DECOMPOSE_SYSTEM_PROMPT = (
    "Decide whether the question below is actually multiple independent questions bundled "
    "into one (different topics joined by \"and\", or a list of separate asks) rather than "
    f"one genuine question. If it is genuinely one question, reply with exactly: {SINGLE_SENTINEL}\n"
    "Otherwise, split it into at most 3 independent, self-contained sub-questions, one per "
    "line, each understandable on its own — resolve pronouns like \"it\"/\"that\" against the "
    "original question. Do not answer any of them, do not add numbering or preamble. Reply "
    "with only the sub-questions, one per line."
)


def looks_compound(question: str) -> bool:
    lowered = question.lower()
    return lowered.count("?") > 1 or any(hint in lowered for hint in COMPOUND_HINTS)


RETRY_REWRITE_SYSTEM_PROMPT = (
    "A sub-question from a larger question found no supporting evidence in the source "
    "material. Rewrite it as a differently-phrased, self-contained question aimed at the "
    "same underlying information need — the original phrasing may just not match how the "
    "source text describes it (e.g. a comparison sub-question often finds more by asking "
    "directly about the other side of the comparison instead of the comparison itself). "
    "Reply with only the rewritten question, nothing else."
)


class QueryDecomposer:
    """Splits a compound question into independent sub-questions before retrieval — a
    single embedding of "what does X do, and how is that different from Y?" is a blurry
    average of both topics, which hurts routing and hybrid search for either one.

    decompose() always returns a non-empty list: [question] unchanged when it isn't
    compound, on LLM failure, or on turn 1 of a simple question — so callers never need
    to special-case "not decomposed", they just loop over however many came back."""

    def __init__(self, llm: LLMClient, max_subquestions: int):
        self.llm = llm
        self.max_subquestions = max_subquestions

    def decompose(self, question: str) -> list[str]:
        if not looks_compound(question):
            return [question]
        try:
            reply = self.llm.complete(question, system=DECOMPOSE_SYSTEM_PROMPT).strip()
        except RuntimeError:
            return [question]
        if not reply or reply == SINGLE_SENTINEL:
            return [question]
        parts = [line.strip("-•* \t") for line in reply.splitlines()]
        parts = [p for p in parts if p][: self.max_subquestions]
        return parts or [question]

    def rewrite_for_retry(self, sub_question: str, original_question: str) -> str:
        """Called once per sub-question that came back with no evidence — see
        Pipeline._retrieve's retry pass. Returns sub_question unchanged on any failure
        so the caller can treat "unchanged" as "nothing to retry"."""
        prompt = f"Original question: {original_question}\nSub-question with no evidence: {sub_question}"
        try:
            reply = self.llm.complete(prompt, system=RETRY_REWRITE_SYSTEM_PROMPT).strip()
        except RuntimeError:
            return sub_question
        return reply or sub_question
