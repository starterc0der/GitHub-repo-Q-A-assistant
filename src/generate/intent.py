from __future__ import annotations

from src.llm_client import LLMClient

# Cheap, free, catches the obvious phrasings before anything pays for an LLM call.
BROAD_INTENT_KEYWORDS = (
    "all ", "all.", "all?", "every ", "each ", "entire", "complete list",
    "whole dataset", "whole list", "compare all", "across all", "every single",
)

CLASSIFIER_SYSTEM_PROMPT = (
    "Classify the question as SPECIFIC (a fact, a single item, or a small NAMED/BOUNDED "
    "set of items — e.g. \"Pandu's two other sons, Nakula and Sahadeva\" is still SPECIFIC, "
    "not BROAD, because it names exactly who) or BROAD (asks for open-ended coverage with "
    "no natural bound — every item, the whole dataset, comparing everything). Naming two or "
    "three specific things is SPECIFIC; only classify BROAD when nothing bounds the scope. "
    "Reply with exactly one word: SPECIFIC or BROAD."
)


def matches_broad_keywords(question: str) -> bool:
    lowered = question.lower()
    return any(kw in lowered for kw in BROAD_INTENT_KEYWORDS)


class BroadIntentClassifier:
    """Only reached when rerank found something AND keywords found nothing — the one
    genuinely ambiguous case worth a bulk-model call."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def is_broad(self, question: str) -> bool:
        try:
            reply = self.llm.complete(question, system=CLASSIFIER_SYSTEM_PROMPT).strip().upper()
        except RuntimeError:
            return False  # can't classify — trust the precise answer already in hand
        return reply.startswith("BROAD")
