from __future__ import annotations

from src.llm_client import LLMClient

# Cheap, free, catches the obvious phrasings before anything pays for an LLM call.
BROAD_INTENT_KEYWORDS = (
    "all ", "all.", "all?", "every ", "each ", "entire", "complete list",
    "whole dataset", "whole list", "compare all", "across all", "every single",
)

CLASSIFIER_SYSTEM_PROMPT = (
    "Classify the question as asking about ONE specific thing (a fact, a single item, a "
    "narrow lookup) or BROAD coverage across many items (comparing everything, summarizing "
    "the whole dataset, every entry). Reply with exactly one word: SPECIFIC or BROAD."
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
