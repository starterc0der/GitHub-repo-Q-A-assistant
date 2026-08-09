from __future__ import annotations

from src.generate.intent import BroadIntentClassifier, matches_broad_keywords


class FakeLLM:
    def __init__(self, response: str | None = None, fail: bool = False):
        self.response = response
        self.fail = fail

    def complete(self, prompt: str, system: str | None = None) -> str:
        if self.fail:
            raise RuntimeError("llm unreachable")
        return self.response


def test_matches_broad_keywords_catches_common_phrasings() -> None:
    assert matches_broad_keywords("compare all the cars engine cc with me visually")
    assert matches_broad_keywords("show every model in this dataset")
    assert matches_broad_keywords("give me the complete list of prices")


def test_matches_broad_keywords_ignores_specific_questions() -> None:
    assert not matches_broad_keywords("what is the engine cc of the Tata Nano GenX?")
    assert not matches_broad_keywords("does the S24 have a bigger battery than the S25?")


def test_is_broad_reads_llm_classification() -> None:
    classifier = BroadIntentClassifier(FakeLLM(response="BROAD"))
    assert classifier.is_broad("how does displacement vary across the dataset") is True

    classifier = BroadIntentClassifier(FakeLLM(response="SPECIFIC"))
    assert classifier.is_broad("what's the price of the iPhone 16") is False


def test_is_broad_defaults_to_false_on_llm_failure() -> None:
    """Can't classify right now — default to trusting the precise answer already in hand
    rather than paying for a wide read on a guess."""
    classifier = BroadIntentClassifier(FakeLLM(fail=True))
    assert classifier.is_broad("some question") is False
