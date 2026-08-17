from __future__ import annotations

from src.api.chat_routes import _windowed_history


def _turns(n: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for i in range(1, n + 1):
        rows.append(("user", f"question {i}"))
        rows.append(("assistant", f"answer {i}"))
    return rows


def _system_text(history: list[tuple[str, str]]) -> str:
    assert history[0][0] == "system"
    return history[0][1]


def test_short_chat_returns_raw_history_unchanged_no_fold() -> None:
    calls = []
    history, facts, folded = _windowed_history(
        _turns(3), turns=3, already_folded=0, existing_facts=[],
        extract_fn=lambda q, a: calls.append((q, a)) or "should not be called",
    )

    assert history == _turns(3)
    assert facts == []
    assert folded == 0
    assert calls == []  # turn 4 (this call represents turns 1-3 already stored) — nothing to fold yet


def test_turn_5_extracts_turn_1_as_a_permanent_fact() -> None:
    # 4 completed turns stored, asking a 5th — window keeps last 3 (turns 2-4), turn 1 ages out.
    calls = []

    def extract_fn(question, answer):
        calls.append((question, answer))
        return "fact about turn 1"

    history, facts, folded = _windowed_history(
        _turns(4), turns=3, already_folded=0, existing_facts=[], extract_fn=extract_fn,
    )

    assert calls == [("question 1", "answer 1")]
    assert facts == [{"turn": 1, "fact": "fact about turn 1"}]
    assert folded == 1
    assert "chronological order" in _system_text(history)
    assert "Turn 1: fact about turn 1" in _system_text(history)
    assert history[1:] == _turns(4)[2:]


def test_turn_6_appends_turn_2_without_touching_turn_1s_fact() -> None:
    # Turn 1 already extracted (already_folded=1); 5 completed turns stored, asking a 6th —
    # window keeps last 3 (turns 3-5), turn 2 ages out and gets appended, turn 1 untouched.
    calls = []

    def extract_fn(question, answer):
        calls.append((question, answer))
        return "fact about turn 2"

    history, facts, folded = _windowed_history(
        _turns(5), turns=3, already_folded=1,
        existing_facts=[{"turn": 1, "fact": "fact about turn 1"}], extract_fn=extract_fn,
    )

    assert calls == [("question 2", "answer 2")]  # turn 1 never re-passed to extract_fn
    assert facts == [
        {"turn": 1, "fact": "fact about turn 1"},
        {"turn": 2, "fact": "fact about turn 2"},
    ]
    assert folded == 2
    text = _system_text(history)
    assert "Turn 1: fact about turn 1" in text
    assert "Turn 2: fact about turn 2" in text
    assert text.index("Turn 1:") < text.index("Turn 2:")
    assert history[1:] == _turns(5)[4:]


def test_already_folded_turns_are_not_reextracted() -> None:
    # Nothing new has aged out since the last call — extract_fn must not be called again.
    calls = []
    history, facts, folded = _windowed_history(
        _turns(4), turns=3, already_folded=1,
        existing_facts=[{"turn": 1, "fact": "fact about turn 1"}],
        extract_fn=lambda q, a: calls.append((q, a)) or "unused",
    )

    assert calls == []
    assert facts == [{"turn": 1, "fact": "fact about turn 1"}]
    assert folded == 1
    assert "Turn 1: fact about turn 1" in _system_text(history)
    assert history[1:] == _turns(4)[2:]


def test_refusal_turn_extracts_nothing_but_still_counts_as_folded() -> None:
    # A refusal/no-answer turn (extract_fn returns None) contributes no fact, but must
    # still be marked folded — otherwise it would be re-extracted forever.
    history, facts, folded = _windowed_history(
        _turns(4), turns=3, already_folded=0, existing_facts=[], extract_fn=lambda q, a: None,
    )

    assert facts == []
    assert folded == 1
    assert history == _turns(4)[2:]  # no system message at all — nothing to say yet


def test_multiple_turns_aging_out_at_once_each_get_extracted_individually() -> None:
    # A gap since the last call (e.g. history_turns changed) — every newly-aged-out turn
    # gets its OWN extract_fn call, not one call covering all of them.
    calls = []

    def extract_fn(question, answer):
        calls.append((question, answer))
        return f"fact about {question}"

    history, facts, folded = _windowed_history(
        _turns(6), turns=2, already_folded=0, existing_facts=[], extract_fn=extract_fn,
    )

    assert calls == [("question 1", "answer 1"), ("question 2", "answer 2"), ("question 3", "answer 3"), ("question 4", "answer 4")]
    assert folded == 4
    assert facts == [
        {"turn": 1, "fact": "fact about question 1"},
        {"turn": 2, "fact": "fact about question 2"},
        {"turn": 3, "fact": "fact about question 3"},
        {"turn": 4, "fact": "fact about question 4"},
    ]
