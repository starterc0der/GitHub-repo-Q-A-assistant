from __future__ import annotations

from src.api.chat_routes import _windowed_history


def _turns(n: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for i in range(1, n + 1):
        rows.append(("user", f"question {i}"))
        rows.append(("assistant", f"answer {i}"))
    return rows


def test_short_chat_returns_raw_history_unchanged_no_fold() -> None:
    calls = []
    history, summary, folded = _windowed_history(
        _turns(3), turns=3, already_folded=0, existing_summary="",
        fold_fn=lambda s, t: calls.append(t) or "should not be called",
    )

    assert history == _turns(3)
    assert summary == ""
    assert folded == 0
    assert calls == []  # turn 4 (this call represents turns 1-3 already stored) — nothing to fold yet


def test_turn_5_folds_turn_1_into_summary() -> None:
    # 4 completed turns stored, asking a 5th — window keeps last 3 (turns 2-4), turn 1 ages out.
    folded_calls = []

    def fold_fn(existing, new_turns):
        folded_calls.append((existing, new_turns))
        return "summary of turn 1"

    history, summary, folded = _windowed_history(
        _turns(4), turns=3, already_folded=0, existing_summary="", fold_fn=fold_fn,
    )

    assert folded_calls == [("", [("user", "question 1"), ("assistant", "answer 1")])]
    assert summary == "summary of turn 1"
    assert folded == 1
    assert history == [("system", "summary of turn 1")] + _turns(4)[2:]


def test_turn_6_folds_turn_2_into_existing_summary_not_from_scratch() -> None:
    # Turn 1 already folded (already_folded=1); 5 completed turns stored, asking a 6th —
    # window keeps last 3 (turns 3-5), turn 2 ages out and gets added to the same summary.
    folded_calls = []

    def fold_fn(existing, new_turns):
        folded_calls.append((existing, new_turns))
        return "summary of turns 1-2"

    history, summary, folded = _windowed_history(
        _turns(5), turns=3, already_folded=1, existing_summary="summary of turn 1", fold_fn=fold_fn,
    )

    assert folded_calls == [("summary of turn 1", [("user", "question 2"), ("assistant", "answer 2")])]
    assert summary == "summary of turns 1-2"
    assert folded == 2
    assert history == [("system", "summary of turns 1-2")] + _turns(5)[4:]


def test_already_folded_turns_are_not_refolded() -> None:
    # Nothing new has aged out since the last call — fold_fn must not be called again.
    calls = []
    history, summary, folded = _windowed_history(
        _turns(4), turns=3, already_folded=1, existing_summary="summary of turn 1",
        fold_fn=lambda s, t: calls.append(t) or "unused",
    )

    assert calls == []
    assert summary == "summary of turn 1"
    assert folded == 1
    assert history == [("system", "summary of turn 1")] + _turns(4)[2:]
