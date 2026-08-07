from __future__ import annotations

import time

import pytest

from src.cancellation import (
    Cancelled,
    check,
    interruptible_sleep,
    is_cancelled,
    reset_canceller,
    set_canceller,
)


def test_check_is_a_noop_with_no_canceller_installed() -> None:
    assert is_cancelled() is False
    check()


def test_check_raises_once_the_predicate_flips() -> None:
    flag = {"stop": False}
    token = set_canceller(lambda: flag["stop"])
    try:
        check()  # still live
        flag["stop"] = True
        with pytest.raises(Cancelled):
            check()
    finally:
        reset_canceller(token)


def test_cancelled_is_not_a_runtimeerror() -> None:
    """Every LLM stage catches RuntimeError to fall back to a template. Cancellation has
    to punch through those handlers rather than be swallowed as a recoverable failure."""
    assert not issubclass(Cancelled, RuntimeError)

    token = set_canceller(lambda: True)
    try:
        with pytest.raises(Cancelled):
            try:
                check()
            except RuntimeError:  # the shape used in summarizer/contextualizer/compressor
                pytest.fail("Cancelled was swallowed by an `except RuntimeError` handler")
    finally:
        reset_canceller(token)


def test_interruptible_sleep_abandons_the_wait_when_cancelled() -> None:
    """The rate-limit backoff is where the seconds go — a 65s sleep must not outlive the
    caller by 65 seconds."""
    deadline = time.monotonic() + 0.25
    token = set_canceller(lambda: time.monotonic() > deadline)
    started = time.monotonic()
    try:
        with pytest.raises(Cancelled):
            interruptible_sleep(30, tick=0.05)
    finally:
        reset_canceller(token)
    assert time.monotonic() - started < 2.0


def test_interruptible_sleep_runs_to_completion_when_not_cancelled() -> None:
    started = time.monotonic()
    interruptible_sleep(0.2, tick=0.05)
    assert time.monotonic() - started >= 0.2


def test_canceller_does_not_leak_between_requests() -> None:
    token = set_canceller(lambda: True)
    reset_canceller(token)
    assert is_cancelled() is False
