"""Cooperative cancellation: threads can't be killed, so work checks `check()` at safe
points and bails. The predicate rides a ContextVar (copied into threadpool workers) so
the four layers between the route and `complete()` don't need to know about HTTP.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextvars import ContextVar, Token

_canceller: ContextVar[Callable[[], bool] | None] = ContextVar("canceller", default=None)


class Cancelled(Exception):
    """Not a RuntimeError on purpose: the LLM stages catch that to fall back to
    templates, and cancellation must punch through rather than look recoverable."""


def set_canceller(predicate: Callable[[], bool]) -> Token:
    """Install a predicate that returns True once the work should stop."""
    return _canceller.set(predicate)


def reset_canceller(token: Token) -> None:
    _canceller.reset(token)


def is_cancelled() -> bool:
    predicate = _canceller.get()
    return bool(predicate and predicate())


def check() -> None:
    """Raise if the work has been cancelled. Call before anything expensive."""
    if is_cancelled():
        raise Cancelled("caller disconnected")


def interruptible_sleep(seconds: float, tick: float = 0.4) -> None:
    """time.sleep that abandons the wait on disconnect — a 65s rate-limit backoff would
    otherwise hold the quota slot long after the browser closed."""
    deadline = time.monotonic() + seconds
    while True:
        check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(tick, remaining))
