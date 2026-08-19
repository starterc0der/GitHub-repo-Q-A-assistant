from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client.http.exceptions import UnexpectedResponse

from src.api.routes import _space_row, pipeline
from src.cancellation import Cancelled, reset_canceller, set_canceller
from src.config import settings
from src.db import connect, new_id, now

logger = logging.getLogger(__name__)

router = APIRouter()


def _hash_question(question: str) -> str:
    normalized = " ".join(question.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _get_chat_row(chat_id: str) -> dict[str, object]:
    with connect(settings.db_path) as conn:
        row = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No chat {chat_id!r}")
    return dict(row)


def _touch_chat(chat_id: str) -> None:
    with connect(settings.db_path) as conn:
        conn.execute("UPDATE chats SET updated_at=? WHERE id=?", (now(), chat_id))


def _set_title_if_new(chat_id: str, title: str) -> None:
    with connect(settings.db_path) as conn:
        conn.execute(
            "UPDATE chats SET title=? WHERE id=? AND title='New chat'", (title[:60], chat_id)
        )


def _windowed_history(
    rows: list[tuple[str, str]], turns: int, already_folded: int, existing_facts: list[dict],
    extract_fn,
) -> tuple[list[tuple[str, str]], list[dict], int]:
    """rows: every (role, content) message in the chat, oldest first, two per completed
    turn. Turns 1..turns pay nothing extra — the raw window still holds everything. Past
    that, each new turn ages exactly one older turn out of the window; extract_fn pulls
    one atomic fact from it (or None, for a refusal/no-answer turn) and the fact is
    appended PERMANENTLY as {"turn": N, "fact": ...} — never rewritten by a later call.
    That's the load-bearing difference from the old whole-summary refold: re-summarizing
    the same running text on every fold caused early facts to erode over many turns
    (verified live — two of the earliest facts in a 15-turn test were gone by the end,
    and outside knowledge started leaking in). An atomic fact written once and left
    untouched can't drift.

    Returns the history to send — the facts as one "Turn N: fact" system message,
    followed by the last `turns` raw turns, never the unbounded full transcript — plus
    the updated fact list and folded-turn count. Pure function — the caller owns DB
    reads/writes and threading the (possibly blocking) extract_fn off the event loop."""
    total_turns = len(rows) // 2
    if total_turns <= turns:
        return list(rows), existing_facts, already_folded
    fold_through = total_turns - turns
    facts = list(existing_facts)
    for turn_idx in range(already_folded, fold_through):
        question, answer = rows[turn_idx * 2][1], rows[turn_idx * 2 + 1][1]
        fact = extract_fn(question, answer)
        if fact is not None:
            facts.append({"turn": turn_idx + 1, "fact": fact})
    window = rows[fold_through * 2 :]
    # The turn numbers alone don't tell a reader they're chronological — without this
    # framing line, "what was the first thing asked" gets answered from whichever fact
    # seems most salient rather than the lowest turn number (seen live: a turn-13 fact
    # picked over the real turn-1 one). Stated once here so every consumer of this system
    # message (meta answers, the rewrite/router call, generation) sees the same claim,
    # instead of repeating an ordering instruction in each of their own prompts.
    facts_text = "\n".join(f"Turn {f['turn']}: {f['fact']}" for f in facts)
    summary_text = (
        f"Earlier facts from this conversation, in chronological order (Turn 1 is the "
        f"earliest, higher numbers are more recent):\n{facts_text}"
    )
    history = ([("system", summary_text)] if facts else []) + window
    return history, facts, fold_through


def _load_history(chat_id: str, turns: int) -> list[tuple[str, str]]:
    """Recent raw turns plus the accumulated fact list for everything older — see
    _windowed_history. Used for both retrieval-rewrite/generation continuity AND meta/recap
    answers alike: one mechanism, never the fixed-N-only window and never the whole
    unbounded chat."""
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id=? ORDER BY seq", (chat_id,)
        ).fetchall()
        chat = conn.execute(
            "SELECT memory_summary, memory_folded_turns FROM chats WHERE id=?", (chat_id,)
        ).fetchone()
    all_turns = [(r["role"], r["content"]) for r in rows]
    try:
        existing_facts = json.loads(chat["memory_summary"]) if chat["memory_summary"] else []
    except (json.JSONDecodeError, TypeError):
        existing_facts = []  # pre-existing plain-prose summary from before this format — start fresh
    history, facts, folded = _windowed_history(
        all_turns, turns, chat["memory_folded_turns"], existing_facts,
        pipeline.conversation_memory.extract,
    )
    if folded != chat["memory_folded_turns"]:
        with connect(settings.db_path) as conn:
            conn.execute(
                "UPDATE chats SET memory_summary=?, memory_folded_turns=? WHERE id=?",
                (json.dumps(facts), folded, chat_id),
            )
    return history


def _insert_message(
    chat_id: str, role: str, content: str, *,
    standalone_question: str | None = None,
    cache_hit: bool = False,
    cached_from: str | None = None,
    cache_kind: str | None = None,
    cache_match_question: str | None = None,
    cache_match_score: float | None = None,
    trace: str | None = None,
    chart: str | None = None,
    table: str | None = None,
) -> str:
    message_id = new_id()
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO messages "
            "(id, chat_id, seq, role, content, standalone_question, cache_hit, cached_from, "
            "cache_kind, cache_match_question, cache_match_score, trace, chart, table_data, created_at) "
            "SELECT ?, ?, COALESCE(MAX(seq), -1) + 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? FROM messages WHERE chat_id=?",
            (
                message_id, chat_id, role, content, standalone_question,
                int(cache_hit), cached_from, cache_kind, cache_match_question, cache_match_score,
                trace, chart, table, now(), chat_id,
            ),
        )
    return message_id


def _cache_lookup(space_id: str, raw_question: str) -> dict[str, object] | None:
    """Exact-match first (free, zero wrong-answer risk) — only ever populated from turn-1
    (history-free) questions, see the module docstring below for why that's the
    correctness-load-bearing rule. On a miss, falls back to the semantic cache
    (Pipeline.semantic_cache_lookup): a paraphrase of an already-answered question
    ("tell me about Bhishma" after "who is Bhishma") embeds close to it even though the
    text hashes differently — exact-match structurally can't catch that."""
    question_hash = _hash_question(raw_question)
    with connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT message_id FROM qa_cache WHERE space_id=? AND question_hash=?",
            (space_id, question_hash),
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE qa_cache SET hit_count = hit_count + 1 WHERE space_id=? AND question_hash=?",
                (space_id, question_hash),
            )
            message = conn.execute("SELECT * FROM messages WHERE id=?", (row["message_id"],)).fetchone()
            if message is not None:
                logger.info("CACHE HIT (exact) | space=%s | question=%r", space_id, raw_question)
                return {**dict(message), "cache_kind": "exact", "cache_match_question": None, "cache_match_score": None}

    semantic = pipeline.semantic_cache_lookup(space_id, raw_question)
    if semantic is None:
        return None
    message_id, matched_question, score = semantic
    with connect(settings.db_path) as conn:
        message = conn.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
    if message is None:
        return None
    logger.info(
        "CACHE HIT (semantic) | space=%s | question=%r | matched=%r | score=%.3f",
        space_id, raw_question, matched_question, score,
    )
    return {**dict(message), "cache_kind": "semantic", "cache_match_question": matched_question, "cache_match_score": score}


def _cache_put(space_id: str, raw_question: str, message_id: str) -> None:
    question_hash = _hash_question(raw_question)
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO qa_cache (space_id, question_hash, question, message_id, hit_count, created_at) "
            "VALUES (?, ?, ?, ?, 0, ?)",
            (space_id, question_hash, raw_question, message_id, now()),
        )
    pipeline.semantic_cache_put(space_id, question_hash, raw_question, message_id)


def _sse(event: dict) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode()


class _StreamDone:
    """Wraps query_trace_stream()'s return value (the finished QueryTrace). A bare
    StopIteration can't cross the `await run_in_threadpool(...)` boundary: CPython
    converts a StopIteration escaping any coroutine frame into a RuntimeError
    ("coroutine raised StopIteration"), so it never reaches a `except StopIteration`
    on the async side. _advance() below catches it purely on the sync side instead."""

    __slots__ = ("value",)

    def __init__(self, value: object):
        self.value = value


def _advance(gen):
    try:
        return next(gen)
    except StopIteration as stop:
        return _StreamDone(stop.value)


class CreateChatRequest(BaseModel):
    title: str = "New chat"


class SendMessageRequest(BaseModel):
    content: str


@router.post("/spaces/{space_id}/chats")
def create_chat(space_id: str, request: CreateChatRequest = CreateChatRequest()) -> dict[str, object]:
    _space_row(space_id)
    chat_id = new_id()
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO chats (id, space_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (chat_id, space_id, request.title, now(), now()),
        )
    return _get_chat_row(chat_id)


@router.get("/spaces/{space_id}/chats")
def list_chats(space_id: str) -> dict[str, object]:
    _space_row(space_id)
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM chats WHERE space_id=? ORDER BY updated_at DESC", (space_id,)
        ).fetchall()
    return {"chats": [dict(r) for r in rows]}


@router.delete("/chats/{chat_id}")
def delete_chat(chat_id: str) -> dict[str, str]:
    _get_chat_row(chat_id)
    with connect(settings.db_path) as conn:
        conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    return {"status": "deleted"}


def _parse_chart(row: dict[str, object]) -> dict[str, object]:
    row["chart"] = json.loads(row["chart"]) if row.get("chart") else None
    row["table"] = json.loads(row["table"]) if row.get("table") else None
    return row


@router.get("/chats/{chat_id}/messages")
def list_messages(chat_id: str) -> dict[str, object]:
    _get_chat_row(chat_id)
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT id, role, content, standalone_question, cache_hit, cached_from, cache_kind, cache_match_question, cache_match_score, chart, table_data AS \"table\", created_at, "
            "(trace IS NOT NULL) AS has_trace FROM messages WHERE chat_id=? ORDER BY seq",
            (chat_id,),
        ).fetchall()
    return {"messages": [_parse_chart(dict(r)) for r in rows]}


@router.get("/messages/{message_id}/trace")
def message_trace(message_id: str) -> dict[str, object]:
    with connect(settings.db_path) as conn:
        row = conn.execute("SELECT trace FROM messages WHERE id=?", (message_id,)).fetchone()
    if row is None or row["trace"] is None:
        raise HTTPException(status_code=404, detail="No trace for this message")
    return json.loads(row["trace"])


@router.get("/messages/{message_id}/vectors")
def message_vectors(message_id: str) -> dict[str, object]:
    """The vector-space PCA data, computed on demand instead of on every message send —
    see Pipeline.vectors_trace(). Re-embeds the same question query_trace() embedded
    (embedding is deterministic), scoped to the message's own space."""
    with connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT trace, chat_id FROM messages WHERE id=?", (message_id,)
        ).fetchone()
    if row is None or row["trace"] is None:
        raise HTTPException(status_code=404, detail="No trace for this message")
    trace = json.loads(row["trace"])
    if trace.get("meta"):
        raise HTTPException(status_code=404, detail="No vector space for a conversation-only answer")
    space_id = _get_chat_row(row["chat_id"])["space_id"]
    try:
        vectors = pipeline.vectors_trace(trace["question"], space_id)
    except UnexpectedResponse as exc:
        if exc.status_code != 404:
            raise
        raise HTTPException(
            status_code=409, detail=f"Space {space_id!r} has no vectors in Qdrant yet."
        ) from exc
    return asdict(vectors)


@router.post("/chats/{chat_id}/messages")
async def send_message(chat_id: str, request: Request, body: SendMessageRequest) -> StreamingResponse:
    """The full turn, streamed: rewrite -> cache check -> retrieve -> compress -> answer
    (streamed token-by-token) -> persist.

    Every path — meta-answer, cache hit, real generation, error — emits the same SSE
    shape: zero or more {"type":"delta"} events, then exactly one {"type":"done"} or
    {"type":"error"} event. One response shape for the frontend to handle, regardless of
    which internal path produced the answer.

    Two things make this correct rather than just fast:
    - Retrieval embeds the REWRITTEN standalone question; generation answers the user's
      RAW words plus real chat history — conflating the two is the classic bug here.
    - The cache is only written to (and only checked) on turn 1, i.e. when history is
      empty. The rewriter is a stochastic LLM call, so keying on its output gives a near-
      zero hit rate; worse, an answer generated under one chat's history could otherwise
      leak into an unrelated chat that never had that context. Keying on the raw,
      pre-rewrite question only when there's no history makes every cached entry
      context-free by construction.
    """
    chat = _get_chat_row(chat_id)
    return StreamingResponse(
        _send_message_events(chat_id, chat["space_id"], body.content, request),
        media_type="text/event-stream",
    )


async def _send_message_events(
    chat_id: str, space_id: str, raw_question: str, request: Request
) -> AsyncIterator[bytes]:
    """Owns cancellation for the whole turn — one disconnect watcher for the entire
    stream, rather than one per blocking step (each run_in_threadpool call below inherits
    the ContextVar canceller installed here, same mechanism _run_cancellable used, just
    scoped to the request instead of to each call)."""
    cancelled = threading.Event()

    async def watch_client() -> None:
        while not cancelled.is_set():
            if await request.is_disconnected():
                cancelled.set()
                return
            await asyncio.sleep(0.4)

    token = set_canceller(cancelled.is_set)
    watcher = asyncio.create_task(watch_client())
    try:
        async for event in _run_turn(chat_id, space_id, raw_question):
            yield event
    except Cancelled:
        logger.info("Chat message cancelled by client disconnect: %r", raw_question)
    except UnexpectedResponse as exc:
        if exc.status_code != 404:
            raise
        yield _sse({"type": "error", "message": f"Space {space_id!r} has no vectors in Qdrant yet."})
    except RuntimeError as exc:
        yield _sse({"type": "error", "message": str(exc)})
    finally:
        cancelled.set()
        watcher.cancel()
        reset_canceller(token)


async def _run_turn(chat_id: str, space_id: str, raw_question: str) -> AsyncIterator[bytes]:
    # Loaded before the insert below, so the question being asked right now isn't
    # duplicated into its own history.
    history = await run_in_threadpool(_load_history, chat_id, settings.history_turns)
    route, route_ms, route_tokens = await run_in_threadpool(pipeline.route_question, raw_question, history)

    # A question about the conversation/assistant itself ("what have we talked about",
    # "hi", "what can you do") rather than the document — the route call above already
    # classified this as a side effect (see Pipeline.route_question), so skip retrieval
    # and answer straight from history.
    if route.is_meta:
        _insert_message(chat_id, "user", raw_question)
        content = await run_in_threadpool(pipeline.answer_meta, raw_question, history)
        yield _sse({"type": "delta", "text": content})
        # No retrieval stages ran, so this isn't a QueryTrace — just the actual
        # transcript the model saw plus what it answered, so "why did it say that" is
        # still answerable (content also lives in the message row, but the trace should
        # be self-contained like every other trace shape, not require a second lookup).
        # "answer_text", not "answer" — a real QueryTrace's "answer" is a nested
        # {text, model, ...} object (see AnswerTrace); insights_routes.py's
        # `trace.get("answer") or {}` runs on every trace regardless of shape, and a
        # plain string there would crash the very next `.get()` call on it.
        meta_trace = json.dumps({
            "meta": True,
            "question": raw_question,
            "history": [{"role": role, "content": text} for role, text in history],
            "answer_text": content,
        })
        assistant_id = _insert_message(chat_id, "assistant", content, trace=meta_trace)
        _touch_chat(chat_id)
        yield _sse({"type": "done", "message": _get_message(assistant_id)})
        return

    # For parallel/sequential mode there's no single "resolved compound question" string
    # — the route call already split it into sub_questions directly — so the trace's
    # primary question field just falls back to the user's raw words, same as turn 1.
    standalone = route.sub_questions[0] if route.mode == "single" else raw_question
    _insert_message(
        chat_id, "user", raw_question,
        standalone_question=standalone if history and route.mode == "single" else None,
    )

    cache_ms: float | None = None
    # A live-data question's answer is a live sensor reading, not a fact — caching it
    # under the exact question text would serve a stale reading forever on any repeat of
    # the same wording, since the cache has no way to know the underlying data changed
    # since. Skip lookup too, not just the write: an entry cached before this guard
    # existed (or an old build) must not keep getting served. A report question is
    # skipped for the same reason — even a "yesterday" window can touch a still-filling
    # daily/hourly/5min row (see ReportWindowResolver), so it's simplest to never cache
    # rather than track whether a given window happens to be fully closed.
    if not history and not route.wants_live_data and not route.wants_report:
        cache_started = time.monotonic()
        cached = _cache_lookup(space_id, raw_question)
        cache_ms = (time.monotonic() - cache_started) * 1000
        if cached is not None:
            # This turn did a cache lookup, nothing else — its own timings/tokens must
            # say that, not silently carry the ORIGINAL run's retrieval/LLM costs as if
            # they happened again just now.
            cached_trace = json.loads(cached["trace"]) if cached["trace"] else {}
            cached_trace["timings"] = {"cache": cache_ms}
            cached_trace["tokens"] = {}
            assistant_id = _insert_message(
                chat_id, "assistant", cached["content"],
                cache_hit=True, cached_from=cached["id"], trace=json.dumps(cached_trace),
                chart=cached["chart"], table=cached["table_data"], cache_kind=cached["cache_kind"],
                cache_match_question=cached["cache_match_question"],
                cache_match_score=cached["cache_match_score"],
            )
            _set_title_if_new(chat_id, raw_question)
            _touch_chat(chat_id)
            yield _sse({"type": "delta", "text": cached["content"]})
            yield _sse({"type": "done", "message": _get_message(assistant_id)})
            return

    gen = pipeline.query_trace_stream(
        standalone, space_id, raw_question=raw_question, history=history,
        decompose_result=route, decompose_ms=route_ms, decompose_tokens=route_tokens,
    )
    trace = None
    while True:
        result = await run_in_threadpool(_advance, gen)
        if isinstance(result, _StreamDone):
            trace = result.value
            break
        yield _sse({"type": "delta", "text": result})

    if cache_ms is not None:
        trace.timings["cache"] = cache_ms
    trace_dict = asdict(trace)
    answer = trace_dict.get("answer") or {}
    chart = answer.get("chart")
    table = answer.get("table")
    # answer.text is legitimately "" (not missing) in two different cases that must not
    # be confused: an LLM failure (falls through to error, then NO_MATCH) — or a
    # live-data reply that's purely a ```table block with no surrounding prose, which is
    # a real, valid answer, not a failure. NO_MATCH is only right when there's neither
    # text NOR a chart/table to show.
    content = answer.get("text") or answer.get("error") or ("" if (chart or table) else pipeline.NO_MATCH)
    assistant_id = _insert_message(
        chat_id, "assistant", content, trace=json.dumps(trace_dict),
        chart=json.dumps(chart) if chart else None,
        table=json.dumps(table) if table else None,
    )

    if not history and not route.wants_live_data and not route.wants_report:
        _cache_put(space_id, raw_question, assistant_id)
    _set_title_if_new(chat_id, raw_question)
    _touch_chat(chat_id)
    yield _sse({"type": "done", "message": _get_message(assistant_id)})


def _get_message(message_id: str) -> dict[str, object]:
    with connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT id, role, content, standalone_question, cache_hit, cached_from, cache_kind, cache_match_question, cache_match_score, chart, table_data AS \"table\", created_at, "
            "(trace IS NOT NULL) AS has_trace FROM messages WHERE id=?",
            (message_id,),
        ).fetchone()
    return _parse_chart(dict(row))
