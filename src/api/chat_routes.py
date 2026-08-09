from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
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


def _load_history(chat_id: str, turns: int) -> list[tuple[str, str]]:
    """Last N user+assistant pairs, oldest first — what the LLM sees for chat continuity."""
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id=? ORDER BY seq DESC LIMIT ?",
            (chat_id, turns * 2),
        ).fetchall()
    return [(r["role"], r["content"]) for r in reversed(rows)]


def _insert_message(
    chat_id: str, role: str, content: str, *,
    standalone_question: str | None = None,
    cache_hit: bool = False,
    cached_from: str | None = None,
    trace: str | None = None,
    chart: str | None = None,
) -> str:
    message_id = new_id()
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO messages "
            "(id, chat_id, seq, role, content, standalone_question, cache_hit, cached_from, trace, chart, created_at) "
            "SELECT ?, ?, COALESCE(MAX(seq), -1) + 1, ?, ?, ?, ?, ?, ?, ?, ? FROM messages WHERE chat_id=?",
            (
                message_id, chat_id, role, content, standalone_question,
                int(cache_hit), cached_from, trace, chart, now(), chat_id,
            ),
        )
    return message_id


def _cache_lookup(space_id: str, raw_question: str) -> dict[str, object] | None:
    """Exact-match only, and only ever populated from turn-1 (history-free) questions —
    see the module docstring below for why that's the correctness-load-bearing rule."""
    question_hash = _hash_question(raw_question)
    with connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT message_id FROM qa_cache WHERE space_id=? AND question_hash=?",
            (space_id, question_hash),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE qa_cache SET hit_count = hit_count + 1 WHERE space_id=? AND question_hash=?",
            (space_id, question_hash),
        )
        message = conn.execute("SELECT * FROM messages WHERE id=?", (row["message_id"],)).fetchone()
    return dict(message) if message else None


def _cache_put(space_id: str, raw_question: str, message_id: str) -> None:
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO qa_cache (space_id, question_hash, question, message_id, hit_count, created_at) "
            "VALUES (?, ?, ?, ?, 0, ?)",
            (space_id, _hash_question(raw_question), raw_question, message_id, now()),
        )


async def _run_cancellable(request: Request, work: Callable[[], object]) -> object:
    """Run blocking pipeline work in a threadpool, aborting it if the client goes away."""
    cancelled = threading.Event()

    async def watch_client() -> None:
        while not cancelled.is_set():
            if await request.is_disconnected():
                cancelled.set()
                return
            await asyncio.sleep(0.4)

    def runner() -> object:
        token = set_canceller(cancelled.is_set)
        try:
            return work()
        finally:
            reset_canceller(token)

    watcher = asyncio.create_task(watch_client())
    try:
        return await run_in_threadpool(runner)
    finally:
        cancelled.set()
        watcher.cancel()


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
    return row


@router.get("/chats/{chat_id}/messages")
def list_messages(chat_id: str) -> dict[str, object]:
    _get_chat_row(chat_id)
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT id, role, content, standalone_question, cache_hit, cached_from, chart, created_at, "
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


@router.post("/chats/{chat_id}/messages")
async def send_message(chat_id: str, request: Request, body: SendMessageRequest) -> Response:
    """The full turn: rewrite -> cache check -> retrieve -> compress -> answer -> persist.

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
    space_id = chat["space_id"]
    raw_question = body.content

    history = _load_history(chat_id, settings.history_turns)
    standalone = pipeline.rewrite_standalone(raw_question, history) if history else raw_question

    _insert_message(
        chat_id, "user", raw_question,
        standalone_question=standalone if history else None,
    )

    if not history:
        cached = _cache_lookup(space_id, raw_question)
        if cached is not None:
            assistant_id = _insert_message(
                chat_id, "assistant", cached["content"],
                cache_hit=True, cached_from=cached["id"], trace=cached["trace"],
                chart=cached["chart"],
            )
            _touch_chat(chat_id)
            return JSONResponse(_get_message(assistant_id))

    try:
        trace = await _run_cancellable(
            request,
            lambda: asdict(
                pipeline.query_trace(standalone, space_id, raw_question=raw_question, history=history)
            ),
        )
    except UnexpectedResponse as exc:
        if exc.status_code != 404:
            raise
        raise HTTPException(
            status_code=409, detail=f"Space {space_id!r} has no vectors in Qdrant yet."
        ) from exc
    except Cancelled:
        logger.info("Chat message cancelled by client disconnect: %r", raw_question)
        return Response(status_code=499)

    answer = trace.get("answer") or {}
    # answer.text is "" (not missing) on an LLM failure — `or` catches that; a plain
    # dict.get default would not, since the key is present.
    content = answer.get("text") or answer.get("error") or pipeline.NO_MATCH
    chart = answer.get("chart")
    assistant_id = _insert_message(
        chat_id, "assistant", content, trace=json.dumps(trace),
        chart=json.dumps(chart) if chart else None,
    )

    if not history:
        _cache_put(space_id, raw_question, assistant_id)
    _set_title_if_new(chat_id, raw_question)
    _touch_chat(chat_id)

    return JSONResponse(_get_message(assistant_id))


def _get_message(message_id: str) -> dict[str, object]:
    with connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT id, role, content, standalone_question, cache_hit, cached_from, chart, created_at, "
            "(trace IS NOT NULL) AS has_trace FROM messages WHERE id=?",
            (message_id,),
        ).fetchone()
    return _parse_chart(dict(row))
