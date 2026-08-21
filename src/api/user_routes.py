from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.insights_routes import (
    _cache_hit_by_day,
    _daily_buckets,
    _faithfulness_by_day,
    _gate_outcomes_by_day,
    _resolve_range,
    _tokens_by_day,
)
from src.auth import require_admin
from src.config import settings
from src.db import connect, now

router = APIRouter(dependencies=[Depends(require_admin)])


def _user_row(user_id: str) -> dict:
    with connect(settings.db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No user {user_id!r}")
    return dict(row)


@router.get("/users")
def list_users() -> dict[str, object]:
    with connect(settings.db_path) as conn:
        users = conn.execute("SELECT id, email, name, role, created_at FROM users ORDER BY created_at").fetchall()
        memberships = conn.execute(
            "SELECT sm.user_id, s.id AS space_id, s.name AS space_name, s.color "
            "FROM space_members sm JOIN spaces s ON s.id = sm.space_id"
        ).fetchall()
        last_active_rows = conn.execute(
            "SELECT user_id, MAX(created_at) AS last_active FROM messages WHERE user_id IS NOT NULL GROUP BY user_id"
        ).fetchall()
    spaces_by_user: dict[str, list[dict]] = {}
    for m in memberships:
        spaces_by_user.setdefault(m["user_id"], []).append(
            {"id": m["space_id"], "name": m["space_name"], "color": m["color"]}
        )
    last_active_by_user = {r["user_id"]: r["last_active"] for r in last_active_rows}
    return {
        "users": [
            {**dict(u), "spaces": spaces_by_user.get(u["id"], []), "last_active": last_active_by_user.get(u["id"])}
            for u in users
        ]
    }


class UpdateRoleRequest(BaseModel):
    role: str


@router.patch("/users/{user_id}")
def update_user_role(user_id: str, request: UpdateRoleRequest) -> dict[str, object]:
    if request.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'")
    _user_row(user_id)
    with connect(settings.db_path) as conn:
        # Bump token_version so a demoted/promoted user's existing token is rejected on
        # their very next request rather than staying valid until it expires.
        conn.execute(
            "UPDATE users SET role=?, token_version=token_version+1 WHERE id=?",
            (request.role, user_id),
        )
    return _user_row(user_id)


@router.post("/users/{user_id}/spaces/{space_id}")
def assign_space(user_id: str, space_id: str) -> dict[str, str]:
    _user_row(user_id)
    with connect(settings.db_path) as conn:
        space = conn.execute("SELECT id FROM spaces WHERE id=?", (space_id,)).fetchone()
        if space is None:
            raise HTTPException(status_code=404, detail=f"No space {space_id!r}")
        conn.execute(
            "INSERT OR IGNORE INTO space_members (space_id, user_id, created_at) VALUES (?, ?, ?)",
            (space_id, user_id, now()),
        )
    return {"status": "assigned"}


@router.delete("/users/{user_id}/spaces/{space_id}")
def unassign_space(user_id: str, space_id: str) -> dict[str, str]:
    _user_row(user_id)
    with connect(settings.db_path) as conn:
        conn.execute(
            "DELETE FROM space_members WHERE space_id=? AND user_id=?", (space_id, user_id)
        )
        # Revoke immediately — see update_user_role's comment.
        conn.execute("UPDATE users SET token_version=token_version+1 WHERE id=?", (user_id,))
    return {"status": "unassigned"}


def _user_assistant_messages(user_id: str) -> list[dict]:
    """Every assistant reply generated for this user's questions, trace parsed —
    mirrors insights_routes._space_assistant_messages but scoped by user_id instead of
    space_id, since a user's activity can span several spaces."""
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT m.*, c.space_id, s.name AS space_name FROM messages m "
            "JOIN chats c ON m.chat_id = c.id JOIN spaces s ON s.id = c.space_id "
            "WHERE m.user_id = ? AND m.role = 'assistant' ORDER BY m.created_at",
            (user_id,),
        ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["trace_obj"] = json.loads(d["trace"]) if d["trace"] else {}
        out.append(d)
    return out


def _user_turns(user_id: str, limit: int, offset: int) -> tuple[list[dict], int]:
    """Recent question/answer turns for this user — the question text (from the user
    row) paired with its reply's tokens/cache_hit (from the following assistant row in
    the same chat), newest first."""
    with connect(settings.db_path) as conn:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE role='user' AND user_id=?", (user_id,)
        ).fetchone()["n"]
        rows = conn.execute(
            "SELECT u.content AS question, u.created_at AS created_at, u.chat_id AS chat_id, "
            "c.space_id AS space_id, s.name AS space_name, a.trace AS trace, a.cache_hit AS cache_hit "
            "FROM messages u "
            "JOIN messages a ON a.chat_id = u.chat_id AND a.seq = u.seq + 1 AND a.role = 'assistant' "
            "JOIN chats c ON c.id = u.chat_id "
            "JOIN spaces s ON s.id = c.space_id "
            "WHERE u.role = 'user' AND u.user_id = ? "
            "ORDER BY u.created_at DESC LIMIT ? OFFSET ?",
            (user_id, limit, offset),
        ).fetchall()
    turns = []
    for row in rows:
        trace_obj = json.loads(row["trace"]) if row["trace"] else {}
        tokens = trace_obj.get("tokens") or {}
        turns.append({
            "question": row["question"], "created_at": row["created_at"],
            "chat_id": row["chat_id"], "space_id": row["space_id"], "space_name": row["space_name"],
            "prompt_tokens": tokens.get("prompt_tokens", 0),
            "completion_tokens": tokens.get("completion_tokens", 0),
            "cache_hit": bool(row["cache_hit"]),
        })
    return turns, total


@router.get("/users/{user_id}/insights")
def user_insights(
    user_id: str, start: str | None = None, end: str | None = None,
    turns_limit: int = 20, turns_offset: int = 0,
) -> dict[str, object]:
    user = _user_row(user_id)
    all_messages = _user_assistant_messages(user_id)
    range_start, range_end, range_min_date, range_max_date = _resolve_range(all_messages, start, end)
    messages = [m for m in all_messages if range_start <= m["created_at"][:10] <= range_end]

    total = len(messages)
    cache_hits = sum(1 for m in messages if m["cache_hit"])
    today = datetime.now(UTC).date().isoformat()
    this_month = today[:7]

    def _sum_tokens(msgs: list[dict]) -> int:
        result = 0
        for m in msgs:
            tokens = m["trace_obj"].get("tokens") or {}
            result += tokens.get("prompt_tokens", 0) + tokens.get("completion_tokens", 0)
        return result

    total_tokens = _sum_tokens(messages)
    tokens_today = _sum_tokens([m for m in messages if m["created_at"][:10] == today])
    tokens_this_month = _sum_tokens([m for m in messages if m["created_at"][:7] == this_month])

    by_space: dict[str, dict] = {}
    for m in messages:
        bucket = by_space.setdefault(m["space_id"], {"space_id": m["space_id"], "space_name": m["space_name"], "question_count": 0, "tokens": 0})
        bucket["question_count"] += 1
        bucket["tokens"] += _sum_tokens([m])

    daily = _daily_buckets(messages, range_start, range_end)
    turns, turns_total = _user_turns(user_id, turns_limit, turns_offset)

    with connect(settings.db_path) as conn:
        assigned_spaces = [
            dict(r) for r in conn.execute(
                "SELECT s.id AS space_id, s.name AS space_name, s.color FROM space_members sm "
                "JOIN spaces s ON s.id = sm.space_id WHERE sm.user_id=? ORDER BY s.name",
                (user_id,),
            ).fetchall()
        ]

    return {
        "user": {"id": user["id"], "email": user["email"], "name": user["name"], "role": user["role"], "created_at": user["created_at"]},
        "question_count": total,
        "total_tokens": total_tokens,
        "tokens_today": tokens_today,
        "tokens_this_month": tokens_this_month,
        "avg_tokens_per_question": (total_tokens / total) if total else 0.0,
        "cache_hit_rate": (cache_hits / total) if total else 0.0,
        "tokens_by_day": _tokens_by_day(daily),
        "questions_by_day": [{"date": b["date"], "total": b["total"]} for b in daily],
        "cache_hit_by_day": _cache_hit_by_day(daily),
        "gate_outcomes_by_day": _gate_outcomes_by_day(daily),
        "faithfulness_by_day": _faithfulness_by_day(daily),
        "by_space": sorted(by_space.values(), key=lambda b: b["tokens"], reverse=True),
        "assigned_spaces": assigned_spaces,
        "turns": turns,
        "turns_total": turns_total,
        "range_start": range_start,
        "range_end": range_end,
        "range_min_date": range_min_date,
        "range_max_date": range_max_date,
    }
