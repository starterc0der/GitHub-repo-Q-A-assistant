from __future__ import annotations

import json

from fastapi import APIRouter

from src.api.chat_routes import _get_chat_row
from src.api.routes import _space_row
from src.config import settings
from src.db import connect
from src.pipeline import Pipeline

router = APIRouter()

# Which pipeline stages ever make a real LLM/API call, vs. always-local computation.
# cache/embed/route/hybrid/rerank use only local models or plain lookups — never the
# chat LLM. decompose and gate usually skip the LLM (a cheap pre-filter decides most
# questions don't need it) but genuinely can call it, so they're flagged True rather
# than hidden as free. compress/generate always call it, when reached at all.
STAGE_IS_API = {
    "cache": False, "decompose": True, "embed": False, "route": False,
    "hybrid": False, "rerank": False, "gate": True, "retry": True, "compress": True,
    "generate": True,
}
STAGE_ORDER = list(STAGE_IS_API)


def _space_assistant_messages(space_id: str) -> list[dict]:
    """Every assistant message across every chat in this space, trace parsed. The
    common input every insights query aggregates over."""
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT m.* FROM messages m JOIN chats c ON m.chat_id = c.id "
            "WHERE c.space_id = ? AND m.role = 'assistant' ORDER BY m.created_at",
            (space_id,),
        ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["trace_obj"] = json.loads(d["trace"]) if d["trace"] else {}
        out.append(d)
    return out


def _gate_outcome(msg: dict) -> str:
    trace = msg["trace_obj"]
    if trace.get("meta"):
        return "answered"  # a conversation-only reply, not a retrieval gate outcome
    answer = trace.get("answer") or {}
    if answer.get("text") == Pipeline.NO_MATCH:
        return "no-match"
    if trace.get("wide_fallback"):
        return "wide-fallback"
    # A decomposed question where some (not all) sub-questions found no evidence — a
    # real answer was generated, but it's not a full success like a plain "answered",
    # so it must not get silently folded into that bucket. See Pipeline._retrieve.
    if trace.get("sufficiency") == "partial":
        return "partial"
    return "answered"


def _total_latency_ms(trace: dict) -> float:
    return sum((trace.get("timings") or {}).values())


GATE_KEYS = ("answered", "no-match", "wide-fallback", "partial")


def _empty_gate_counts() -> dict[str, int]:
    return {k: 0 for k in GATE_KEYS}


def _gate_outcomes_by_day(messages: list[dict], days: int = 14) -> list[dict]:
    """Real day-by-day history from actual message timestamps — not fabricated. A day
    with no questions reports null percentages (no data), not 0%, since those mean
    different things: "nothing asked" vs. "asked, and none were gated"."""
    from datetime import UTC, datetime, timedelta

    by_day: dict[str, dict[str, int]] = {}
    for m in messages:
        day = m["created_at"][:10]
        bucket = by_day.setdefault(day, _empty_gate_counts())
        bucket[_gate_outcome(m)] += 1

    today = datetime.now(UTC).date()
    out = []
    for i in range(days - 1, -1, -1):
        day = (today - timedelta(days=i)).isoformat()
        counts = by_day.get(day, _empty_gate_counts())
        total = sum(counts.values())
        out.append({
            "date": day,
            "total": total,
            "answered_pct": (counts["answered"] / total) if total else None,
            "no_match_pct": (counts["no-match"] / total) if total else None,
            "wide_fallback_pct": (counts["wide-fallback"] / total) if total else None,
            "partial_pct": (counts["partial"] / total) if total else None,
        })
    return out


@router.get("/spaces/{space_id}/insights")
def space_insights(space_id: str) -> dict[str, object]:
    _space_row(space_id)
    messages = _space_assistant_messages(space_id)

    total = len(messages)
    cache_hits = sum(1 for m in messages if m["cache_hit"])
    decomposed = sum(1 for m in messages if len(m["trace_obj"].get("sub_questions") or []) > 1)
    gate_counts = _empty_gate_counts()
    for m in messages:
        gate_counts[_gate_outcome(m)] += 1

    latencies = [_total_latency_ms(m["trace_obj"]) for m in messages if m["trace_obj"].get("timings")]
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0}
    for m in messages:
        tokens = m["trace_obj"].get("tokens") or {}
        total_tokens["prompt_tokens"] += tokens.get("prompt_tokens", 0)
        total_tokens["completion_tokens"] += tokens.get("completion_tokens", 0)

    stage_latency = []
    for key in STAGE_ORDER:
        vals = [m["trace_obj"]["timings"][key] for m in messages if key in (m["trace_obj"].get("timings") or {})]
        stage_latency.append({
            "stage": key, "is_api": STAGE_IS_API[key],
            "avg_ms": (sum(vals) / len(vals)) if vals else None,
            "sample_count": len(vals),
        })

    with connect(settings.db_path) as conn:
        chats = conn.execute(
            "SELECT id, title, updated_at FROM chats WHERE space_id=? ORDER BY updated_at DESC",
            (space_id,),
        ).fetchall()

    by_chat: dict[str, list[dict]] = {}
    for m in messages:
        by_chat.setdefault(m["chat_id"], []).append(m)

    chat_rows = []
    for chat in chats:
        chat_msgs = by_chat.get(chat["id"], [])
        chat_latencies = [_total_latency_ms(m["trace_obj"]) for m in chat_msgs if m["trace_obj"].get("timings")]
        chat_tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        for m in chat_msgs:
            t = m["trace_obj"].get("tokens") or {}
            chat_tokens["prompt_tokens"] += t.get("prompt_tokens", 0)
            chat_tokens["completion_tokens"] += t.get("completion_tokens", 0)
        chat_rows.append({
            "id": chat["id"], "title": chat["title"], "last_active": chat["updated_at"],
            "message_count": len(chat_msgs) * 2,  # user + assistant per turn
            "question_count": len(chat_msgs),
            "cache_hit_rate": (sum(1 for m in chat_msgs if m["cache_hit"]) / len(chat_msgs)) if chat_msgs else 0.0,
            "avg_latency_ms": (sum(chat_latencies) / len(chat_latencies)) if chat_latencies else 0.0,
            "tokens": chat_tokens,
        })
    chat_rows.sort(key=lambda r: r["question_count"], reverse=True)

    return {
        "space_id": space_id,
        "question_count": total,
        "cache_hit_rate": (cache_hits / total) if total else 0.0,
        "decomposition_rate": (decomposed / total) if total else 0.0,
        "avg_latency_ms": avg_latency_ms,
        "tokens": total_tokens,
        "gate_outcomes": gate_counts,
        "gate_outcomes_by_day": _gate_outcomes_by_day(messages),
        "stage_latency": stage_latency,
        "chats": chat_rows,
    }


@router.get("/spaces/{space_id}/chats/{chat_id}/insights")
def chat_insights(space_id: str, chat_id: str) -> dict[str, object]:
    _space_row(space_id)
    chat = _get_chat_row(chat_id)
    messages = [m for m in _space_assistant_messages(space_id) if m["chat_id"] == chat_id]

    latencies = [_total_latency_ms(m["trace_obj"]) for m in messages if m["trace_obj"].get("timings")]
    rows = []
    for m in messages:
        trace = m["trace_obj"]
        answer = trace.get("answer") or {}
        rows.append({
            "message_id": m["id"],
            "asked_at": m["created_at"],
            "question": trace.get("question") or m["content"][:80],
            "gate_outcome": _gate_outcome(m),
            "cache_hit": bool(m["cache_hit"]),
            "cache_kind": m["cache_kind"],
            "chunk_count": len(trace.get("final_chunks") or []),
            "latency_ms": _total_latency_ms(trace),
            "model": answer.get("model") or "",
            "tokens": trace.get("tokens") or {"prompt_tokens": 0, "completion_tokens": 0},
        })

    return {
        "chat_id": chat_id,
        "title": chat["title"],
        "question_count": len(messages),
        "cache_hit_rate": (sum(1 for m in messages if m["cache_hit"]) / len(messages)) if messages else 0.0,
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "questions": rows,
    }


@router.get("/spaces/{space_id}/insights/chunks")
def chunk_insights(space_id: str) -> dict[str, object]:
    """Computed on demand by scanning stored traces — no dedicated chunk-stats table,
    since this only runs when someone opens the Insights tab, not on every query."""
    _space_row(space_id)
    messages = _space_assistant_messages(space_id)

    stats: dict[str, dict[str, object]] = {}

    def touch(chunk_id: str, file_path: str) -> dict[str, object]:
        return stats.setdefault(chunk_id, {
            "chunk_id": chunk_id, "file_path": file_path,
            "candidate_count": 0, "reranked_count": 0, "used_count": 0, "rerank_scores": [],
        })

    for m in messages:
        trace = m["trace_obj"]
        for c in trace.get("candidates") or []:
            entry = touch(c["chunk"]["id"], c["chunk"]["file_path"])
            entry["candidate_count"] += 1
        for r in trace.get("reranked") or []:
            entry = touch(r["chunk"]["id"], r["chunk"]["file_path"])
            entry["reranked_count"] += 1
            entry["rerank_scores"].append(r["rerank_score"])
        for fc in trace.get("final_chunks") or []:
            if fc.get("dropped"):
                continue
            entry = touch(fc["chunk"]["id"], fc["chunk"]["file_path"])
            entry["used_count"] += 1

    chunks = []
    for entry in stats.values():
        scores = entry.pop("rerank_scores")
        entry["avg_rerank_score"] = (sum(scores) / len(scores)) if scores else None
        chunks.append(entry)
    chunks.sort(key=lambda c: c["candidate_count"], reverse=True)

    return {"space_id": space_id, "chunk_count": len(chunks), "chunks": chunks}
