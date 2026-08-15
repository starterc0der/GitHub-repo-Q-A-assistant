from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.api.insights_routes import (
    MAX_RANGE_DAYS,
    _cache_hit_by_day,
    _daily_buckets,
    _gate_outcomes_by_day,
    _resolve_range,
    _tokens_by_day,
)


def _msg(created_at: str, cache_hit: bool = False, prompt_tokens: int = 0, completion_tokens: int = 0) -> dict:
    return {
        "created_at": f"{created_at}T12:00:00+00:00",
        "cache_hit": cache_hit,
        "trace_obj": {"tokens": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}},
    }


def test_resolve_range_defaults_to_last_14_days_ending_today() -> None:
    today = datetime.now(UTC).date()

    start, end, _min, max_date = _resolve_range([], None, None)

    assert end == today.isoformat() == max_date
    assert start == (today - timedelta(days=13)).isoformat()


def test_resolve_range_uses_explicit_start_and_end() -> None:
    start, end, _min, _max = _resolve_range([], "2026-01-01", "2026-01-10")

    assert start == "2026-01-01"
    assert end == "2026-01-10"


def test_resolve_range_swaps_reversed_start_and_end() -> None:
    start, end, _min, _max = _resolve_range([], "2026-01-10", "2026-01-01")

    assert start == "2026-01-01"
    assert end == "2026-01-10"


def test_resolve_range_clamps_end_to_today() -> None:
    today = datetime.now(UTC).date()
    far_future = (today + timedelta(days=30)).isoformat()

    _start, end, _min, _max = _resolve_range([], today.isoformat(), far_future)

    assert end == today.isoformat()


def test_resolve_range_clamps_oversized_span() -> None:
    start, end, _min, _max = _resolve_range([], "2000-01-01", "2026-01-01")

    span_days = (
        datetime.fromisoformat(end).date() - datetime.fromisoformat(start).date()
    ).days
    assert span_days == MAX_RANGE_DAYS


def test_resolve_range_min_date_comes_from_earliest_message() -> None:
    messages = [_msg("2025-03-01"), _msg("2025-06-15")]

    _start, _end, range_min, _max = _resolve_range(messages, None, None)

    assert range_min == "2025-03-01"


def test_resolve_range_min_date_falls_back_to_today_with_no_messages() -> None:
    today = datetime.now(UTC).date()

    _start, _end, range_min, _max = _resolve_range([], None, None)

    assert range_min == today.isoformat()


def test_daily_buckets_aggregates_gate_cache_and_tokens_per_day() -> None:
    messages = [
        _msg("2026-01-01", cache_hit=True, prompt_tokens=100, completion_tokens=10),
        _msg("2026-01-01", cache_hit=False, prompt_tokens=50, completion_tokens=5),
        _msg("2026-01-03", cache_hit=False, prompt_tokens=20, completion_tokens=2),
    ]

    daily = _daily_buckets(messages, "2026-01-01", "2026-01-03")

    assert [b["date"] for b in daily] == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert daily[0]["total"] == 2
    assert daily[0]["cache_hits"] == 1
    assert daily[0]["prompt_tokens"] == 150
    assert daily[0]["completion_tokens"] == 15
    assert daily[1]["total"] == 0  # no messages that day — zero-filled, not skipped
    assert daily[2]["total"] == 1


def test_gate_outcomes_by_day_reports_null_on_a_day_with_no_questions() -> None:
    daily = _daily_buckets([_msg("2026-01-01")], "2026-01-01", "2026-01-02")

    result = _gate_outcomes_by_day(daily)

    assert result[0]["answered_pct"] == 1.0
    assert result[1]["total"] == 0
    assert result[1]["answered_pct"] is None


def test_cache_hit_by_day_reports_null_on_a_day_with_no_questions() -> None:
    daily = _daily_buckets([_msg("2026-01-01", cache_hit=True)], "2026-01-01", "2026-01-02")

    result = _cache_hit_by_day(daily)

    assert result[0]["hit_rate"] == 1.0
    assert result[1]["hit_rate"] is None


def test_tokens_by_day_sums_prompt_and_completion_tokens() -> None:
    daily = _daily_buckets(
        [_msg("2026-01-01", prompt_tokens=100, completion_tokens=10)], "2026-01-01", "2026-01-01"
    )

    result = _tokens_by_day(daily)

    assert result[0] == {"date": "2026-01-01", "prompt_tokens": 100, "completion_tokens": 10}
