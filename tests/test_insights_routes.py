from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.api.insights_routes import (
    MAX_RANGE_DAYS,
    _cache_hit_by_day,
    _daily_buckets,
    _decomposition_by_day,
    _faithful_or_none,
    _faithfulness_by_day,
    _gate_outcomes_by_day,
    _latency_by_day,
    _resolve_range,
    _tokens_by_day,
)


def _msg(
    created_at: str, cache_hit: bool = False, prompt_tokens: int = 0, completion_tokens: int = 0,
    citations: list[list[str]] | None = None, timings: dict[str, float] | None = None,
    sub_questions: list[str] | None = None,
) -> dict:
    """citations: one list of chunk_ids per claim, e.g. [["c1"], []] = 2 claims, 2nd unsupported."""
    return {
        "created_at": f"{created_at}T12:00:00+00:00",
        "cache_hit": cache_hit,
        "trace_obj": {
            "tokens": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            "answer": {"citations": [{"chunk_ids": ids} for ids in citations]} if citations is not None else {},
            "timings": timings,
            "sub_questions": sub_questions,
        },
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


def test_faithful_or_none_is_none_when_message_has_no_citations() -> None:
    assert _faithful_or_none(_msg("2026-01-01")) is None
    assert _faithful_or_none(_msg("2026-01-01", citations=[])) is None


def test_faithful_or_none_true_when_every_claim_has_a_supporting_chunk() -> None:
    msg = _msg("2026-01-01", citations=[["c1"], ["c2", "c3"]])

    assert _faithful_or_none(msg) is True


def test_faithful_or_none_false_when_any_claim_is_unsupported() -> None:
    msg = _msg("2026-01-01", citations=[["c1"], []])

    assert _faithful_or_none(msg) is False


def test_faithfulness_by_day_reports_null_on_a_day_with_nothing_to_check() -> None:
    # 2 messages that day: one fully faithful, one with an unsupported claim -> 1 of 2.
    daily = _daily_buckets(
        [
            _msg("2026-01-01", citations=[["c1"]]),
            _msg("2026-01-01", citations=[["c1"], []]),
        ],
        "2026-01-01", "2026-01-02",
    )

    result = _faithfulness_by_day(daily)

    assert result[0]["total"] == 2
    assert result[0]["faithful_rate"] == 0.5
    assert result[1]["total"] == 0
    assert result[1]["faithful_rate"] is None


def test_faithfulness_by_day_excludes_meta_and_no_match_messages_from_denominator() -> None:
    # A meta/no-match message has no citations at all — must not count as a 0%-faithful
    # answer, and must not count toward the day's total either.
    daily = _daily_buckets(
        [_msg("2026-01-01", citations=[["c1"]]), _msg("2026-01-01")], "2026-01-01", "2026-01-01"
    )

    result = _faithfulness_by_day(daily)

    assert result[0]["total"] == 1
    assert result[0]["faithful_rate"] == 1.0


def test_tokens_by_day_sums_prompt_and_completion_tokens() -> None:
    daily = _daily_buckets(
        [_msg("2026-01-01", prompt_tokens=100, completion_tokens=10)], "2026-01-01", "2026-01-01"
    )

    result = _tokens_by_day(daily)

    assert result[0] == {"date": "2026-01-01", "prompt_tokens": 100, "completion_tokens": 10}


def test_latency_by_day_reports_null_on_a_day_with_no_timed_messages() -> None:
    daily = _daily_buckets(
        [_msg("2026-01-01", timings={"generate": 800.0, "rerank": 200.0})], "2026-01-01", "2026-01-02"
    )

    result = _latency_by_day(daily)

    assert result[0]["avg_ms"] == 1000.0
    assert result[1]["avg_ms"] is None


def test_decomposition_by_day_reports_null_on_a_day_with_no_questions() -> None:
    daily = _daily_buckets(
        [
            _msg("2026-01-01", sub_questions=["a", "b"]),
            _msg("2026-01-01", sub_questions=["a"]),
        ],
        "2026-01-01", "2026-01-02",
    )

    result = _decomposition_by_day(daily)

    assert result[0]["rate"] == 0.5
    assert result[1]["rate"] is None
