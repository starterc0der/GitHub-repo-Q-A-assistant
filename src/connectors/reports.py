from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date

import psycopg2

from src.config import settings
from src.connectors.live_data import DevicePoint, PlaceDevices
from src.crypto import decrypt
from src.db import connect
from src.llm_client import LLMClient

# Shared by every ULB — unlike live-data's per-ULB Redis prefix, historical reports all
# live in one Postgres database regardless of place/device. Only CTC device ids are
# populated in it today; a Puri/BBSR device simply returns no rows, handled the same as
# any other "no report data for this device" case (see fetch_report_data).
REPORT_DB_NAME = "watco_stream_db"
REPORT_METRICS = ("pressure", "flow", "totalizer")
REPORT_GRANULARITIES = ("5min", "hourly", "daily", "monthly")

REPORT_WINDOW_SYSTEM_PROMPT = (
    "Given a question about a historical/aggregated sensor report and today's date, "
    "extract the time window it's asking about. Reply with exactly 4 lines, nothing "
    "else:\n"
    "Line 1: metric — one of pressure, flow, totalizer, or all (if the question doesn't "
    "name one specifically).\n"
    "Line 2: granularity — one of 5min, hourly, daily, monthly.\n"
    "  - 5min: the question is phrased in minutes (\"last 10 minutes\", \"last half hour\").\n"
    "  - hourly: phrased in hours (\"last 3 hours\"), or a single day where an "
    "hour-by-hour breakdown/trend is implied.\n"
    "  - daily: a single day, a range of days, a week, or the CURRENT (still in "
    "progress) month — the current month's own monthly total isn't finalized until the "
    "month ends, so an in-progress month is reported from its daily rows instead.\n"
    "  - monthly: a fully completed PAST month, for pressure or flow only — there is no "
    "monthly totalizer table at all, so a totalizer question ALWAYS uses daily "
    "(summed across the month), even for a fully completed past month.\n"
    "Line 3: start_date — YYYY-MM-DD, the first calendar day the window covers "
    "(inclusive). For an hour/minute-scale window, this is still just the day it falls "
    "on — exact minute/hour filtering happens later from the fetched data, not here.\n"
    "Line 4: end_date — YYYY-MM-DD, the last calendar day the window covers (inclusive). "
    "Same as start_date for a single day or a sub-day window.\n"
    "Never explain, never add extra lines, never use any other date format."
)

# A report question virtually always names a date ("from august 5 to august 10"), and a
# bare day-of-month number can coincidentally token-match a zone/sector number in a
# completely unrelated place's name (e.g. "...to august 10" false-matching "Zone_10" or
# "...Sector_10..." — confirmed: "zone 8" alone correctly matches only Zone_08, but
# adding a date range pulled in every other zone whose name happened to contain "10").
# Stripped only for PLACE MATCHING — window resolution and the final narration still see
# the untouched original question.
_MONTH_DAY_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)


def strip_date_phrases(question: str) -> str:
    return _MONTH_DAY_RE.sub("", question)


@dataclass
class ReportWindow:
    metric: str  # "pressure" | "flow" | "totalizer" | "all"
    granularity: str  # "5min" | "hourly" | "daily" | "monthly"
    start_date: date
    end_date: date  # inclusive


class ReportWindowResolver:
    """One cheap bulk-LLM call that turns a report question's natural-language time
    reference ("yesterday", "last 10 minutes", "this month") into a concrete window —
    see ReportWindow. Deliberately does NOT try to compute exact minute/hour slicing
    itself (see REPORT_WINDOW_SYSTEM_PROMPT); it only picks which table(s) to open and
    which calendar days to fetch. The final answer LLM, given the full fetched buckets
    plus the original question, decides which subset actually answers it — same
    fetch-everything-relevant/let-the-LLM-narrow-it pattern as live_data.py."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def resolve(self, question: str, today: date) -> ReportWindow | None:
        prompt = f"Today's date is {today.isoformat()}.\n\nQuestion: {question}"
        try:
            reply = self.llm.complete(prompt, system=REPORT_WINDOW_SYSTEM_PROMPT).strip()
        except RuntimeError:
            return None
        lines = [line.strip() for line in reply.splitlines() if line.strip()]
        if len(lines) < 4:
            return None
        metric = lines[0].lower()
        granularity = lines[1].lower()
        if metric not in REPORT_METRICS and metric != "all":
            return None
        if granularity not in REPORT_GRANULARITIES:
            return None
        try:
            start = date.fromisoformat(lines[2])
            end = date.fromisoformat(lines[3])
        except ValueError:
            return None
        if end < start:
            start, end = end, start
        return ReportWindow(metric=metric, granularity=granularity, start_date=start, end_date=end)


def _space_postgres_connector(space_id: str) -> dict | None:
    """The first Postgres connector configured for this space, or None — same contract
    as live_data._space_redis_connector. Its stored `database` is whatever the space's
    core-data DB is (e.g. ctc_core_db); reports live in a different, fixed database on
    the same server, so REPORT_DB_NAME is used instead, not connector['database']."""
    with connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT host, port, username, encrypted_password, ssl "
            "FROM connectors WHERE space_id=? AND kind='postgres' ORDER BY created_at LIMIT 1",
            (space_id,),
        ).fetchone()
    return dict(row) if row else None


def _pg_connect(connector: dict):
    password = decrypt(connector["encrypted_password"], settings.connector_encryption_key)
    return psycopg2.connect(
        host=connector["host"], port=connector["port"], dbname=REPORT_DB_NAME,
        user=connector["username"], password=password,
        sslmode="require" if connector["ssl"] else "prefer", connect_timeout=5,
    )


@dataclass
class PointReport:
    point: DevicePoint
    metric: str
    # granularity in {"daily", "monthly"}: {date_or_month_str: value}, one entry per
    # period in range. granularity in {"5min", "hourly"}: {date_str: {"HH:MM": value}},
    # one entry per day in range. Empty dict means no rows at all for this point/metric.
    values: dict


def fetch_report_data(
    space_id: str, points: list[DevicePoint], window: ReportWindow
) -> list[PointReport] | None:
    """Fetches every requested metric for every point over the window. Returns None
    (never raises) when there's no Postgres connector configured for the space — the
    caller falls back to a normal answer, same contract as live_data.fetch_live_readings.
    A point/metric with zero matching rows still gets a PointReport with empty `values`
    (not omitted) so build_report_context can say so honestly rather than silently
    dropping it."""
    connector = _space_postgres_connector(space_id)
    if connector is None:
        return None
    metrics = REPORT_METRICS if window.metric == "all" else (window.metric,)

    reports: list[PointReport] = []
    conn = None
    try:
        conn = _pg_connect(connector)
        cur = conn.cursor()
        for point in points:
            for metric in metrics:
                reports.append(_fetch_one(cur, point, metric, window))
    except Exception:
        return None
    finally:
        if conn is not None:
            conn.close()
    return reports


def _fetch_one(cur, point: DevicePoint, metric: str, window: ReportWindow) -> PointReport:
    # No totalizer_monthly table exists at all (see REPORT_WINDOW_SYSTEM_PROMPT) — a
    # totalizer's monthly figure is just its daily rows summed anyway, so silently
    # redirect here rather than relying solely on the classifier prompt to never ask
    # for it (it also can't, when metric="all" bundles totalizer with pressure/flow,
    # which DO have monthly tables).
    granularity = "daily" if (metric == "totalizer" and window.granularity == "monthly") else window.granularity
    table = f"{metric}_{granularity}"
    if metric == "totalizer" and window.granularity == "monthly":
        # Sum the daily rows into one monthly-shaped total, same key convention as a
        # real monthly row (see the else branch below) so build_report_context doesn't
        # need to know this redirect happened.
        cur.execute(
            f"SELECT value FROM {table} "  # noqa: S608
            "WHERE device_id=%s AND pid=%s AND type=%s AND metric_date BETWEEN %s AND %s",
            (point.device_id, point.pid, point.device_type, window.start_date, window.end_date),
        )
        rows = cur.fetchall()
        values = {window.start_date.replace(day=1).isoformat(): sum(v for (v,) in rows)} if rows else {}
        return PointReport(point=point, metric=metric, values=values)
    if granularity == "daily":
        cur.execute(
            f"SELECT metric_date, value FROM {table} "  # noqa: S608 (table from a closed whitelist, see REPORT_GRANULARITIES/REPORT_METRICS)
            "WHERE device_id=%s AND pid=%s AND type=%s AND metric_date BETWEEN %s AND %s "
            "ORDER BY metric_date",
            (point.device_id, point.pid, point.device_type, window.start_date, window.end_date),
        )
        values = {d.isoformat(): v for d, v in cur.fetchall()}
    elif granularity in ("hourly", "5min"):
        cur.execute(
            f"SELECT metric_date, value_json FROM {table} "  # noqa: S608
            "WHERE device_id=%s AND pid=%s AND type=%s AND metric_date BETWEEN %s AND %s "
            "ORDER BY metric_date",
            (point.device_id, point.pid, point.device_type, window.start_date, window.end_date),
        )
        values = {d.isoformat(): vj for d, vj in cur.fetchall()}
    else:  # monthly — one row per calendar month, keyed by that month's first-of-month date
        cur.execute(
            f"SELECT metric_month, value FROM {table} "  # noqa: S608
            "WHERE device_id=%s AND pid=%s AND type=%s AND metric_month BETWEEN %s AND %s "
            "ORDER BY metric_month",
            (point.device_id, point.pid, point.device_type, window.start_date, window.end_date),
        )
        values = {m.isoformat(): v for m, v in cur.fetchall()}
    return PointReport(point=point, metric=metric, values=values)


def build_report_context(
    places: list[PlaceDevices], window: ReportWindow, reports: list[PointReport]
) -> str:
    """The already-fetched, human-readable context handed to the generation LLM —
    mirrors live_data.build_live_data_context's shape/purpose exactly. `reports` is
    matched back to each place's points by (device_id, pid, device_type, metric)."""
    by_point: dict[tuple[str, str, str, str], PointReport] = {
        (r.point.device_id, r.point.pid, r.point.device_type, r.metric): r for r in reports
    }
    metrics = REPORT_METRICS if window.metric == "all" else (window.metric,)
    window_label = (
        window.start_date.isoformat()
        if window.start_date == window.end_date
        else f"{window.start_date.isoformat()} to {window.end_date.isoformat()}"
    )
    sections = [f"Requested window: {window_label} ({window.granularity} granularity)."]
    for place in places:
        lines = [f"Place: {place.place_name}"]
        for p in place.points:
            label = f"{p.location} ({p.device_type})"
            for metric in metrics:
                report = by_point.get((p.device_id, p.pid, p.device_type, metric))
                if report is None or not report.values:
                    lines.append(f"- {label} {metric}: no report data available")
                    continue
                lines.append(f"- {label} {metric}: {json.dumps(report.values)}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def infer_chart_kind(question: str) -> str:
    q = question.lower()
    return "bar" if ("bar chart" in q or "bar graph" in q) else "trend"


def _point_values_flat(report: PointReport) -> dict[str, float]:
    """Normalizes daily/monthly values (already flat: {period: value}) and hourly/5min
    values (nested: {day: {"HH:MM": value}}) into one flat {label: value} — an
    hourly/5min label combines the day and the intraday bucket ("2026-08-05 14:00")
    since each is its own real data point."""
    flat: dict[str, float] = {}
    for key, val in report.values.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                flat[f"{key} {sub_key}"] = sub_val
        else:
            flat[key] = val
    return flat


def _report_entries(
    matches: list[PlaceDevices], reports: list[PointReport], window: ReportWindow
) -> tuple[list[tuple[str, dict[str, float]]], list[str]]:
    """One (label, flat_values) pair per matched point/metric that has real data, plus
    the labels of every matched point/metric that doesn't — label is prefixed with the
    place name whenever more than one place is matched, since two different places
    routinely share a sub-place name (both commonly have one called "ESR"), and without
    the prefix there'd be no way to tell which place a given row/series belongs to."""
    multi_place = len(matches) > 1
    metrics = REPORT_METRICS if window.metric == "all" else (window.metric,)
    by_point = {(r.point.device_id, r.point.pid, r.point.device_type, r.metric): r for r in reports}
    entries: list[tuple[str, dict[str, float]]] = []
    missing: list[str] = []
    for place in matches:
        for p in place.points:
            for metric in metrics:
                label = f"{place.place_name} — {p.location} ({p.device_type})" if multi_place else f"{p.location} ({p.device_type})"
                if len(metrics) > 1:
                    label = f"{label} [{metric}]"
                report = by_point.get((p.device_id, p.pid, p.device_type, metric))
                if report is None or not report.values:
                    missing.append(label)
                    continue
                entries.append((label, _point_values_flat(report)))
    return entries, missing


def build_report_block(
    matches: list[PlaceDevices], reports: list[PointReport], window: ReportWindow,
    wants_chart: bool, chart_kind: str,
) -> tuple[str, dict | None, dict | None]:
    """Builds the answer text and ```table/```chart JSON directly from the already-
    fetched data instead of asking an LLM to format it — removes the two failure modes
    observed doing that: malformed JSON on a long multi-series array, and the model
    sometimes omitting the block entirely despite real data existing even after a
    retry (confirmed live, repeatedly — not a one-off). Returns (text, table, chart) —
    at most one of table/chart is set; text is a deterministic note about any matched
    point/metric with no data, or the sole answer when nothing at all has data."""
    multi_place = len(matches) > 1
    entries, missing = _report_entries(matches, reports, window)
    if not entries:
        return "No report data is available for any of the requested places and sub-places for this window.", None, None
    text = f"No report data available for: {', '.join(missing)}." if missing else ""

    if not wants_chart:
        all_keys = sorted({k for _label, values in entries for k in values})
        if len(all_keys) <= 1:
            key = all_keys[0] if all_keys else None
            columns = ["Place", "Sub-place", "Value"] if multi_place else ["Sub-place", "Value"]
            rows = []
            for label, values in entries:
                value = values.get(key) if key else None
                if value is None:
                    continue
                if multi_place:
                    place_part, sub_part = label.split(" — ", 1)
                    rows.append([place_part, sub_part, round(value, 4)])
                else:
                    rows.append([label, round(value, 4)])
            return text, {"columns": columns, "rows": rows}, None
        columns = ["Sub-place", *all_keys]
        rows = [
            [label, *[round(values[k], 4) if k in values else "" for k in all_keys]]
            for label, values in entries
        ]
        return text, {"columns": columns, "rows": rows}, None

    # Chart series must be equal-length and all-numeric (see ChartParser._validate) —
    # restrict categories to ones every included series actually has a value for, so
    # there's never a null/placeholder to invent.
    shared_keys: set[str] | None = None
    for _label, values in entries:
        shared_keys = set(values) if shared_keys is None else shared_keys & set(values)
    categories = sorted(shared_keys or set())
    if not categories:
        return "No report data is available for any of the requested places and sub-places for this window.", None, None
    series = [{"name": label, "values": [round(values[k], 4) for k in categories]} for label, values in entries]
    title_metric = window.metric if window.metric != "all" else "Report"
    title = f"{title_metric.capitalize()} — {', '.join(p.place_name for p in matches)}"
    return text, None, {"title": title, "kind": chart_kind, "categories": categories, "series": series}
