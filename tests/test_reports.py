from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from src.config import settings
from src.connectors.live_data import DevicePoint, PlaceDevices
from src.connectors.reports import (
    PointReport,
    ReportWindow,
    build_report_block,
    build_report_context,
    fetch_report_data,
    infer_chart_kind,
    strip_date_phrases,
)
from src.crypto import encrypt
from src.db import connect, init_db, new_id, now


@pytest.fixture(autouse=True)
def _isolated_db_and_key(tmp_path, monkeypatch):
    db_path = str(tmp_path / "app.db")
    monkeypatch.setattr(settings, "db_path", db_path)
    monkeypatch.setattr(settings, "connector_encryption_key", Fernet.generate_key().decode())
    init_db(db_path)


def _make_space_with_postgres_connector() -> str:
    space_id = new_id()
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO spaces (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (space_id, "Demo", now(), now()),
        )
        conn.execute(
            "INSERT INTO connectors (id, space_id, kind, name, host, port, database, "
            "username, encrypted_password, db_index, tls, ssl, status, last_tested_at, created_at) "
            "VALUES (?, ?, 'postgres', 'watco db', 'pg.example.internal', 5432, 'core_db', "
            "'testuser', ?, NULL, 0, 0, 'connected', ?, ?)",
            (new_id(), space_id, encrypt("testpass123", settings.connector_encryption_key), now(), now()),
        )
    return space_id


def _cursor_returning(rows: list[tuple]) -> MagicMock:
    cur = MagicMock()
    cur.fetchall.return_value = rows
    return cur


def test_strip_date_phrases_removes_month_and_day_only() -> None:
    # Regression: "...to august 10" was false-matching "Zone_10"/"...Sector_10..." via
    # plain token overlap with the place-matcher, purely because both contain "10" —
    # unrelated to the actual place named in the question ("zone 8").
    q = "show me the pressure trend for zone 8 from august 5 to august 10"

    stripped = strip_date_phrases(q)

    assert "5" not in stripped.split() and "10" not in stripped.split()
    assert "zone 8" in stripped  # the actual place reference is untouched


def test_strip_date_phrases_leaves_non_date_text_alone() -> None:
    assert strip_date_phrases("what is the pressure at zone 8") == "what is the pressure at zone 8"


def test_build_report_context_reports_missing_data_honestly() -> None:
    place = PlaceDevices(
        "Zone_08_Krushak_Bazar", "cuttack",
        [DevicePoint("inlet", "AA-BB", "FC1", "ESR")],
    )
    window = ReportWindow("pressure", "daily", date(2026, 8, 5), date(2026, 8, 5))

    context = build_report_context([place], window, reports=[])

    assert "ESR (inlet) pressure: no report data available" in context


def test_build_report_context_includes_fetched_values() -> None:
    point = DevicePoint("inlet", "AA-BB", "FC1", "ESR")
    place = PlaceDevices("Zone_08_Krushak_Bazar", "cuttack", [point])
    window = ReportWindow("pressure", "daily", date(2026, 8, 5), date(2026, 8, 5))
    report = PointReport(point=point, metric="pressure", values={"2026-08-05": 3.2})

    context = build_report_context([place], window, reports=[report])

    assert '"2026-08-05": 3.2' in context
    assert "no report data available" not in context


def test_infer_chart_kind_bar_only_when_explicitly_named() -> None:
    assert infer_chart_kind("compare pressure between zone 8 and 11 as a bar chart") == "bar"
    assert infer_chart_kind("show it as a bar graph") == "bar"
    assert infer_chart_kind("compare pressure between zone 8 and 11 as a graph") == "trend"
    assert infer_chart_kind("show me the pressure trend for zone 8") == "trend"


_PT_A = DevicePoint("inlet", "AA-BB", "1", "ESR")
_PT_B = DevicePoint("outlet", "AA-BB", "2", "Deer_Park")


def test_build_report_block_single_place_table_has_no_place_prefix() -> None:
    place = PlaceDevices("Zone_08", "cuttack", [_PT_A])
    window = ReportWindow("pressure", "daily", date(2026, 8, 5), date(2026, 8, 5))
    reports = [PointReport(point=_PT_A, metric="pressure", values={"2026-08-05": 3.2})]

    text, table, chart = build_report_block([place], reports, window, wants_chart=False, chart_kind="trend")

    assert chart is None
    assert table == {"columns": ["Sub-place", "Value"], "rows": [["ESR (inlet)", 3.2]]}
    assert text == ""


def test_build_report_block_multi_place_table_prefixes_place_name() -> None:
    place_a = PlaceDevices("Zone_08", "cuttack", [_PT_A])
    place_b = PlaceDevices("Zone_11", "cuttack", [_PT_B])
    window = ReportWindow("pressure", "daily", date(2026, 8, 5), date(2026, 8, 5))
    reports = [
        PointReport(point=_PT_A, metric="pressure", values={"2026-08-05": 3.2}),
        PointReport(point=_PT_B, metric="pressure", values={"2026-08-05": 2.9}),
    ]

    text, table, chart = build_report_block(
        [place_a, place_b], reports, window, wants_chart=False, chart_kind="trend"
    )

    assert chart is None
    assert table["columns"] == ["Place", "Sub-place", "Value"]
    assert ["Zone_08", "ESR (inlet)", 3.2] in table["rows"]
    assert ["Zone_11", "Deer_Park (outlet)", 2.9] in table["rows"]


def test_build_report_block_multi_place_chart_prefixes_series_names() -> None:
    place_a = PlaceDevices("Zone_08", "cuttack", [_PT_A])
    place_b = PlaceDevices("Zone_11", "cuttack", [_PT_B])
    window = ReportWindow("pressure", "daily", date(2026, 8, 5), date(2026, 8, 6))
    reports = [
        PointReport(point=_PT_A, metric="pressure", values={"2026-08-05": 3.1, "2026-08-06": 3.2}),
        PointReport(point=_PT_B, metric="pressure", values={"2026-08-05": 2.8, "2026-08-06": 2.9}),
    ]

    text, table, chart = build_report_block(
        [place_a, place_b], reports, window, wants_chart=True, chart_kind="bar"
    )

    assert table is None
    assert chart["kind"] == "bar"
    assert chart["categories"] == ["2026-08-05", "2026-08-06"]
    names = {s["name"] for s in chart["series"]}
    assert names == {"Zone_08 — ESR (inlet)", "Zone_11 — Deer_Park (outlet)"}


def test_build_report_block_includes_available_data_despite_partial_missing() -> None:
    # Regression: the LLM-generated version of this repeatedly (confirmed live, not
    # hypothesized) described only the missing sub-place and omitted the chart/table
    # entirely, even though other sub-places had real data. Building it in code makes
    # that failure mode structurally impossible — there's no "should I include this"
    # judgment call left to get wrong.
    place = PlaceDevices("Zone_08", "cuttack", [_PT_A, _PT_B])
    window = ReportWindow("pressure", "daily", date(2026, 8, 5), date(2026, 8, 5))
    reports = [
        PointReport(point=_PT_A, metric="pressure", values={}),
        PointReport(point=_PT_B, metric="pressure", values={"2026-08-05": 2.9}),
    ]

    text, table, chart = build_report_block([place], reports, window, wants_chart=False, chart_kind="trend")

    assert table == {"columns": ["Sub-place", "Value"], "rows": [["Deer_Park (outlet)", 2.9]]}
    assert "ESR (inlet)" in text
    assert "no report data" in text.lower() or "no report data available" in text.lower()


def test_build_report_block_returns_no_block_when_nothing_has_data() -> None:
    place = PlaceDevices("Zone_08", "cuttack", [_PT_A])
    window = ReportWindow("pressure", "daily", date(2026, 8, 5), date(2026, 8, 5))
    reports = [PointReport(point=_PT_A, metric="pressure", values={})]

    text, table, chart = build_report_block([place], reports, window, wants_chart=True, chart_kind="trend")

    assert table is None and chart is None
    assert "no report data is available for any" in text.lower()


def test_fetch_report_data_returns_none_without_a_configured_connector() -> None:
    space_id = new_id()
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO spaces (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (space_id, "Demo", now(), now()),
        )
    window = ReportWindow("pressure", "daily", date(2026, 8, 5), date(2026, 8, 5))

    assert fetch_report_data(space_id, [DevicePoint("inlet", "AA-BB", "FC1", "ESR")], window) is None


@patch("src.connectors.reports.psycopg2.connect")
def test_fetch_report_data_daily(mock_connect: MagicMock) -> None:
    space_id = _make_space_with_postgres_connector()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = _cursor_returning([(date(2026, 8, 5), 3.2)])
    mock_connect.return_value = mock_conn
    window = ReportWindow("pressure", "daily", date(2026, 8, 5), date(2026, 8, 5))

    reports = fetch_report_data(space_id, [DevicePoint("inlet", "AA-BB", "FC1", "ESR")], window)

    assert reports[0].values == {"2026-08-05": 3.2}
    # connects to the fixed report DB, not the connector's own stored `database`
    assert mock_connect.call_args.kwargs["dbname"] == "watco_stream_db"
    assert mock_connect.call_args.kwargs["password"] == "testpass123"


@patch("src.connectors.reports.psycopg2.connect")
def test_fetch_report_data_redirects_totalizer_monthly_to_summed_daily(mock_connect: MagicMock) -> None:
    # No totalizer_monthly table exists at all — a monthly totalizer request must sum
    # the daily rows instead of querying a table that doesn't exist.
    space_id = _make_space_with_postgres_connector()
    mock_conn = MagicMock()
    mock_cursor = _cursor_returning([(4.6801,), (8.116,)])
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn
    window = ReportWindow("totalizer", "monthly", date(2026, 7, 1), date(2026, 7, 31))

    reports = fetch_report_data(space_id, [DevicePoint("inlet", "AA-BB", "FC1", "ESR")], window)

    assert reports[0].values == {"2026-07-01": pytest.approx(12.7961)}
    executed_sql = mock_cursor.execute.call_args.args[0]
    assert "totalizer_daily" in executed_sql
    assert "totalizer_monthly" not in executed_sql


@patch("src.connectors.reports.psycopg2.connect")
def test_fetch_report_data_returns_none_on_connection_failure(mock_connect: MagicMock) -> None:
    space_id = _make_space_with_postgres_connector()
    mock_connect.side_effect = ConnectionError("boom")
    window = ReportWindow("pressure", "daily", date(2026, 8, 5), date(2026, 8, 5))

    assert fetch_report_data(space_id, [DevicePoint("inlet", "AA-BB", "FC1", "ESR")], window) is None
