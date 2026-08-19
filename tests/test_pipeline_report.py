from __future__ import annotations

from datetime import date
from unittest.mock import patch

from src.config import Settings
from src.connectors.live_data import DevicePoint
from src.connectors.reports import PointReport, ReportWindow
from src.index.schema import CodeChunk
from src.pipeline import Pipeline

PLACE_DOC = """\
## Zone_08_Krushak_Bazar
ULB: cuttack
Inlet: device_id=00-00-54-5F-05-FC, pid=FC1, location=ESR
Outlets (sub-places):
- device_id=00-00-54-5F-05-FC, pid=FC3, sub_place=Mahtab_Nagar
"""

_ESR_POINT = DevicePoint("inlet", "00-00-54-5F-05-FC", "FC1", "ESR")
_MAHTAB_POINT = DevicePoint("outlet", "00-00-54-5F-05-FC", "FC3", "Mahtab_Nagar")


class _FakeWindowResolver:
    def __init__(self, window: ReportWindow | None) -> None:
        self.window = window
        self.calls: list[tuple[str, date]] = []

    def resolve(self, question: str, today: date) -> ReportWindow | None:
        self.calls.append((question, today))
        return self.window


def _pipeline_stub(window: ReportWindow | None) -> tuple[Pipeline, _FakeWindowResolver]:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = Settings()
    resolver = _FakeWindowResolver(window)
    pipeline.report_window_resolver = resolver
    return pipeline, resolver


def _chunk(code: str) -> CodeChunk:
    return CodeChunk(
        id="c1", space_id="demo", source_id="src1", file_path="ctc device data",
        language="text", symbol_name=None, start_line=1, end_line=1, code=code,
    )


_WINDOW = ReportWindow("pressure", "daily", date(2026, 8, 5), date(2026, 8, 5))


@patch("src.pipeline.fetch_report_data")
def test_try_report_answer_matches_place_and_builds_a_table(mock_fetch) -> None:
    pipeline, resolver = _pipeline_stub(_WINDOW)
    mock_fetch.return_value = [
        PointReport(point=_ESR_POINT, metric="pressure", values={"2026-08-05": 3.2}),
        PointReport(point=_MAHTAB_POINT, metric="pressure", values={"2026-08-05": 2.9}),
    ]

    text, table, chart, tool_trace = pipeline._try_report_answer(
        "pressure report for zone 8 on august 5", "space1", [_chunk(PLACE_DOC)]
    )

    assert text == ""
    assert chart is None
    assert table["columns"] == ["Sub-place", "Value"]
    assert ["ESR (inlet)", 3.2] in table["rows"]
    assert ["Mahtab_Nagar (outlet)", 2.9] in table["rows"]
    assert tool_trace.matched_places == ["Zone_08_Krushak_Bazar"]
    assert tool_trace.metric == "pressure" and tool_trace.granularity == "daily"
    # resolver got the real question, not something hardcoded
    assert resolver.calls[0][0] == "pressure report for zone 8 on august 5"


@patch("src.pipeline.fetch_report_data")
def test_try_report_answer_builds_a_chart_when_wants_chart(mock_fetch) -> None:
    window = ReportWindow("pressure", "daily", date(2026, 8, 5), date(2026, 8, 6))
    pipeline, _resolver = _pipeline_stub(window)
    mock_fetch.return_value = [
        PointReport(point=_ESR_POINT, metric="pressure", values={"2026-08-05": 3.1, "2026-08-06": 3.2}),
    ]

    text, table, chart, _tool_trace = pipeline._try_report_answer(
        "show me the pressure trend for zone 8", "space1", [_chunk(PLACE_DOC)], wants_chart=True,
    )

    assert table is None
    assert chart["kind"] == "trend"
    assert chart["categories"] == ["2026-08-05", "2026-08-06"]
    assert chart["series"] == [{"name": "ESR (inlet)", "values": [3.1, 3.2]}]


@patch("src.pipeline.fetch_report_data")
def test_try_report_answer_uses_bar_kind_when_explicitly_asked(mock_fetch) -> None:
    window = ReportWindow("pressure", "daily", date(2026, 8, 5), date(2026, 8, 5))
    pipeline, _resolver = _pipeline_stub(window)
    mock_fetch.return_value = [PointReport(point=_ESR_POINT, metric="pressure", values={"2026-08-05": 3.1})]

    _text, _table, chart, _tool_trace = pipeline._try_report_answer(
        "show me the pressure as a bar chart for zone 8", "space1", [_chunk(PLACE_DOC)], wants_chart=True,
    )

    assert chart["kind"] == "bar"


def test_try_report_answer_returns_none_when_no_place_doc_chunk_present() -> None:
    pipeline, _resolver = _pipeline_stub(_WINDOW)

    result = pipeline._try_report_answer(
        "pressure report for zone 8", "space1", [_chunk("Once upon a time in Ayodhya...")]
    )

    assert result is None


def test_try_report_answer_returns_none_when_question_matches_no_place() -> None:
    pipeline, _resolver = _pipeline_stub(_WINDOW)

    result = pipeline._try_report_answer("who is Krishna", "space1", [_chunk(PLACE_DOC)])

    assert result is None


def test_try_report_answer_returns_none_when_window_cannot_be_resolved() -> None:
    pipeline, _resolver = _pipeline_stub(window=None)

    result = pipeline._try_report_answer(
        "pressure report for zone 8 sometime", "space1", [_chunk(PLACE_DOC)]
    )

    assert result is None


@patch("src.pipeline.fetch_report_data")
def test_try_report_answer_returns_none_when_no_postgres_connector_available(mock_fetch) -> None:
    pipeline, _resolver = _pipeline_stub(_WINDOW)
    mock_fetch.return_value = None

    result = pipeline._try_report_answer("pressure report for zone 8", "space1", [_chunk(PLACE_DOC)])

    assert result is None


@patch("src.pipeline.fetch_report_data")
def test_try_report_answer_narrates_honest_no_data_without_falling_back(mock_fetch) -> None:
    # A matched place with a real Postgres connection but zero rows must still get a
    # real (honest) answer, not a silent None-triggered fallback to the normal
    # chunk-grounded answer describing the static catalog doc instead.
    pipeline, _resolver = _pipeline_stub(_WINDOW)
    mock_fetch.return_value = [PointReport(point=_ESR_POINT, metric="pressure", values={})]

    result = pipeline._try_report_answer("pressure report for zone 8", "space1", [_chunk(PLACE_DOC)])

    assert result is not None
    text, table, chart, _tool_trace = result
    assert "no report data is available for any" in text.lower()
    assert table is None and chart is None


@patch("src.pipeline.fetch_report_data")
def test_try_report_answer_includes_available_data_despite_partial_missing(mock_fetch) -> None:
    # The old LLM-generated version of this repeatedly (confirmed live) described only
    # the missing sub-place and omitted the table entirely, even with other sub-places
    # having real data. Building it in code removes that judgment call altogether.
    pipeline, _resolver = _pipeline_stub(_WINDOW)
    mock_fetch.return_value = [
        PointReport(point=_ESR_POINT, metric="pressure", values={}),
        PointReport(point=_MAHTAB_POINT, metric="pressure", values={"2026-08-05": 2.9}),
    ]

    text, table, chart, _tool_trace = pipeline._try_report_answer(
        "pressure report for zone 8", "space1", [_chunk(PLACE_DOC)]
    )

    assert table == {"columns": ["Sub-place", "Value"], "rows": [["Mahtab_Nagar (outlet)", 2.9]]}
    assert "ESR (inlet)" in text
