from __future__ import annotations

from unittest.mock import patch

from src.config import Settings
from src.index.schema import CodeChunk
from src.pipeline import Pipeline

PLACE_DOC = """\
## Zone_11_Gorakabar
ULB: cuttack
Inlet: device_id=00-80-F4-2D-32-35, pid=351, location=ESR
Outlets (sub-places):
- device_id=00-80-F4-2D-32-35, pid=351, sub_place=Deer_Park

## Zone_16_Killa
ULB: cuttack
Inlet: device_id=A8-74-1D-16-73-10, pid=101, location=ESR
Outlets (sub-places):
- device_id=A8-74-1D-16-73-10, pid=102, sub_place=KILLA_SUB_DMA_2
"""


class _FakeLLM:
    def __init__(self, reply: str = "The pressure at Deer_Park is 4.17.") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str | None]] = []
        self.last_usage: dict | None = None

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        return self.reply


def _pipeline_stub(reply: str = "The pressure at Deer_Park is 4.17.") -> tuple[Pipeline, _FakeLLM]:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = Settings()
    llm = _FakeLLM(reply)
    pipeline.llm = llm
    return pipeline, llm


def _chunk(code: str) -> CodeChunk:
    return CodeChunk(
        id="c1", space_id="demo", source_id="src1", file_path="ctc device data",
        language="text", symbol_name=None, start_line=1, end_line=1, code=code,
    )


@patch("src.pipeline.fetch_live_readings")
def test_try_live_data_answer_matches_place_and_narrates_readings(mock_fetch) -> None:
    pipeline, llm = _pipeline_stub()
    mock_fetch.return_value = {
        "{notification}:ctc:device:latest:inlet:00-80-F4-2D-32-35:351": {
            "payload": {"pressure": [4.17], "flow": [40.19], "totalizer": [4.9]}
        }
    }

    text, table, tool_trace = pipeline._try_live_data_answer(
        "what is the pressure at Deer_Park", "space1", [_chunk(PLACE_DOC)]
    )

    assert (text, table) == (llm.reply, None)  # no ```table block in this reply -> table is None
    mock_fetch.assert_called_once()
    assert mock_fetch.call_args.args[0] == "space1"
    assert mock_fetch.call_args.args[1] == "cuttack"
    # the context the LLM actually saw includes the extracted reading, not raw JSON
    prompt, system = llm.calls[0]
    assert "pressure=4.17" in prompt
    assert system is not None and "totalizer" in system.lower()
    # the tool-call trace mirrors what was actually fetched/sent, for the pipeline breakdown UI
    assert tool_trace.matched_places == ["Zone_11_Gorakabar"]
    assert tool_trace.redis_keys == ["{notification}:ctc:device:latest:inlet:00-80-F4-2D-32-35:351"]
    assert "pressure=4.17" in tool_trace.context


@patch("src.pipeline.fetch_live_readings")
def test_try_live_data_answer_extracts_a_table_block_from_the_reply(mock_fetch) -> None:
    reply = '```table\n{"columns": ["Place", "Pressure"], "rows": [["Deer_Park", 4.17]]}\n```'
    pipeline, _llm = _pipeline_stub(reply)
    mock_fetch.return_value = {
        "{notification}:ctc:device:latest:inlet:00-80-F4-2D-32-35:351": {
            "payload": {"pressure": [4.17], "flow": [40.19], "totalizer": [4.9]}
        }
    }

    text, table, _tool_trace = pipeline._try_live_data_answer(
        "what is the pressure at Deer_Park", "space1", [_chunk(PLACE_DOC)]
    )

    assert text == ""
    assert table == {"columns": ["Place", "Pressure"], "rows": [["Deer_Park", 4.17]]}


@patch("src.pipeline.fetch_live_readings")
def test_try_live_data_answer_covers_multiple_matched_places_at_once(mock_fetch) -> None:
    pipeline, llm = _pipeline_stub()
    mock_fetch.return_value = {}  # readings content doesn't matter for this assertion

    text, table, tool_trace = pipeline._try_live_data_answer(
        "compare Zone_11_Gorakabar and Zone_16_Killa", "space1", [_chunk(PLACE_DOC)]
    )

    assert (text, table) == (llm.reply, None)
    assert mock_fetch.call_count == 2  # one fetch per matched place
    prompt, _system = llm.calls[0]
    assert "Zone_11_Gorakabar" in prompt and "Zone_16_Killa" in prompt
    assert set(tool_trace.matched_places) == {"Zone_11_Gorakabar", "Zone_16_Killa"}


def test_try_live_data_answer_returns_none_when_no_place_doc_chunk_present() -> None:
    pipeline, _llm = _pipeline_stub()

    result = pipeline._try_live_data_answer(
        "what is the pressure at Deer_Park", "space1", [_chunk("Once upon a time in Ayodhya...")]
    )

    assert result is None


def test_try_live_data_answer_returns_none_when_question_matches_no_place() -> None:
    pipeline, _llm = _pipeline_stub()

    result = pipeline._try_live_data_answer("who is Krishna", "space1", [_chunk(PLACE_DOC)])

    assert result is None


@patch("src.pipeline.fetch_live_readings")
def test_try_live_data_answer_returns_none_when_no_redis_source_available(mock_fetch) -> None:
    pipeline, _llm = _pipeline_stub()
    mock_fetch.return_value = None  # no connector configured / unknown ULB

    result = pipeline._try_live_data_answer(
        "what is the pressure at Deer_Park", "space1", [_chunk(PLACE_DOC)]
    )

    assert result is None


@patch("src.pipeline.fetch_live_readings")
def test_try_live_data_answer_returns_none_on_llm_failure(mock_fetch) -> None:
    pipeline, llm = _pipeline_stub()
    mock_fetch.return_value = {}

    def _raise(*a, **k):
        raise RuntimeError("LLM down")

    llm.complete = _raise

    result = pipeline._try_live_data_answer(
        "what is the pressure at Deer_Park", "space1", [_chunk(PLACE_DOC)]
    )

    assert result is None
