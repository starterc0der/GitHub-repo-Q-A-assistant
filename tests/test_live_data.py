from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from src.config import settings
from src.connectors.live_data import (
    DevicePoint,
    build_live_data_context,
    extract_table,
    fetch_live_readings,
    find_matching_places,
    parse_place_blocks,
    redis_prefix_for_ulb,
)
from src.crypto import encrypt
from src.db import connect, init_db, new_id, now

SAMPLE_TEXT = """\
- device_id=00-80-F4-2A-D4-EE, pid=EE4, sub_place=Ambika Lane

## Zone_11_Gorakabar
ULB: cuttack
Inlet: device_id=00-80-F4-2D-32-35, pid=351, location=ESR
Outlets (sub-places):
- device_id=00-80-F4-2D-32-35, pid=354, sub_place=Bauri_Sahi
- device_id=00-80-F4-2D-32-35, pid=351, sub_place=Deer_Park

## Zone_16_Killa
ULB: cuttack
Inlet: device_id=A8-74-1D-16-73-10, pid=101, location=ESR
Outlets (sub-places):
- device_id=A8-74-1D-16-73-10, pid=102, sub_place=KILLA_SUB_DMA_2
"""


def test_parse_place_blocks_finds_every_place_in_a_multi_place_chunk() -> None:
    places = parse_place_blocks(SAMPLE_TEXT)

    assert [p.place_name for p in places] == ["Zone_11_Gorakabar", "Zone_16_Killa"]
    assert places[0].ulb == "cuttack"
    assert places[0].points == [
        DevicePoint("inlet", "00-80-F4-2D-32-35", "351", "ESR"),
        DevicePoint("outlet", "00-80-F4-2D-32-35", "354", "Bauri_Sahi"),
        DevicePoint("outlet", "00-80-F4-2D-32-35", "351", "Deer_Park"),
    ]


def test_parse_place_blocks_returns_empty_for_unrelated_text() -> None:
    assert parse_place_blocks("Once upon a time in Ayodhya...") == []


def test_find_matching_places_matches_on_place_name() -> None:
    places = parse_place_blocks(SAMPLE_TEXT)

    matches = find_matching_places("what is the pressure in Zone_11_Gorakabar", places)

    assert [p.place_name for p in matches] == ["Zone_11_Gorakabar"]


def test_find_matching_places_matches_a_single_word_place_name_exactly() -> None:
    # A one-token place name (e.g. a real BBSR/Puri place like "Acharyavihar") scores
    # only 1 on token overlap — the same as a generic noise word like "zone" — but
    # unlike "zone", it's the ENTIRE name, so it must still match, not get filtered as noise.
    text = "## Acharyavihar\nULB: bhubaneswar\nInlet: device_id=AA-BB, pid=1, location=ESR\n"
    places = parse_place_blocks(text)

    matches = find_matching_places("what is the pressure at Acharyavihar", places)

    assert [p.place_name for p in matches] == ["Acharyavihar"]


def test_find_matching_places_does_not_strong_match_on_a_shared_generic_location() -> None:
    # Regression: "ESR" is a single-token SUB-PLACE location shared by nearly every
    # place's inlet in this dataset — unlike a single-token PLACE name (see the test
    # above), it must NOT get the full-subset-match strong-score exemption, or a
    # question merely containing the word "ESR" (e.g. quoting "ESR (inlet)" back from a
    # prior answer) would strong-match almost the entire corpus.
    text = (
        "## Zone_08_Krushak_Bazar\nULB: cuttack\nInlet: device_id=00-00-54-5F-05-FC, pid=1, location=ESR\n"
        "## Zone_09_Sewerage_Store\nULB: cuttack\nInlet: device_id=A8-74-1D-10-F0-EB, pid=2, location=ESR\n"
    )
    places = parse_place_blocks(text)

    matches = find_matching_places("show ESR (inlet) for Zone_08_Krushak_Bazar", places)

    assert [p.place_name for p in matches] == ["Zone_08_Krushak_Bazar"]


def test_find_matching_places_ignores_stopwords_in_overlap_scoring() -> None:
    # Regression: a real BBSR sub-place named "Road_6_to_Road_8" false-matched an
    # unrelated question about "Zone 08" purely because "to" (a boilerplate connector
    # word, present in both) plus a coincidental "8" digit together crossed the
    # score>=2 threshold — neither alone would have. Stopwords like "to" must never
    # count toward the overlap score at all.
    text = (
        "## Unrelated_Place\nULB: bhubaneswar\nInlet: device_id=AA-11-22-33-44-55, "
        "pid=1, location=UGR-1\n"
        "- device_id=AA-11-22-33-44-55, pid=2, sub_place=Road_6_to_Road_8\n"
    )
    places = parse_place_blocks(text)

    matches = find_matching_places("what is the pressure at Zone 08 from august 5 to august 10", places)

    assert matches == []


def test_find_matching_places_does_not_strong_match_on_a_shared_generic_inlet_label() -> None:
    # Regression: "UGR-1" (like "ESR") is a generic inlet-location label shared by many
    # unrelated zones — but unlike "ESR" it tokenizes to TWO words ("ugr","1"), so a bare
    # mention of it already scored 2 on its own, false-matching every other zone whose
    # inlet also happens to sit at a UGR-1 — confirmed live: a question about Malisahi's
    # own UGR-1 inlet pulled in Zone_46_Sartol/Zone_15_Killa_Head_Works/Zone_27_Ranihat
    # purely because they're ALSO on a UGR-1, nothing else in the question overlapped.
    text = (
        "## Zone_46_Sartol\nULB: cuttack\nInlet: device_id=A8-74-1D-16-71-96, pid=961, location=UGR-1\n"
        "- device_id=A8-74-1D-16-71-96, pid=962, sub_place=Nua_Pada\n"
        "## Malisahi\nULB: bhubaneswar\nInlet: device_id=2C-CF-67-12-75-F9, pid=F92, location=UGR-1\n"
        "- device_id=2C-CF-67-12-75-F9, pid=F92, sub_place=Lalchand_Line\n"
    )
    places = parse_place_blocks(text)

    matches = find_matching_places("give me the latest sensor readings for the Malisahi UGR-1 inlet", places)

    assert [p.place_name for p in matches] == ["Malisahi"]


def test_find_matching_places_ignores_enumeration_digits_in_sub_place_names() -> None:
    # Regression: "Line" (from the question's own "Lalchand_Line") plus a coincidental
    # "1" (from "UGR-1" elsewhere in the question) crossed the threshold against an
    # unrelated "Master_Line_for_SDMA_1_2_3" sub-place purely via its enumeration digits.
    text = (
        "## Zone_29_Khan_Nagar\nULB: cuttack\nInlet: device_id=00-80-F4-2A-D4-EE, pid=EE1, location=ESR\n"
        "- device_id=00-80-F4-2A-D4-EE, pid=EE1, sub_place=Master_Line_for_SDMA_1_2_3\n"
        "## Malisahi\nULB: bhubaneswar\nInlet: device_id=2C-CF-67-12-75-F9, pid=F92, location=UGR-1\n"
        "- device_id=2C-CF-67-12-75-F9, pid=F92, sub_place=Lalchand_Line\n"
    )
    places = parse_place_blocks(text)

    matches = find_matching_places(
        "give me the latest sensor readings for the Malisahi UGR-1 inlet and Lalchand_Line", places
    )

    assert [p.place_name for p in matches] == ["Malisahi"]


def test_find_matching_places_matches_on_sub_place_name() -> None:
    places = parse_place_blocks(SAMPLE_TEXT)

    matches = find_matching_places("show me the latest reading at Deer_Park", places)

    assert [p.place_name for p in matches] == ["Zone_11_Gorakabar"]


def test_find_matching_places_matches_on_device_id() -> None:
    places = parse_place_blocks(SAMPLE_TEXT)

    matches = find_matching_places("status of device A8-74-1D-16-73-10 please", places)

    assert [p.place_name for p in matches] == ["Zone_16_Killa"]


def test_find_matching_places_supports_asking_about_more_than_one_place() -> None:
    places = parse_place_blocks(SAMPLE_TEXT)

    matches = find_matching_places("compare Zone_11_Gorakabar and Zone_16_Killa", places)

    assert {p.place_name for p in matches} == {"Zone_11_Gorakabar", "Zone_16_Killa"}


def test_find_matching_places_tolerates_a_one_letter_typo_in_a_place_name() -> None:
    # Regression: "give me latest data of malasahi" (a real user typo for "Malisahi")
    # scored 0 under plain exact-token overlap and fell all the way through to a
    # generic "couldn't find that" answer instead of the live-data lookup.
    text = "## Malisahi\nULB: bhubaneswar\nInlet: device_id=AA-BB, pid=1, location=UGR-1\n"
    places = parse_place_blocks(text)

    matches = find_matching_places("give me latest data of malasahi", places)

    assert [p.place_name for p in matches] == ["Malisahi"]


def test_find_matching_places_does_not_fuzzy_match_short_unrelated_words() -> None:
    # The fuzzy path only applies to longer tokens — "zone" (4 letters) must not
    # loosely match an unrelated short word and manufacture a false-positive place hit.
    places = parse_place_blocks(SAMPLE_TEXT)

    matches = find_matching_places("what does zore mean", places)

    assert matches == []


def test_find_matching_places_returns_empty_list_when_nothing_overlaps() -> None:
    places = parse_place_blocks(SAMPLE_TEXT)

    assert find_matching_places("who is Krishna", places) == []


def test_redis_prefix_for_ulb_known_and_unknown() -> None:
    assert redis_prefix_for_ulb("cuttack") == "ctc"
    assert redis_prefix_for_ulb("Cuttack") == "ctc"  # case-insensitive
    assert redis_prefix_for_ulb("bhubaneswar") == "bbsr"
    assert redis_prefix_for_ulb("puri") == "puri"
    assert redis_prefix_for_ulb("nimapada") is None  # not onboarded yet


def test_extract_table_parses_a_valid_table_block() -> None:
    text = '```table\n{"columns": ["Place", "Pressure"], "rows": [["Deer_Park", 4.17]]}\n```'

    remaining, table = extract_table(text)

    assert remaining == ""
    assert table == {"columns": ["Place", "Pressure"], "rows": [["Deer_Park", 4.17]]}


def test_extract_table_strips_the_block_and_keeps_surrounding_prose() -> None:
    text = 'before\n```table\n{"columns": ["A"], "rows": [[1]]}\n```\nafter'

    remaining, table = extract_table(text)

    assert remaining == "before\n\nafter"
    assert table is not None


def test_extract_table_returns_none_for_malformed_json() -> None:
    text = "```table\nnot json\n```"

    remaining, table = extract_table(text)

    # stripped, not left showing broken JSON — see TableParser.extract's comment.
    assert remaining == ""
    assert table is None


def test_extract_table_returns_none_for_a_row_with_the_wrong_column_count() -> None:
    text = '```table\n{"columns": ["A", "B"], "rows": [[1]]}\n```'

    _remaining, table = extract_table(text)

    assert table is None


def test_extract_table_returns_none_with_no_block_present() -> None:
    assert extract_table("just plain text") == ("just plain text", None)


@pytest.fixture(autouse=True)
def _isolated_db_and_key(tmp_path, monkeypatch):
    db_path = str(tmp_path / "app.db")
    monkeypatch.setattr(settings, "db_path", db_path)
    monkeypatch.setattr(settings, "connector_encryption_key", Fernet.generate_key().decode())
    init_db(db_path)


def _make_space_with_redis_connector(**overrides) -> str:
    space_id = new_id()
    fields = dict(
        host="redis.example.internal", port=8100, username="testuser", password="testpass123",
        db_index=0, tls=0,
    )
    fields.update(overrides)
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO spaces (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (space_id, "Demo", now(), now()),
        )
        conn.execute(
            "INSERT INTO connectors (id, space_id, kind, name, host, port, database, "
            "username, encrypted_password, db_index, tls, ssl, status, last_tested_at, created_at) "
            "VALUES (?, ?, 'redis', 'watco redis', ?, ?, NULL, ?, ?, ?, ?, 0, 'connected', ?, ?)",
            (
                new_id(), space_id, fields["host"], fields["port"], fields["username"],
                encrypt(fields["password"], settings.connector_encryption_key),
                fields["db_index"], fields["tls"], now(), now(),
            ),
        )
    return space_id


_POINTS = [DevicePoint("inlet", "00-80-F4-2D-32-35", "351", "ESR")]


def test_fetch_live_readings_returns_none_without_a_known_ulb_prefix() -> None:
    space_id = _make_space_with_redis_connector()

    assert fetch_live_readings(space_id, "bhubaneswar", _POINTS) is None


def test_fetch_live_readings_returns_none_without_a_configured_connector() -> None:
    space_id = new_id()
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO spaces (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (space_id, "Demo", now(), now()),
        )

    assert fetch_live_readings(space_id, "cuttack", _POINTS) is None


@patch("src.connectors.live_data.redis.Redis")
def test_fetch_live_readings_builds_the_hash_tagged_key_and_parses_json(mock_redis_cls: MagicMock) -> None:
    space_id = _make_space_with_redis_connector()
    mock_client = MagicMock()
    mock_client.mget.return_value = ['{"payload": {"pressure": [4.17], "flow": [40.19], "totalizer": [4.9]}}']
    mock_redis_cls.return_value = mock_client

    readings = fetch_live_readings(space_id, "cuttack", _POINTS)

    expected_key = "{notification}:ctc:device:latest:inlet:00-80-F4-2D-32-35:351"
    assert mock_client.mget.call_args.args[0] == [expected_key]
    assert readings[expected_key]["payload"]["pressure"] == [4.17]
    # credentials from the stored connector reached the client, not placeholders
    assert mock_redis_cls.call_args.kwargs["username"] == "testuser"
    assert mock_redis_cls.call_args.kwargs["password"] == "testpass123"


@patch("src.connectors.live_data.redis.Redis")
def test_fetch_live_readings_missing_key_reads_as_none_not_a_crash(mock_redis_cls: MagicMock) -> None:
    space_id = _make_space_with_redis_connector()
    mock_client = MagicMock()
    mock_client.mget.return_value = [None]
    mock_redis_cls.return_value = mock_client

    readings = fetch_live_readings(space_id, "cuttack", _POINTS)

    assert readings == {"{notification}:ctc:device:latest:inlet:00-80-F4-2D-32-35:351": None}


@patch("src.connectors.live_data.redis.Redis")
def test_fetch_live_readings_returns_none_on_connection_failure(mock_redis_cls: MagicMock) -> None:
    space_id = _make_space_with_redis_connector()
    mock_client = MagicMock()
    mock_client.mget.side_effect = ConnectionError("boom")
    mock_redis_cls.return_value = mock_client

    assert fetch_live_readings(space_id, "cuttack", _POINTS) is None


def test_build_live_data_context_extracts_single_element_arrays() -> None:
    place = parse_place_blocks(SAMPLE_TEXT)[0]
    key = "{notification}:ctc:device:latest:inlet:00-80-F4-2D-32-35:351"
    readings = {key: {"payload": {"pressure": [0], "flow": [191.14], "totalizer": [1.5491]}}}

    context = build_live_data_context([place], readings)

    assert "ESR (inlet) (device_id 00-80-F4-2D-32-35, pid 351): pressure=0, flow=191.14, totalizer=1.5491" in context
    assert "Bauri_Sahi (outlet) (device_id 00-80-F4-2D-32-35, pid 354): reading unavailable" in context
    assert "Deer_Park (outlet)" in context and "reading unavailable" in context  # different key, not fetched here


def test_build_live_data_context_shows_level_only_on_inlet_and_chlorine_once_overall() -> None:
    place = parse_place_blocks(SAMPLE_TEXT)[0]
    inlet_key = "{notification}:ctc:device:latest:inlet:00-80-F4-2D-32-35:351"
    outlet1_key = "{notification}:ctc:device:latest:outlet:00-80-F4-2D-32-35:354"
    outlet2_key = "{notification}:ctc:device:latest:outlet:00-80-F4-2D-32-35:351"
    readings = {
        inlet_key: {"payload": {"pressure": [0], "flow": [1], "totalizer": [1], "level": 2.66}},
        outlet1_key: {"payload": {"pressure": [1], "flow": [2], "totalizer": [3], "chlorine": [0.19]}},
        outlet2_key: {"payload": {"pressure": [4], "flow": [5], "totalizer": [6], "chlorine": [0.19]}},
    }

    context = build_live_data_context([place], readings)

    assert "ESR (inlet) (device_id 00-80-F4-2D-32-35, pid 351): pressure=0, flow=1, totalizer=1, level=2.66" in context
    assert "Bauri_Sahi (outlet)" in context and "level" not in context.split("Bauri_Sahi")[1].split("\n")[0]
    assert context.count("Chlorine (overall for this place") == 1
    assert "Chlorine (overall for this place, same across every outlet): 0.19" in context


def test_build_live_data_context_covers_multiple_places() -> None:
    places = parse_place_blocks(SAMPLE_TEXT)

    context = build_live_data_context(places, {})

    assert "Place: Zone_11_Gorakabar" in context
    assert "Place: Zone_16_Killa" in context
