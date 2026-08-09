from __future__ import annotations

from src.generate.answer import ChartParser


def test_extract_returns_none_when_no_chart_block_present() -> None:
    text, chart = ChartParser().extract("Just a plain answer, no chart here.")

    assert chart is None
    assert text == "Just a plain answer, no chart here."


def test_extract_parses_a_valid_chart_and_strips_it_from_the_text() -> None:
    text = (
        "iPhone 16 has 48MP, iPhone 17 has 48MP too.\n\n"
        "```chart\n"
        '{"title": "Camera", "categories": ["Main"], '
        '"series": [{"name": "iPhone 16", "values": [48]}, {"name": "iPhone 17", "values": [48]}]}\n'
        "```"
    )

    stripped, chart = ChartParser().extract(text)

    assert "```chart" not in stripped
    assert "iPhone 16 has 48MP" in stripped
    assert chart == {
        "title": "Camera",
        "categories": ["Main"],
        "series": [
            {"name": "iPhone 16", "values": [48.0]},
            {"name": "iPhone 17", "values": [48.0]},
        ],
    }


def test_extract_rejects_malformed_json() -> None:
    text = "answer\n```chart\nnot valid json {{{\n```"

    stripped, chart = ChartParser().extract(text)

    assert chart is None
    assert stripped == text  # left untouched — nothing safely parsed to strip


def test_extract_rejects_mismatched_value_and_category_lengths() -> None:
    text = (
        "```chart\n"
        '{"categories": ["A", "B"], "series": [{"name": "X", "values": [1]}]}\n'
        "```"
    )

    _, chart = ChartParser().extract(text)

    assert chart is None


def test_extract_rejects_non_numeric_values() -> None:
    text = (
        "```chart\n"
        '{"categories": ["A"], "series": [{"name": "X", "values": ["not a number"]}]}\n'
        "```"
    )

    _, chart = ChartParser().extract(text)

    assert chart is None


def test_extract_defaults_missing_title_to_empty_string() -> None:
    text = (
        "```chart\n"
        '{"categories": ["A"], "series": [{"name": "X", "values": [1]}]}\n'
        "```"
    )

    _, chart = ChartParser().extract(text)

    assert chart["title"] == ""
