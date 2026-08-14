from __future__ import annotations

from src.index.schema import FileSummary
from src.ingest.summarizer import Summarizer


class FakeLLM:
    def __init__(self, response: str | None = None, fail: bool = False):
        self.response = response
        self.fail = fail

    def complete(self, prompt: str, system: str | None = None) -> str:
        if self.fail:
            raise RuntimeError("ollama unreachable")
        return self.response


def test_summarize_file_uses_llm_response() -> None:
    summarizer = Summarizer(FakeLLM(response=" Parses CLI args. "))

    result = summarizer.summarize_file(
        "demo", "src1", "cli.py", "python", "def main(): ...", ["main"], "A CLI tool"
    )

    assert result == FileSummary(
        space_id="demo",
        source_id="src1",
        file_path="cli.py",
        language="python",
        summary="Parses CLI args.",
        symbols=["main"],
    )


def test_summarize_file_falls_back_to_template_on_llm_failure() -> None:
    summarizer = Summarizer(FakeLLM(fail=True))

    result = summarizer.summarize_file(
        "demo", "src1", "cli.py", "python", "def main(): ...", ["main"], "A CLI tool"
    )

    assert result.summary == "cli.py: main."


def test_summarize_repo_falls_back_to_file_tree_on_llm_failure() -> None:
    summarizer = Summarizer(FakeLLM(fail=True))

    result = summarizer.summarize_repo(readme="", file_tree="a.py\nb.py")

    assert result == "Repository with files:\na.py\nb.py"


def test_template_summary_skips_the_llm_and_truncates() -> None:
    summarizer = Summarizer(FakeLLM(fail=True))

    result = summarizer.template_summary("demo", "src1", "notes.txt", "text", "x" * 2500)

    assert len(result.summary) == 2000
    assert result.symbols == []


def test_summarize_csv_uses_llm_response_and_sets_csv_language() -> None:
    summarizer = Summarizer(FakeLLM(response="Phone specs: name and price."))

    result = summarizer.summarize_csv(
        "demo", "src1", "phones.csv", "Name,Price\niPhone 16,799\niPhone 17,899\n"
    )

    assert result.summary == "Phone specs: name and price."
    assert result.language == "csv"
    assert result.symbols == []


def test_summarize_csv_falls_back_to_header_on_llm_failure() -> None:
    summarizer = Summarizer(FakeLLM(fail=True))

    result = summarizer.summarize_csv("demo", "src1", "phones.csv", "Name,Price\niPhone 16,799\n")

    assert result.summary == "CSV data with columns: Name,Price"
