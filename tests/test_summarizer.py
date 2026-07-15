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
        "demo", "cli.py", "python", "def main(): ...", ["main"], "A CLI tool"
    )

    assert result == FileSummary(
        repo="demo",
        file_path="cli.py",
        language="python",
        summary="Parses CLI args.",
        symbols=["main"],
    )


def test_summarize_file_falls_back_to_template_on_llm_failure() -> None:
    summarizer = Summarizer(FakeLLM(fail=True))

    result = summarizer.summarize_file(
        "demo", "cli.py", "python", "def main(): ...", ["main"], "A CLI tool"
    )

    assert result.summary == "cli.py: main."


def test_summarize_repo_falls_back_to_file_tree_on_llm_failure() -> None:
    summarizer = Summarizer(FakeLLM(fail=True))

    result = summarizer.summarize_repo(readme="", file_tree="a.py\nb.py")

    assert result == "Repository with files:\na.py\nb.py"
