from __future__ import annotations

from src.index.schema import FileSummary
from src.llm_client import LLMClient

MAX_CODE_CHARS_IN_PROMPT = 4000

FILE_SYSTEM_PROMPT = (
    "You summarize source files for a code-search index. Reply with 2-3 sentences "
    "describing the file's purpose. No preamble, no markdown."
)

REPO_SYSTEM_PROMPT = (
    "You summarize a software repository for a code-search index. Reply with one short "
    "paragraph describing what the project does and how it's organized. No preamble."
)

CSV_SYSTEM_PROMPT = (
    "You summarize a CSV dataset for a search index, from its column names and a few "
    "sample rows. Reply with one sentence describing what the data represents. "
    "No preamble, no markdown."
)


class Summarizer:
    """Generates the file- and repo-level summaries that back the routing index."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def summarize_repo(self, readme: str, file_tree: str) -> str:
        prompt = f"README:\n{readme[:MAX_CODE_CHARS_IN_PROMPT]}\n\nFile tree:\n{file_tree}"
        try:
            return self.llm.complete(prompt, system=REPO_SYSTEM_PROMPT).strip()
        except RuntimeError:
            return f"Repository with files:\n{file_tree}"

    def summarize_file(
        self,
        space_id: str,
        source_id: str,
        file_path: str,
        language: str,
        code: str,
        symbols: list[str],
        repo_context: str,
    ) -> FileSummary:
        prompt = (
            f"Repo context: {repo_context}\n\n"
            f"File: {file_path}\nSymbols: {', '.join(symbols) or '(none)'}\n\n"
            f"```\n{code[:MAX_CODE_CHARS_IN_PROMPT]}\n```"
        )
        try:
            summary = self.llm.complete(prompt, system=FILE_SYSTEM_PROMPT).strip()
        except RuntimeError:
            # A single flaky call must not abort ingestion of the whole repo.
            summary = f"{file_path}: {', '.join(symbols) or 'no top-level symbols found'}."
        return FileSummary(
            space_id=space_id,
            source_id=source_id,
            file_path=file_path,
            language=language,
            summary=summary,
            symbols=symbols,
        )

    def summarize_csv(
        self, space_id: str, source_id: str, file_path: str, text: str, sample_rows: int = 5
    ) -> FileSummary:
        """One call per CSV, not per chunk — cheap, and helps routing tell unrelated CSVs
        apart better than raw column names/sample rows alone."""
        lines = text.splitlines()
        header = lines[0] if lines else ""
        sample = "\n".join(lines[1 : 1 + sample_rows])
        prompt = f"Columns: {header}\n\nSample rows:\n{sample}"
        try:
            summary = self.llm.complete(prompt, system=CSV_SYSTEM_PROMPT).strip()
        except RuntimeError:
            summary = f"CSV data with columns: {header}"
        return FileSummary(
            space_id=space_id, source_id=source_id, file_path=file_path,
            language="csv", summary=summary, symbols=[],
        )

    def template_summary(
        self, space_id: str, source_id: str, file_path: str, language: str, text: str
    ) -> FileSummary:
        """No LLM call: prose (PDF/docx/pasted text) is already self-describing, so the
        raw leading text works fine as the routing-embedding target."""
        return FileSummary(
            space_id=space_id,
            source_id=source_id,
            file_path=file_path,
            language=language,
            summary=text[:500].strip(),
            symbols=[],
        )
