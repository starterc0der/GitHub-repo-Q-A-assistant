from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.index.schema import CodeChunk
from src.llm_client import LLMClient

# Normally one range per bracket ([path:L1-L5]), but models sometimes group several
# ranges for the same file into one bracket ([path:L1-L5, L9-L12]) despite the prompt
# asking for one marker per claim, or — especially for a single-row citation, e.g. one
# CSV row — drop the range entirely and write just [path:L5]. RANGE_PATTERN accepts both
# so a citation to one line still resolves instead of silently matching nothing.
CITATION_PATTERN = re.compile(r"\[([^\]:]+):(L\d+(?:-L\d+)?(?:\s*,\s*L\d+(?:-L\d+)?)*)\]")
RANGE_PATTERN = re.compile(r"L(\d+)(?:-L(\d+))?")
CHART_PATTERN = re.compile(r"```chart\s*\n(.*?)\n```", re.DOTALL)

SYSTEM_PROMPT = (
    "Answer only from the code chunks in the user message — that is your entire world; "
    "never use outside knowledge of similar projects, libraries, or conventions.\n"
    "\n"
    "Rules:\n"
    "1. Every claim must cite its source: [file_path:Lstart-Lend], exact and verbatim from "
    "the chunk headers — one marker per claim, never invent or round a path/line. If you "
    "cannot cite a line, do not write the claim.\n"
    "2. If the chunks do not answer the question, say so and stop — do not pad with unrelated "
    "code. Distinguish: (a) about this repo but not covered here — name the file/function you "
    "would need; (b) not about this repo at all — say so, never answer from general knowledge. "
    "Same if no chunks were given at all. A refusal is a correct answer; a guess is a failure.\n"
    "3. Partial coverage gets a partial answer, with what is unsupported stated explicitly.\n"
    "4. Chunks are excerpts — absence here means unknown, not absent from the repo. Never say "
    "something is missing or does not exist just because it is not in front of you.\n"
    "5. Describe only what the code visibly does — never infer behavior from a name or assumed "
    "convention. Mark any inference explicitly; never blend it into a cited claim.\n"
    "6. Match names — functions, files, identifiers — character for character.\n"
    "7. Only if the question asks for a comparison or graph AND the chunks contain comparable "
    "numeric values, START your answer with a fenced ```chart block (before any prose) of "
    'this exact JSON shape: {"title":"...","categories":["..."],"series":[{"name":"...",'
    '"values":[...]}]}. values must be plain numbers, one per category, same order. Put it '
    "first so it is never cut off by a long explanation. Omit it entirely otherwise — most "
    "questions don't need one.\n"
    "\n"
    "Be concise and concrete, no preamble."
)


@dataclass
class Citation:
    file_path: str
    start_line: int
    end_line: int
    snippet: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 1.0
    chart: dict | None = None


class ChartParser:
    """Extracts an optional ```chart JSON block, validates its shape, strips it from the
    display text. Malformed JSON is treated as no chart, never shown broken."""

    def extract(self, text: str) -> tuple[str, dict | None]:
        match = CHART_PATTERN.search(text)
        if not match:
            return text, None
        chart = self._validate(match.group(1))
        if chart is None:
            return text, None
        return (text[: match.start()] + text[match.end() :]).strip(), chart

    def _validate(self, raw: str) -> dict | None:
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        categories = data.get("categories")
        series = data.get("series")
        if not isinstance(categories, list) or not categories:
            return None
        if not isinstance(series, list) or not series:
            return None
        cleaned_series = []
        for s in series:
            if not isinstance(s, dict) or not isinstance(s.get("name"), str):
                return None
            values = s.get("values")
            if not isinstance(values, list) or len(values) != len(categories):
                return None
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                return None
            cleaned_series.append({"name": s["name"], "values": [float(v) for v in values]})
        title = data.get("title")
        return {
            "title": title if isinstance(title, str) else "",
            "categories": [str(c) for c in categories],
            "series": cleaned_series,
        }


class CitationParser:
    """Extracts [file:Lstart-Lend] markers from LLM output and matches them to chunks."""

    def parse(self, text: str, chunks: list[CodeChunk]) -> list[Citation]:
        citations = []
        for file_path, ranges in CITATION_PATTERN.findall(text):
            for start_str, end_str in RANGE_PATTERN.findall(ranges):
                start_line = int(start_str)
                end_line = int(end_str) if end_str else start_line
                chunk = self._find_chunk(file_path, start_line, chunks)
                if chunk is None:
                    continue
                citations.append(
                    Citation(
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        snippet=self._extract_snippet(chunk, start_line, end_line),
                    )
                )
        return citations

    def _find_chunk(
        self, file_path: str, start_line: int, chunks: list[CodeChunk]
    ) -> CodeChunk | None:
        for chunk in chunks:
            if chunk.file_path == file_path and chunk.start_line <= start_line <= chunk.end_line:
                return chunk
        return None

    def _extract_snippet(self, chunk: CodeChunk, start_line: int, end_line: int) -> str:
        lines = chunk.code.splitlines()
        offset = max(0, start_line - chunk.start_line)
        count = max(1, end_line - start_line + 1)
        return "\n".join(lines[offset : offset + count])


class AnswerGenerator:
    """Turns retrieved chunks + a question into a cited, grounded answer."""

    def __init__(
        self, llm: LLMClient, citation_parser: CitationParser, chart_parser: ChartParser | None = None
    ):
        self.llm = llm
        self.citation_parser = citation_parser
        self.chart_parser = chart_parser or ChartParser()

    def answer(
        self, question: str, chunks: list[CodeChunk], history: list[tuple[str, str]] | None = None
    ) -> Answer:
        prompt = self.build_prompt(question, chunks)
        text = self.llm.complete(prompt, system=SYSTEM_PROMPT, history=history)
        text, chart = self.chart_parser.extract(text)
        citations = self.citation_parser.parse(text, chunks)
        confidence = 1.0 if citations else 0.3
        return Answer(text=text, citations=citations, confidence=confidence, chart=chart)

    def build_prompt(self, question: str, chunks: list[CodeChunk]) -> str:
        context = "\n\n".join(
            f"[{chunk.file_path}:L{chunk.start_line}-L{chunk.end_line}]\n{chunk.code}"
            for chunk in chunks
        )
        return f"Code chunks:\n{context}\n\nQuestion: {question}\n\nAnswer:"
