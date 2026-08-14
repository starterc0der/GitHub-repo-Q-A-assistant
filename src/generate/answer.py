from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

from src.index.schema import CodeChunk
from src.llm_client import LLMClient

CHART_PATTERN = re.compile(r"```chart\s*\n(.*?)\n```", re.DOTALL)

SYSTEM_PROMPT = (
    "Answer only from the source material in the user message — that is your entire world; "
    "never use outside knowledge of similar projects, libraries, or conventions.\n"
    "\n"
    "Rules:\n"
    "1. If the source material does not answer the question, say so and stop — do not pad "
    "with unrelated content. Distinguish: (a) about this repo but not covered here — name the "
    "file/function you would need; (b) not about this repo at all — say so, never answer from "
    "general knowledge. Same if no source material was given at all. A refusal is a correct "
    "answer; a guess is a failure.\n"
    "2. Partial coverage gets a partial answer, with what is unsupported stated explicitly.\n"
    "3. The source material is excerpts — absence here means unknown, not absent from the "
    "repo. Never say something is missing or does not exist just because it is not in front "
    "of you.\n"
    "4. Describe only what is visibly stated — never infer behavior from a name or assumed "
    "convention. Mark any inference explicitly, never blend it into a stated fact.\n"
    "5. Match names — functions, files, identifiers — character for character.\n"
    "6. Only if the question asks for a comparison or graph AND the source material contains "
    "comparable numeric values, START your answer with a fenced ```chart block (before any "
    'prose) of this exact JSON shape: {"title":"...","categories":["..."],"series":[{"name":'
    '"...","values":[...]}]}. values must be plain numbers, one per category, same order. Put '
    "it first so it is never cut off by a long explanation. Omit it entirely otherwise — most "
    "questions don't need one.\n"
    "\n"
    "Do not cite file paths or line numbers, and never refer to \"chunks\", \"excerpts\", "
    "\"the provided text/context\", or how this material reached you — the reader never sees "
    "any of that. Answer in plain, natural prose, as if you simply knew the answer. Be concise "
    "and concrete, no preamble."
)


@dataclass
class Answer:
    text: str
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


class AnswerGenerator:
    """Turns retrieved chunks + a question into a grounded answer."""

    def __init__(self, llm: LLMClient, chart_parser: ChartParser | None = None):
        self.llm = llm
        self.chart_parser = chart_parser or ChartParser()

    def answer(
        self, question: str, chunks: list[CodeChunk], history: list[tuple[str, str]] | None = None,
        insufficient: list[str] | None = None,
    ) -> Answer:
        prompt = self.build_prompt(question, chunks, insufficient)
        text = self.llm.complete(prompt, system=SYSTEM_PROMPT, history=history)
        return self.finalize(text)

    def answer_stream(
        self, question: str, chunks: list[CodeChunk], history: list[tuple[str, str]] | None = None,
        insufficient: list[str] | None = None,
    ) -> Iterator[str]:
        """Yields raw text deltas as they arrive. Chart extraction needs the complete
        text (it's a fenced block, not parseable mid-stream), so callers should
        accumulate the deltas and call finalize() once the stream ends."""
        prompt = self.build_prompt(question, chunks, insufficient)
        yield from self.llm.stream(prompt, system=SYSTEM_PROMPT, history=history)

    def finalize(self, text: str) -> Answer:
        text, chart = self.chart_parser.extract(text)
        return Answer(text=text, chart=chart)

    def build_prompt(
        self, question: str, chunks: list[CodeChunk], insufficient: list[str] | None = None
    ) -> str:
        context = "\n\n".join(
            f"[{chunk.file_path}:L{chunk.start_line}-L{chunk.end_line}]\n{chunk.code}"
            for chunk in chunks
        )
        # Named explicitly rather than left for the model to notice mid-answer: retrieval
        # already knows which part of a compound question came back empty (each
        # sub-question's own rerank gate), so state it as a fact instead of hoping the
        # model discovers the gap in its own writing.
        note = ""
        if insufficient:
            parts = "; ".join(f'"{q}"' for q in insufficient)
            note = (
                f"\n\nNote: no supporting evidence was found for: {parts}. State that gap "
                "plainly, in your own words — never as \"the chunks/excerpts don't mention...\"."
            )
        return f"Source material:\n{context}\n\nQuestion: {question}{note}\n\nAnswer:"
