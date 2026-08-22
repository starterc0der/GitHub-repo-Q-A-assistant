from __future__ import annotations

from src.generate.decomposer import (
    DecomposeResult,
    FLAG_BROAD,
    MAX_HOPS,
    META_MARKER,
    PARALLEL_MARKER,
    ROUTE_GRAMMAR,
    SEQUENTIAL_MARKER,
    SINGLE_SENTINEL,
    parse_route_header,
)
from src.llm_client import LLMClient

SYSTEM_PROMPT = (
    "You classify and resolve the latest question in a conversation before retrieval "
    "runs, so the pipeline can pick the right strategy for it — the same job "
    "QueryDecomposer does on turn 1, extended to also resolve history.\n"
    "First: is the latest question asking specifically about THIS CONVERSATION's own "
    "history — what was discussed, who was mentioned, a recap of the chat itself — or a "
    "greeting/capability question about the assistant? A request to list, name, or "
    "summarize the people/topics/entities already discussed IS this case (e.g. \"list "
    "everyone we've talked about\", \"who have I asked you about\") — classify it "
    f"{META_MARKER} even when many names would be involved and even when the "
    "conversation history you're given is long or detailed; a rich history is not the "
    "same thing as a question about the source material. A request to summarize or "
    "overview the DOCUMENT/SOURCE MATERIAL itself IS a different case, even if it uses "
    "the word \"summary\" or \"overview\" — that is a normal document question about the "
    f"source, classify it {FLAG_BROAD} below instead. If it genuinely is about the "
    f"conversation or the assistant itself, reply with exactly: {META_MARKER}\n"
    f"Otherwise, reply with:\n{ROUTE_GRAMMAR}\n"
    "In every case below, resolve pronouns and implicit references against the "
    "conversation history — retrieval only ever sees what you write here, never the raw "
    "history. Preserve the user's intent exactly — do not answer the question, expand "
    "its scope, or add assumptions. If the latest question is a near-repeat of an "
    "EARLIER question in this conversation (not a follow-up building on the most recent "
    "answer), rewrite it standalone exactly as broad as that earlier question was asked "
    "— do not narrow it to whichever specific detail the most recent answer happened to "
    "focus on (e.g. re-asking \"give me the latest data\" after an answer that only "
    "covered chlorine must still ask for ALL of the data, not just chlorine again).\n"
    f"If line 2 is {SINGLE_SENTINEL}: one more line — the question rewritten as a "
    "standalone question.\n"
    f"If line 2 is {PARALLEL_MARKER}: list at most 3 self-contained sub-questions, one "
    "per line.\n"
    f"If line 2 is {SEQUENTIAL_MARKER}: list 2 or 3 lines, one per hop — the first hop's "
    "self-contained question, then each later hop's question written with a {hop1}, "
    "{hop2}, etc. placeholder wherever it depends on that earlier hop's answer. Most "
    "chains are only 2 hops.\n"
    "Do not add numbering or preamble."
)


class StandaloneRewriter:
    """The turn-2+ half of query routing (see QueryDecomposer for turn 1) — one cheap LLM
    call that resolves history-dependent references ("there", "it") into self-contained
    question(s) AND classifies the same signals QueryDecomposer does (mode, is_meta,
    is_broad, wants_chart) via the same shared grammar (see ROUTE_GRAMMAR), so a question
    is routed identically regardless of which turn it's asked on.

    Skipped entirely when history is empty — that's QueryDecomposer's job instead — so a
    turn-1 question never pays for two classification calls."""

    def __init__(self, llm: LLMClient, max_subquestions: int):
        self.llm = llm
        self.max_subquestions = max_subquestions

    def rewrite(self, question: str, history: list[tuple[str, str]]) -> DecomposeResult:
        if not history:
            return DecomposeResult("single", [question])
        transcript = "\n".join(f"{role}: {content}" for role, content in history)
        prompt = f"Conversation so far:\n{transcript}\n\nLatest question: {question}"
        try:
            reply = self.llm.complete(prompt, system=SYSTEM_PROMPT).strip()
        except RuntimeError:
            return DecomposeResult("single", [question])
        if not reply:
            return DecomposeResult("single", [question])

        lines = [p for p in (line.strip("-•* \t") for line in reply.splitlines()) if p]
        if not lines:
            return DecomposeResult("single", [question])
        if lines[0].strip().upper() == META_MARKER:
            return DecomposeResult("single", [question], is_meta=True)

        is_broad, wants_chart, wants_live_data, wants_report, mode, rest = parse_route_header(lines)
        if mode == SEQUENTIAL_MARKER:
            hops = rest[:MAX_HOPS]
            if len(hops) > 1:
                return DecomposeResult(
                    "sequential", hops, is_broad=is_broad, wants_chart=wants_chart,
                    wants_live_data=wants_live_data, wants_report=wants_report,
                )
        elif mode == PARALLEL_MARKER:
            parts = rest[: self.max_subquestions]
            if len(parts) > 1:
                return DecomposeResult(
                    "parallel", parts, is_broad=is_broad, wants_chart=wants_chart,
                    wants_live_data=wants_live_data, wants_report=wants_report,
                )
        # SINGLE (or anything unrecognized): one more line — the rewritten standalone
        # question — falls back to the raw question if the model omitted it.
        standalone = rest[0] if rest else question
        return DecomposeResult(
            "single", [standalone], is_broad=is_broad, wants_chart=wants_chart,
            wants_live_data=wants_live_data, wants_report=wants_report,
        )
