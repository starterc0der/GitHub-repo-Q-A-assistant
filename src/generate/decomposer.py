from __future__ import annotations

from dataclasses import dataclass

from src.llm_client import LLMClient

SINGLE_SENTINEL = "SINGLE"
PARALLEL_MARKER = "PARALLEL"
SEQUENTIAL_MARKER = "SEQUENTIAL"
META_MARKER = "META"
FLAG_BROAD = "BROAD"
FLAG_CHART = "CHART"
MAX_HOPS = 3  # caps latency/cost the same way the retry pass caps itself at one attempt

# Shared by both QueryDecomposer (turn 1, no history) and StandaloneRewriter (turn 2+,
# has history) so a question gets classified the same way regardless of when in the
# conversation it's asked — only the SINGLE-mode payload differs (decompose reuses the
# input question verbatim; rewrite must emit its own pronoun-resolved line), everything
# else is one shared grammar.
ROUTE_GRAMMAR = (
    "Line 1: space-separated flags from {broad} (needs a wide summary/overview across "
    "much of the source, not one specific fact) and {chart} (asks for a comparison, "
    "graph, or chart). Write NONE if neither applies.\n"
    "Line 2: {single} if it is genuinely one question; {parallel} if it is multiple "
    "INDEPENDENT questions (different topics joined by \"and\", or a list of separate "
    "asks, each answerable on its own without the others); {sequential} if answering a "
    "later part REQUIRES the answer to an earlier part first — you could not even search "
    "for it without already knowing that answer (e.g. \"which department had the most "
    "failures, and what policy caused them?\" — you can't search for the policy until you "
    "know which department)."
).format(broad=FLAG_BROAD, chart=FLAG_CHART, single=SINGLE_SENTINEL, parallel=PARALLEL_MARKER, sequential=SEQUENTIAL_MARKER)

DECOMPOSE_SYSTEM_PROMPT = (
    "You classify a question before retrieval runs, so the pipeline can pick the right "
    "strategy for it.\n"
    "First: is this question about THIS ASSISTANT itself — a greeting, thanks, or a "
    "question about what the assistant can do — rather than about the source material? "
    "A request to summarize or overview the DOCUMENT/SOURCE MATERIAL is NOT this case, "
    f"even if it uses the word \"summary\" or \"overview\" — classify it {FLAG_BROAD} "
    f"below instead. If it genuinely is about the assistant itself, reply with exactly: "
    f"{META_MARKER}\n"
    f"Otherwise, reply with:\n{ROUTE_GRAMMAR}\n"
    f"If line 2 is {PARALLEL_MARKER}: list at most 3 self-contained sub-questions, one "
    "per line, resolving pronouns like \"it\"/\"that\" against the original question.\n"
    f"If line 2 is {SEQUENTIAL_MARKER}: list 2 or 3 lines, one per hop — the first hop's "
    "self-contained question, then each later hop's question written with a {hop1}, "
    "{hop2}, etc. placeholder wherever it depends on that earlier hop's answer (e.g. "
    "\"What policy caused failures in {hop1}?\"). Most chains are only 2 hops.\n"
    f"If line 2 is {SINGLE_SENTINEL}: no further lines.\n"
    "Do not answer the question, do not add numbering or preamble."
)


def parse_route_header(lines: list[str]) -> tuple[bool, bool, str, list[str]]:
    """Parses the shared flags-line + mode-line header (see ROUTE_GRAMMAR) — returns
    (is_broad, wants_chart, mode, remaining lines after the header). Shared by
    QueryDecomposer and StandaloneRewriter so both interpret the same grammar
    identically; each caller still owns what the remaining lines mean for SINGLE mode."""
    if not lines:
        return False, False, SINGLE_SENTINEL, []
    flags = set(lines[0].strip().upper().split())
    is_broad, wants_chart = FLAG_BROAD in flags, FLAG_CHART in flags
    if len(lines) < 2:
        return is_broad, wants_chart, SINGLE_SENTINEL, []
    return is_broad, wants_chart, lines[1].strip().upper(), lines[2:]


@dataclass
class DecomposeResult:
    mode: str  # "single" | "parallel" | "sequential"
    sub_questions: list[str]
    is_meta: bool = False
    is_broad: bool = False
    wants_chart: bool = False


HOP_RESOLVE_SYSTEM_PROMPT = (
    "You resolve one step of a multi-hop question. You are given INPUT QUESTION, INPUT "
    "CONTEXT (a raw passage retrieved as evidence for the input question — it may be "
    "noisy or only partially relevant), and NEXT HOP TEMPLATE (the next question, "
    "containing a placeholder in curly braces, e.g. {hop1}, standing in for the input "
    "question's answer).\n"
    "Extract the short, direct answer to INPUT QUESTION from INPUT CONTEXT — typically a "
    "name, term, or short phrase, not a full sentence or quote. Substitute it for the "
    "placeholder in NEXT HOP TEMPLATE and reply with only the resulting question, "
    "nothing else.\n"
    "If INPUT CONTEXT does not contain a clear answer to INPUT QUESTION, reply with "
    "exactly: UNRESOLVED"
)

RETRY_REWRITE_SYSTEM_PROMPT = (
    "A sub-question from a larger question found no supporting evidence in the source "
    "material. Before rewriting, identify what SPECIFIC TYPE of evidence would actually "
    "answer it — a root cause, an incident report, a name, a date, a policy or rule, a "
    "specific numeric value, etc. — not just a differently-worded restatement of the same "
    "question. The original phrasing may be too abstract or indirect for how the source "
    "text actually describes this (e.g. a comparison sub-question often finds more by "
    "asking directly about the other side of the comparison than about the comparison "
    "itself; \"why did X fail\" often finds more by searching for the specific root cause, "
    "incident report, or failure record than by re-asking \"why\").\n"
    "Reply with only the rewritten, self-contained question targeting that evidence type — "
    "no reasoning, no explanation, no preamble."
)


class QueryDecomposer:
    """The turn-1 half of query routing (see StandaloneRewriter for turn 2+) — one cheap
    bulk-LLM call, always made (no keyword pre-filter), that classifies a question BEFORE
    retrieval runs so the pipeline can pick the right strategy: is_meta (skip retrieval,
    answer conversationally), is_broad (skip straight to the wide-source path instead of
    paying for hybrid search + rerank only to discard the result), wants_chart (hint the
    answer prompt), and how it decomposes:

    - "parallel": independent parts, all retrievable right away.
    - "sequential": a later part can't even be SEARCHED until an earlier part's answer is
      known (e.g. "which department had the most failures, and what policy caused
      them?") — Pipeline._retrieve runs these as a chain of hops, resolving each later
      hop's {hopN} placeholder from the previous hop's own retrieval via resolve_hop()
      before that hop ever runs. Capped at MAX_HOPS.

    A keyword pre-filter can catch "hi"/"hello" but not "give me an overview" (is_broad)
    or "compare X and Y" (wants_chart) — those are semantic judgments, not lexical
    patterns, which is why this call is unconditional rather than gated like the old
    looks_compound() check it replaces.

    decompose() always returns a non-empty sub_questions list: [question] unchanged
    (mode "single") on LLM failure or when nothing else applies — callers never need to
    special-case "not decomposed"."""

    def __init__(self, llm: LLMClient, max_subquestions: int):
        self.llm = llm
        self.max_subquestions = max_subquestions

    def decompose(self, question: str) -> DecomposeResult:
        try:
            reply = self.llm.complete(question, system=DECOMPOSE_SYSTEM_PROMPT).strip()
        except RuntimeError:
            return DecomposeResult("single", [question])
        if not reply:
            return DecomposeResult("single", [question])

        lines = [p for p in (line.strip("-•* \t") for line in reply.splitlines()) if p]
        if not lines:
            return DecomposeResult("single", [question])
        if lines[0].strip().upper() == META_MARKER:
            return DecomposeResult("single", [question], is_meta=True)

        is_broad, wants_chart, mode, rest = parse_route_header(lines)
        if mode == SEQUENTIAL_MARKER:
            hops = rest[:MAX_HOPS]
            if len(hops) > 1:
                return DecomposeResult("sequential", hops, is_broad=is_broad, wants_chart=wants_chart)
        elif mode == PARALLEL_MARKER:
            parts = rest[: self.max_subquestions]
            if len(parts) > 1:
                return DecomposeResult("parallel", parts, is_broad=is_broad, wants_chart=wants_chart)
        return DecomposeResult("single", [question], is_broad=is_broad, wants_chart=wants_chart)

    def resolve_hop(self, input_question: str, input_context: str, template: str, placeholder: str) -> str:
        """Turns a later hop's placeholder (e.g. "{hop1}") into a clean, self-contained
        question via one cheap bulk-LLM call over the input hop's raw retrieved text. A
        raw truncated chunk dump (the zero-cost approach this replaced) makes for a noisy
        embedding query — e.g. substituting a 200-char quote fragment instead of the
        actual name it contains. Falls back to that same raw-concatenation behavior when
        there's no input context to resolve from (skipping the LLM call entirely —
        nothing to extract), when the model can't find a clear answer, or on any LLM
        failure — this never regresses below what plain concatenation would have
        produced."""
        fallback = (
            template.replace(placeholder, input_context or input_question)
            if placeholder in template
            else (f"{input_context}. {template}" if input_context else template)
        )
        if not input_context:
            return fallback
        prompt = (
            f"INPUT QUESTION: {input_question}\nINPUT CONTEXT: {input_context}\n"
            f"NEXT HOP TEMPLATE: {template}"
        )
        try:
            reply = self.llm.complete(prompt, system=HOP_RESOLVE_SYSTEM_PROMPT).strip()
        except RuntimeError:
            return fallback
        if not reply or reply == "UNRESOLVED":
            return fallback
        return reply

    def rewrite_for_retry(self, sub_question: str, original_question: str) -> str:
        """Called once per sub-question that came back with no evidence — see
        Pipeline._retrieve's retry pass. Returns sub_question unchanged on any failure
        so the caller can treat "unchanged" as "nothing to retry"."""
        prompt = f"Original question: {original_question}\nSub-question with no evidence: {sub_question}"
        try:
            reply = self.llm.complete(prompt, system=RETRY_REWRITE_SYSTEM_PROMPT).strip()
        except RuntimeError:
            return sub_question
        return reply or sub_question
