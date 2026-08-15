from __future__ import annotations

from dataclasses import dataclass

from src.llm_client import LLMClient

SINGLE_SENTINEL = "SINGLE"
PARALLEL_MARKER = "PARALLEL"
SEQUENTIAL_MARKER = "SEQUENTIAL"
MAX_HOPS = 3  # caps latency/cost the same way the retry pass caps itself at one attempt

# Cheap, free, catches the common phrasings before anything pays for an LLM call — mirrors
# BroadIntentClassifier's keyword-first gate in intent.py.
COMPOUND_HINTS = (" and ", " & ", "; ")

DECOMPOSE_SYSTEM_PROMPT = (
    "Decide how the question below breaks down.\n"
    f"- If it is genuinely one question, reply with exactly: {SINGLE_SENTINEL}\n"
    "- If it is multiple INDEPENDENT questions (different topics joined by \"and\", or a "
    "list of separate asks, each answerable on its own without the others), reply "
    f"\"{PARALLEL_MARKER}\" on the first line, then at most 3 self-contained sub-questions, "
    "one per line — resolve pronouns like \"it\"/\"that\" against the original question.\n"
    "- If answering a later part REQUIRES the answer to an earlier part first — you could not "
    "even search for it without already knowing that answer (e.g. \"which department had the "
    "most failures, and what policy caused them?\" — you can't search for the policy until you "
    f"know which department) — reply \"{SEQUENTIAL_MARKER}\" on the first line, then 2 or 3 "
    "lines, one per hop: the first hop's self-contained question, then each later hop's "
    "question written with a {hop1}, {hop2}, etc. placeholder wherever it depends on that "
    "earlier hop's answer (e.g. \"What policy caused failures in {hop1}?\") — most chains are "
    "only 2 hops; use a third only when the question genuinely needs 3 sequential lookups.\n"
    "Do not answer any of them, do not add numbering or preamble. Reply with only the mode "
    "line and the sub-questions/hops, nothing else."
)


def looks_compound(question: str) -> bool:
    lowered = question.lower()
    return lowered.count("?") > 1 or any(hint in lowered for hint in COMPOUND_HINTS)


@dataclass
class DecomposeResult:
    mode: str  # "single" | "parallel" | "sequential"
    sub_questions: list[str]


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
    "material. Rewrite it as a differently-phrased, self-contained question aimed at the "
    "same underlying information need — the original phrasing may just not match how the "
    "source text describes it (e.g. a comparison sub-question often finds more by asking "
    "directly about the other side of the comparison instead of the comparison itself). "
    "Reply with only the rewritten question, nothing else."
)


class QueryDecomposer:
    """Splits a compound question into sub-questions before retrieval — a single embedding
    of "what does X do, and how is that different from Y?" is a blurry average of both
    topics, which hurts routing and hybrid search for either one. Two different splits:

    - "parallel": independent parts, all retrievable right away (existing behavior).
    - "sequential": a later part can't even be SEARCHED until an earlier part's answer is
      known (e.g. "which department had the most failures, and what policy caused
      them?") — Pipeline._retrieve runs these as a chain of hops, resolving each later
      hop's {hopN} placeholder from the previous hop's own retrieval via resolve_hop()
      before that hop ever runs. Capped at MAX_HOPS.

    decompose() always returns a non-empty sub_questions list: [question] unchanged
    (mode "single") when it isn't compound, on LLM failure, or on turn 1 of a simple
    question — so callers never need to special-case "not decomposed"."""

    def __init__(self, llm: LLMClient, max_subquestions: int):
        self.llm = llm
        self.max_subquestions = max_subquestions

    def decompose(self, question: str) -> DecomposeResult:
        if not looks_compound(question):
            return DecomposeResult("single", [question])
        try:
            reply = self.llm.complete(question, system=DECOMPOSE_SYSTEM_PROMPT).strip()
        except RuntimeError:
            return DecomposeResult("single", [question])
        if not reply or reply == SINGLE_SENTINEL:
            return DecomposeResult("single", [question])

        lines = [p for p in (line.strip("-•* \t") for line in reply.splitlines()) if p]
        if not lines:
            return DecomposeResult("single", [question])

        marker, rest = lines[0].strip().upper(), lines[1:]
        if marker == SEQUENTIAL_MARKER:
            hops = rest[:MAX_HOPS]
            return DecomposeResult("sequential", hops) if len(hops) > 1 else DecomposeResult("single", [question])
        # No recognized marker (a plain listing, or an older-style reply) falls back to
        # the original behavior: the whole reply is an independent parallel split.
        parts = (rest if marker == PARALLEL_MARKER else lines)[: self.max_subquestions]
        return DecomposeResult("parallel", parts) if len(parts) > 1 else DecomposeResult("single", [question])

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
