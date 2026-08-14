"""Replays evals/golden_set.py against the REAL pipeline (real Qdrant data, real LLM
calls) and scores each case. Not a pytest suite on purpose — these cost real API calls
and real wall-clock time (~10-30s/case), so this is a "run before/after a change" tool:

    .venv/bin/python -m evals.runner        # or: make eval

Requires the app's Qdrant to be reachable (docker compose up), since it queries whatever
space_ids are hardcoded in golden_set.py. Writes evals/last_run.json for later diffing.
"""

from __future__ import annotations

import json
import sys
import time

from src.config import settings
from src.pipeline import Pipeline
from src.trace import QueryTrace

from evals.golden_set import GOLDEN_SET, EvalCase
from evals.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


def _check_gate(case: EvalCase, trace: QueryTrace) -> tuple[bool, str]:
    is_no_match = trace.answer is not None and trace.answer.text == Pipeline.NO_MATCH
    if case.expect_gate == "no_match":
        return is_no_match, f"wide_fallback={trace.wide_fallback} answer={trace.answer.text[:80]!r}"
    if case.expect_gate == "wide_fallback":
        ok = trace.wide_fallback and not is_no_match
        return ok, f"wide_fallback={trace.wide_fallback} reason={trace.wide_fallback_reason!r}"
    # "answered": a grounded, non-refused answer actually reached generation
    ok = not is_no_match and bool(trace.final_chunks)
    return ok, f"wide_fallback={trace.wide_fallback} final_chunks={len(trace.final_chunks)}"


def run_case(pipeline: Pipeline, case: EvalCase) -> dict:
    started = time.monotonic()
    trace = pipeline.query_trace(case.question, case.space_id)
    elapsed = time.monotonic() - started

    checks: dict[str, bool] = {}
    gate_ok, gate_detail = _check_gate(case, trace)
    checks["gate"] = gate_ok

    if case.expect_files_contain:
        paths = [f.file_path for f in trace.routed_files] + [r.chunk.file_path for r in trace.reranked]
        checks["files"] = any(case.expect_files_contain.lower() in p.lower() for p in paths)

    if case.expect_answer_contains:
        text = (trace.answer.text if trace.answer else "").lower()
        checks["answer_contains"] = all(kw.lower() in text for kw in case.expect_answer_contains)

    if case.expect_decomposed is not None:
        checks["decomposed"] = (len(trace.sub_questions) > 1) == case.expect_decomposed

    faithfulness = None
    # Only meaningful when a real answer was actually generated from chunks — a NO_MATCH
    # refusal or a "too large" wide-fallback message has no claims to check.
    is_no_match = trace.answer is not None and trace.answer.text == Pipeline.NO_MATCH
    # trace.final_chunks is list[CompressedChunkTrace] (chunk + drop verdict), not
    # list[CodeChunk] directly — unwrap to what generation actually received: the KEPT
    # chunks' .chunk. A dropped entry was compression saying "not relevant", so checking
    # the answer against it too would be checking against context it was never given.
    kept_chunks = [fc.chunk for fc in trace.final_chunks if not fc.dropped]
    if trace.answer and not is_no_match and kept_chunks:
        result = pipeline.check_faithfulness(case.question, trace.answer.text, kept_chunks)
        checks["faithful"] = result.supported
        faithfulness = {"checked": result.checked, "unsupported_claims": result.unsupported_claims}

    metrics = None
    if case.expect_relevant_files:
        # A page can span multiple chunks — trace.reranked ranks CHUNKS, but relevance is
        # judged per PAGE, so dedupe to each page's best (first) rank before scoring.
        # Skipping this double-counts a page that shows up twice, which is how R@K and
        # NDCG@K end up above 1.0 — a real bug this caught on its first live run.
        ranked = list(dict.fromkeys(r.chunk.file_path for r in trace.reranked))
        relevant = set(case.expect_relevant_files)
        k = len(ranked) or 1
        metrics = {
            "precision@k": round(precision_at_k(ranked, relevant, k), 3),
            "recall@k": round(recall_at_k(ranked, relevant, k), 3),
            "mrr": round(mrr(ranked, relevant, k), 3),
            "ndcg@k": round(ndcg_at_k(ranked, relevant, k), 3),
        }

    return {
        "name": case.name,
        "question": case.question,
        "passed": all(checks.values()),
        "checks": checks,
        "gate_detail": gate_detail,
        "elapsed_s": round(elapsed, 1),
        "answer": trace.answer.text[:200] if trace.answer else None,
        "metrics": metrics,
        "faithfulness": faithfulness,
    }


def main() -> None:
    pipeline = Pipeline(settings)
    results = [run_case(pipeline, case) for case in GOLDEN_SET]

    print(f"\n{'CASE':32s} RESULT  CHECKS")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        checks_str = ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in r["checks"].items())
        print(f"{r['name']:32s} {status:6s}  {checks_str}  ({r['elapsed_s']}s)")
        if r["metrics"]:
            m = r["metrics"]
            print(f"    P@K={m['precision@k']}  R@K={m['recall@k']}  MRR={m['mrr']}  NDCG@K={m['ndcg@k']}")
        if r["faithfulness"] and r["faithfulness"]["unsupported_claims"]:
            print(f"    UNSUPPORTED CLAIMS: {r['faithfulness']['unsupported_claims']}")
        if not r["passed"]:
            print(f"    question:    {r['question']!r}")
            print(f"    gate_detail: {r['gate_detail']}")
            print(f"    answer:      {r['answer']!r}")

    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} passed")

    scored = [r["metrics"] for r in results if r["metrics"]]
    if scored:
        avg = {k: round(sum(m[k] for m in scored) / len(scored), 3) for k in scored[0]}
        print(
            f"Retrieval ({len(scored)} cases with ground truth): "
            f"P@K={avg['precision@k']}  R@K={avg['recall@k']}  MRR={avg['mrr']}  NDCG@K={avg['ndcg@k']}"
        )

    checked = [r for r in results if r["faithfulness"]]
    if checked:
        faithful = sum(1 for r in checked if r["checks"]["faithful"])
        print(f"Faithfulness ({len(checked)} answered cases): {faithful}/{len(checked)} fully grounded")

    with open("evals/last_run.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Full report: evals/last_run.json")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
