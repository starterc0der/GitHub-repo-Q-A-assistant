from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import settings
from src.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("url")

    # Retrieval runs without a generation backend (every LLM stage has a fallback), so
    # this builds the real prompt even with no generation backend — paste it into any chat UI
    # judge the system prompt's grounding on its own, separately from model quality.
    prompt_parser = subparsers.add_parser("prompt", help="build the final prompt, don't answer")
    prompt_parser.add_argument("repo")
    prompt_parser.add_argument("question")
    prompt_parser.add_argument("-o", "--out", help="write to this file instead of stdout")

    args = parser.parse_args()
    if args.command == "ingest":
        report = Pipeline(settings).ingest_repo(args.url)
        print(f"Ingested {report.repo}: {report.file_count} files, {report.chunk_count} chunks")
    elif args.command == "prompt":
        trace = Pipeline(settings).query_trace(args.question, args.repo)
        text = f"{trace.system_prompt}\n\n---\n\n{trace.final_prompt}\n"
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"Wrote {len(text)} chars (~{len(text) // 4} tokens) to {args.out}")
        else:
            print(text)


if __name__ == "__main__":
    main()
