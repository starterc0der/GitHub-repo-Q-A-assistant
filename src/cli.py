from __future__ import annotations

import argparse
import logging

from src.config import settings
from src.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("url")

    args = parser.parse_args()
    if args.command == "ingest":
        report = Pipeline(settings).ingest_repo(args.url)
        print(f"Ingested {report.repo}: {report.file_count} files, {report.chunk_count} chunks")


if __name__ == "__main__":
    main()
