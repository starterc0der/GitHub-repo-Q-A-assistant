from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_INCLUDE_EXT = (".py", ".js", ".ts", ".go", ".java", ".rs")
SKIP_DIRS = {".git", "node_modules", "vendor", "dist", "build", "__pycache__", ".venv"}
SKIP_FILES = {"package-lock.json", "yarn.lock", "poetry.lock", "Cargo.lock"}


@dataclass
class SkippedFile:
    path: str
    reason: str


@dataclass
class WalkReport:
    total_scanned: int
    kept: list[Path]
    skipped: list[SkippedFile] = field(default_factory=list)


class RepoLoader:
    """Clones a repo to disk and walks it for source files worth chunking."""

    def clone(self, url: str, dest: Path) -> Path:
        if dest.exists():
            shutil.rmtree(dest)  # re-ingesting a repo: git clone refuses a non-empty dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        return dest

    def commit_sha(self, dest: Path) -> str:
        result = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def walk_files(
        self,
        root: Path,
        include_ext: tuple[str, ...] = DEFAULT_INCLUDE_EXT,
        max_bytes: int = 200_000,
    ) -> Iterator[Path]:
        return iter(self.walk_report(root, include_ext, max_bytes).kept)

    def walk_report(
        self,
        root: Path,
        include_ext: tuple[str, ...] = DEFAULT_INCLUDE_EXT,
        max_bytes: int = 200_000,
    ) -> WalkReport:
        total = 0
        kept: list[Path] = []
        skipped: list[SkippedFile] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            total += 1
            reason = self._skip_reason(path, root, include_ext, max_bytes)
            if reason is None:
                kept.append(path)
            else:
                skipped.append(SkippedFile(path=str(path.relative_to(root)), reason=reason))
        return WalkReport(total_scanned=total, kept=kept, skipped=skipped)

    def _skip_reason(
        self, path: Path, root: Path, include_ext: tuple[str, ...], max_bytes: int
    ) -> str | None:
        # Dir membership checked first so files inside e.g. .git report "skip dir: .git"
        # rather than the less useful "unsupported extension" (most such files have no
        # extension we'd include anyway) — this ordering only affects the reported reason,
        # never whether a file is kept or skipped.
        hit = SKIP_DIRS & set(path.relative_to(root).parts)
        if hit:
            return f"skip dir: {sorted(hit)[0]}"
        if path.suffix not in include_ext:
            return "unsupported extension"
        if path.name in SKIP_FILES:
            return "lockfile"
        if path.stat().st_size > max_bytes:
            return "too large"
        return None
