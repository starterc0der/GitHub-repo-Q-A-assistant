from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

DEFAULT_INCLUDE_EXT = (".py", ".js", ".ts", ".go", ".java", ".rs")
SKIP_DIRS = {".git", "node_modules", "vendor", "dist", "build", "__pycache__", ".venv"}
SKIP_FILES = {"package-lock.json", "yarn.lock", "poetry.lock", "Cargo.lock"}


class RepoLoader:
    """Clones a repo to disk and walks it for source files worth chunking."""

    def clone(self, url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        return dest

    def walk_files(
        self,
        root: Path,
        include_ext: tuple[str, ...] = DEFAULT_INCLUDE_EXT,
        max_bytes: int = 200_000,
    ) -> Iterator[Path]:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in include_ext:
                continue
            if path.name in SKIP_FILES:
                continue
            if SKIP_DIRS & set(path.relative_to(root).parts):
                continue
            if path.stat().st_size > max_bytes:
                continue
            yield path
