from __future__ import annotations

from pathlib import Path

from src.ingest.repo_loader import RepoLoader


def _write(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_walk_files_filters_extensions_dirs_and_size(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "main.py")
    _write(tmp_path / "src" / "app.js")
    _write(tmp_path / "readme.md")
    _write(tmp_path / "node_modules" / "pkg" / "index.js")
    _write(tmp_path / ".git" / "config.py")
    _write(tmp_path / "big.py", "x = 1\n" * 100_000)

    found = {p.relative_to(tmp_path) for p in RepoLoader().walk_files(tmp_path, max_bytes=1000)}

    assert Path("src/main.py") in found
    assert Path("src/app.js") in found
    assert Path("readme.md") not in found
    assert Path("node_modules/pkg/index.js") not in found
    assert Path(".git/config.py") not in found
    assert Path("big.py") not in found
