from __future__ import annotations

import subprocess
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


def test_walk_report_records_skip_reasons(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "main.py")
    _write(tmp_path / "readme.md")
    _write(tmp_path / "node_modules" / "pkg" / "index.js")
    _write(tmp_path / "big.py", "x = 1\n" * 100_000)

    report = RepoLoader().walk_report(tmp_path, max_bytes=1000)

    assert report.total_scanned == 4
    assert {p.relative_to(tmp_path) for p in report.kept} == {Path("src/main.py")}
    reasons = {s.path: s.reason for s in report.skipped}
    assert reasons["readme.md"] == "unsupported extension"
    assert reasons["node_modules/pkg/index.js"] == "skip dir: node_modules"
    assert reasons["big.py"] == "too large"


def test_clone_overwrites_an_existing_non_empty_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write(source / "a.py")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "a@a.com"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "a"], cwd=source, check=True)
    subprocess.run(["git", "add", "-A"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=source, check=True)

    dest = tmp_path / "dest"
    RepoLoader().clone(str(source), dest)
    (dest / "stray-file-from-a-previous-run.txt").write_text("leftover")

    RepoLoader().clone(str(source), dest)  # must not raise on the non-empty dest

    assert (dest / "a.py").is_file()
    assert not (dest / "stray-file-from-a-previous-run.txt").exists()


def test_commit_sha_returns_short_hash_of_cloned_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "a.py")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "a@a.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "a"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    sha = RepoLoader().commit_sha(repo)

    assert len(sha) >= 7
    assert all(c in "0123456789abcdef" for c in sha)
