from __future__ import annotations

from src.db import connect, init_db, new_id, now, sweep_stale_ingests


def _make_space(db_path: str) -> str:
    space_id = new_id()
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO spaces (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (space_id, "Demo", now(), now()),
        )
    return space_id


def test_init_db_is_idempotent(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    init_db(db_path)
    init_db(db_path)  # must not raise on re-run


def test_deleting_a_space_cascades_to_sources_chats_messages(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    init_db(db_path)
    space_id = _make_space(db_path)

    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sources (id, space_id, kind, name, created_at) VALUES (?, ?, 'text', 'n', ?)",
            (new_id(), space_id, now()),
        )
        chat_id = new_id()
        conn.execute(
            "INSERT INTO chats (id, space_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (chat_id, space_id, now(), now()),
        )
        conn.execute(
            "INSERT INTO messages (id, chat_id, seq, role, content, created_at) "
            "VALUES (?, ?, 0, 'user', 'hi', ?)",
            (new_id(), chat_id, now()),
        )

        conn.execute("DELETE FROM spaces WHERE id=?", (space_id,))
        assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_init_db_migrates_an_old_sources_table_to_allow_csv(tmp_path) -> None:
    """Simulates a database created before 'csv' was added to the kind CHECK constraint —
    init_db must widen it in place, without losing the row already there."""
    db_path = str(tmp_path / "app.db")
    with connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE spaces (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
              color TEXT NOT NULL DEFAULT 'accent', rerank_min_top_score REAL,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE sources (
              id TEXT PRIMARY KEY,
              space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
              kind TEXT NOT NULL CHECK (kind IN ('repo','pdf','docx','text')),
              name TEXT NOT NULL, uri TEXT, meta TEXT NOT NULL DEFAULT '{}',
              status TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending','ingesting','ready','failed')),
              error TEXT, file_count INTEGER NOT NULL DEFAULT 0,
              chunk_count INTEGER NOT NULL DEFAULT 0, ingest_trace TEXT,
              created_at TEXT NOT NULL, ingested_at TEXT
            );
        """)
        space_id = new_id()
        conn.execute(
            "INSERT INTO spaces (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (space_id, "Demo", now(), now()),
        )
        old_source_id = new_id()
        conn.execute(
            "INSERT INTO sources (id, space_id, kind, name, created_at) VALUES (?, ?, 'repo', 'old', ?)",
            (old_source_id, space_id, now()),
        )

    init_db(db_path)  # should migrate sources in place

    with connect(db_path) as conn:
        # the pre-existing row survived the migration untouched
        row = conn.execute("SELECT kind, name FROM sources WHERE id=?", (old_source_id,)).fetchone()
        assert (row["kind"], row["name"]) == ("repo", "old")
        # and 'csv' is now accepted where it previously would have violated the CHECK
        conn.execute(
            "INSERT INTO sources (id, space_id, kind, name, created_at) VALUES (?, ?, 'csv', 'new', ?)",
            (new_id(), space_id, now()),
        )
        assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 2


def test_sweep_stale_ingests_marks_interrupted_sources_failed(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    init_db(db_path)
    space_id = _make_space(db_path)
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sources (id, space_id, kind, name, status, created_at) "
            "VALUES (?, ?, 'repo', 'n', 'ingesting', ?)",
            (new_id(), space_id, now()),
        )

    changed = sweep_stale_ingests(db_path)
    assert changed == 1
    with connect(db_path) as conn:
        row = conn.execute("SELECT status, error FROM sources").fetchone()
        assert row["status"] == "failed"
        assert row["error"]
