from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS spaces (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  color TEXT NOT NULL DEFAULT 'accent',
  rerank_min_top_score REAL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY,
  space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN ('repo','pdf','docx','text','csv')),
  name TEXT NOT NULL,
  uri TEXT,
  meta TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'pending'
         CHECK (status IN ('pending','ingesting','ready','failed')),
  error TEXT,
  file_count INTEGER NOT NULL DEFAULT 0,
  chunk_count INTEGER NOT NULL DEFAULT 0,
  ingest_trace TEXT,
  created_at TEXT NOT NULL,
  ingested_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sources_space ON sources(space_id);

CREATE TABLE IF NOT EXISTS chats (
  id TEXT PRIMARY KEY,
  space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
  title TEXT NOT NULL DEFAULT 'New chat',
  memory_summary TEXT NOT NULL DEFAULT '',
  memory_folded_turns INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chats_space_updated ON chats(space_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user','assistant')),
  content TEXT NOT NULL,
  standalone_question TEXT,
  cache_hit INTEGER NOT NULL DEFAULT 0,
  cached_from TEXT,
  cache_kind TEXT,
  cache_match_question TEXT,
  cache_match_score REAL,
  trace TEXT,
  chart TEXT,
  table_data TEXT,
  created_at TEXT NOT NULL,
  UNIQUE (chat_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, seq);

CREATE TABLE IF NOT EXISTS qa_cache (
  space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
  question_hash TEXT NOT NULL,
  question TEXT NOT NULL,
  message_id TEXT NOT NULL,
  hit_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  PRIMARY KEY (space_id, question_hash)
);

-- encrypted_password is a Fernet ciphertext (see src/crypto.py) — never plain text,
-- and never selected back out to any API response. A connector is config for a future
-- tool to use, not a tool itself; nothing here executes a query.
CREATE TABLE IF NOT EXISTS connectors (
  id TEXT PRIMARY KEY,
  space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN ('redis','postgres')),
  name TEXT NOT NULL,
  host TEXT NOT NULL,
  port INTEGER NOT NULL,
  database TEXT,
  username TEXT,
  encrypted_password TEXT NOT NULL,
  db_index INTEGER,
  tls INTEGER NOT NULL DEFAULT 0,
  ssl INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'untested' CHECK (status IN ('connected','error','untested')),
  last_tested_at TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_connectors_space ON connectors(space_id);

CREATE TABLE IF NOT EXISTS connector_test_history (
  id TEXT PRIMARY KEY,
  connector_id TEXT NOT NULL REFERENCES connectors(id) ON DELETE CASCADE,
  ok INTEGER NOT NULL,
  message TEXT NOT NULL DEFAULT '',
  tested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_connector_history ON connector_test_history(connector_id, tested_at DESC);
"""


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def connect(db_path: str):
    """A fresh connection per call — SQLite connections are cheap (microseconds) and this
    sidesteps check_same_thread entirely, which matters once background ingest threads
    and request threads both touch the db."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        # ponytail: no migration framework by design — one-off patches only if this trivial.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
        if "chart" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN chart TEXT")
        if "table_data" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN table_data TEXT")
        if "cache_kind" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN cache_kind TEXT")
            conn.execute("ALTER TABLE messages ADD COLUMN cache_match_question TEXT")
            conn.execute("ALTER TABLE messages ADD COLUMN cache_match_score REAL")
        chat_columns = {row["name"] for row in conn.execute("PRAGMA table_info(chats)")}
        if "memory_summary" not in chat_columns:
            conn.execute("ALTER TABLE chats ADD COLUMN memory_summary TEXT NOT NULL DEFAULT ''")
            conn.execute("ALTER TABLE chats ADD COLUMN memory_folded_turns INTEGER NOT NULL DEFAULT 0")
        # SQLite can't ALTER a CHECK constraint — recreate the table to widen 'kind'.
        sources_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='sources'"
        ).fetchone()
        if sources_sql and "'csv'" not in sources_sql["sql"]:
            conn.executescript("""
                ALTER TABLE sources RENAME TO sources_old;
                CREATE TABLE sources (
                  id TEXT PRIMARY KEY,
                  space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
                  kind TEXT NOT NULL CHECK (kind IN ('repo','pdf','docx','text','csv')),
                  name TEXT NOT NULL,
                  uri TEXT,
                  meta TEXT NOT NULL DEFAULT '{}',
                  status TEXT NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending','ingesting','ready','failed')),
                  error TEXT,
                  file_count INTEGER NOT NULL DEFAULT 0,
                  chunk_count INTEGER NOT NULL DEFAULT 0,
                  ingest_trace TEXT,
                  created_at TEXT NOT NULL,
                  ingested_at TEXT
                );
                INSERT INTO sources SELECT * FROM sources_old;
                DROP TABLE sources_old;
                CREATE INDEX IF NOT EXISTS idx_sources_space ON sources(space_id);
            """)


def sweep_stale_ingests(db_path: str) -> int:
    """A process restart mid-ingest leaves a source stuck in 'ingesting' forever since
    nothing will ever flip it — call once at startup so the UI shows 'failed' instead of
    a progress bar that will never move again."""
    with connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE sources SET status='failed', error='Interrupted by restart' "
            "WHERE status='ingesting'"
        )
        return cur.rowcount
