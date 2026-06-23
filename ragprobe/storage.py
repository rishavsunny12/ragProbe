"""SQLite persistence layer for RAGProbe.

Stores embedded chunks (so the expensive embedding step is not repeated),
generated questions, and run reports. Embeddings and metadata are serialized
as JSON strings; this keeps the schema simple and dependency-free.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import Chunk, Question, RunReport

DEFAULT_DB_PATH = ".ragprobe/index.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL,   -- JSON string
    embedding TEXT NOT NULL    -- JSON array of floats
);

CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL          -- full Question JSON
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    data TEXT NOT NULL          -- full RunReport JSON
);
"""


class Storage:
    """Thin wrapper around a SQLite database file."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- context manager -------------------------------------------------

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # -- chunks ----------------------------------------------------------

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Insert or replace chunks. Embeddings must be populated."""
        rows = []
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.id} has no embedding; cannot persist.")
            rows.append(
                (
                    c.id,
                    c.text,
                    json.dumps(c.metadata),
                    json.dumps(c.embedding),
                )
            )
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunks (id, text, metadata, embedding) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def load_chunks(self) -> list[Chunk]:
        cur = self.conn.execute("SELECT id, text, metadata, embedding FROM chunks")
        chunks: list[Chunk] = []
        for row in cur.fetchall():
            chunks.append(
                Chunk(
                    id=row["id"],
                    text=row["text"],
                    metadata=json.loads(row["metadata"]),
                    embedding=json.loads(row["embedding"]),
                )
            )
        return chunks

    def chunk_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) AS n FROM chunks")
        return int(cur.fetchone()["n"])

    # -- questions -------------------------------------------------------

    def save_questions(self, questions: list[Question]) -> None:
        rows = [(q.id, q.model_dump_json()) for q in questions]
        self.conn.executemany(
            "INSERT OR REPLACE INTO questions (id, data) VALUES (?, ?)",
            rows,
        )
        self.conn.commit()

    def load_questions(self) -> list[Question]:
        cur = self.conn.execute("SELECT data FROM questions")
        return [Question.model_validate_json(row["data"]) for row in cur.fetchall()]

    # -- runs ------------------------------------------------------------

    def save_run(self, report: RunReport) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, timestamp, data) VALUES (?, ?, ?)",
            (report.run_id, report.timestamp, report.model_dump_json()),
        )
        self.conn.commit()

    def load_run(self, run_id: str) -> RunReport | None:
        cur = self.conn.execute("SELECT data FROM runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return RunReport.model_validate_json(row["data"])
