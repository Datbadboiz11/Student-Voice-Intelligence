from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.inference import find_project_dir


@dataclass(frozen=True)
class StorageConfig:
    database_path: Path

    @classmethod
    def from_project(cls, project_dir: Path | None = None) -> "StorageConfig":
        root = project_dir or find_project_dir()
        return cls(
            database_path=Path(
                os.getenv(
                    "STUDENT_VOICE_STATE_DB",
                    root / "outputs/app_state/student_voice.db",
                )
            )
        )


class AppStorage:
    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or StorageConfig.from_project()
        self.config.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.config.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback_reviews (
                    feedback_id TEXT PRIMARY KEY,
                    urgency_final TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    reviewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    filters_json TEXT NOT NULL,
                    content_markdown TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS topic_discovery_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_topic TEXT,
                    candidate_count INTEGER NOT NULL,
                    cluster_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS topic_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    suggested_name TEXT NOT NULL,
                    approved_name TEXT,
                    keywords_json TEXT NOT NULL,
                    examples_json TEXT NOT NULL,
                    feedback_ids_json TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES topic_discovery_runs(id)
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
                ON chat_messages(session_id, id);

                CREATE TABLE IF NOT EXISTS admin_feedbacks (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    location TEXT NOT NULL DEFAULT '',
                    source_dataset TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    sentiment_confidence REAL NOT NULL,
                    topic TEXT NOT NULL,
                    topic_confidence REAL NOT NULL,
                    toxic INTEGER NOT NULL,
                    urgency TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'new',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_dataset, content_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_admin_feedbacks_filters
                ON admin_feedbacks(status, topic, sentiment, urgency);

                """
            )

    def review_map(self) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT feedback_id, urgency_final, reviewer, note, reviewed_at FROM feedback_reviews"
            ).fetchall()
        return {row["feedback_id"]: dict(row) for row in rows}

    def save_review(
        self,
        feedback_id: str,
        urgency_final: str,
        reviewer: str,
        note: str,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback_reviews(feedback_id, urgency_final, reviewer, note, reviewed_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(feedback_id) DO UPDATE SET
                    urgency_final = excluded.urgency_final,
                    reviewer = excluded.reviewer,
                    note = excluded.note,
                    reviewed_at = CURRENT_TIMESTAMP
                """,
                (feedback_id, urgency_final, reviewer, note),
            )
            row = connection.execute(
                "SELECT feedback_id, urgency_final, reviewer, note, reviewed_at FROM feedback_reviews WHERE feedback_id = ?",
                (feedback_id,),
            ).fetchone()
        return dict(row)

    def save_report(
        self,
        title: str,
        filters: dict[str, Any],
        content_markdown: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO reports(title, filters_json, content_markdown, data_json) VALUES (?, ?, ?, ?)",
                (
                    title,
                    json.dumps(filters, ensure_ascii=False),
                    content_markdown,
                    json.dumps(data, ensure_ascii=False),
                ),
            )
            row = connection.execute("SELECT * FROM reports WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return self._report_record(row)

    @staticmethod
    def _report_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "filters": json.loads(row["filters_json"]),
            "content_markdown": row["content_markdown"],
            "data": json.loads(row["data_json"]),
            "created_at": row["created_at"],
        }

    def list_reports(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reports ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._report_record(row) for row in rows]

    def get_report(self, report_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        return self._report_record(row) if row else None

    def save_discovery_run(
        self,
        source_topic: str | None,
        candidate_count: int,
        clusters: list[dict[str, Any]],
    ) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO topic_discovery_runs(source_topic, candidate_count, cluster_count) VALUES (?, ?, ?)",
                (source_topic, candidate_count, len(clusters)),
            )
            run_id = int(cursor.lastrowid)
            for cluster in clusters:
                connection.execute(
                    """
                    INSERT INTO topic_clusters(
                        run_id, suggested_name, keywords_json, examples_json, feedback_ids_json, size
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        cluster["suggested_name"],
                        json.dumps(cluster["keywords"], ensure_ascii=False),
                        json.dumps(cluster["examples"], ensure_ascii=False),
                        json.dumps(cluster["feedback_ids"], ensure_ascii=False),
                        cluster["size"],
                    ),
                )
        return {"run_id": run_id, "candidate_count": candidate_count, "cluster_count": len(clusters)}

    @staticmethod
    def _cluster_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "run_id": row["run_id"],
            "suggested_name": row["suggested_name"],
            "approved_name": row["approved_name"],
            "keywords": json.loads(row["keywords_json"]),
            "examples": json.loads(row["examples_json"]),
            "feedback_ids": json.loads(row["feedback_ids_json"]),
            "size": row["size"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_clusters(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT * FROM topic_clusters"
        values: tuple[Any, ...] = ()
        if status:
            sql += " WHERE status = ?"
            values = (status,)
        sql += " ORDER BY id DESC LIMIT ?"
        with self._connect() as connection:
            rows = connection.execute(sql, (*values, limit)).fetchall()
        return [self._cluster_record(row) for row in rows]

    def update_cluster(
        self,
        cluster_id: int,
        status: str,
        approved_name: str | None = None,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE topic_clusters
                SET status = ?, approved_name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, approved_name, cluster_id),
            )
            row = connection.execute("SELECT * FROM topic_clusters WHERE id = ?", (cluster_id,)).fetchone()
        return self._cluster_record(row) if row else None

    @staticmethod
    def _chat_session_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_chat_session(self, title: str) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute("INSERT INTO chat_sessions(title) VALUES (?)", (title,))
            row = connection.execute("SELECT * FROM chat_sessions WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return self._chat_session_record(row)

    def list_chat_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chat_sessions ORDER BY updated_at DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._chat_session_record(row) for row in rows]

    def get_chat_session(self, session_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        return self._chat_session_record(row) if row else None

    def list_chat_messages(self, session_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "result": json.loads(row["result_json"]) if row["result_json"] else None,
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def save_chat_message(
        self,
        session_id: int,
        role: str,
        content: str,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO chat_messages(session_id, role, content, result_json) VALUES (?, ?, ?, ?)",
                (session_id, role, content, json.dumps(result, ensure_ascii=False) if result else None),
            )
            connection.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,)
            )
            row = connection.execute("SELECT * FROM chat_messages WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "created_at": row["created_at"],
        }

    def update_chat_session_title(self, session_id: int, title: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, session_id),
            )
            row = connection.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        return self._chat_session_record(row) if row else None

    def delete_chat_session(self, session_id: int) -> bool:
        with self._connect() as connection:
            connection.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            cursor = connection.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    def get_admin_feedback_by_hash(self, source_dataset: str, content_hash: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM admin_feedbacks WHERE source_dataset = ? AND content_hash = ?", (source_dataset, content_hash)).fetchone()
        return dict(row) if row else None

    def save_admin_feedback(self, record: dict[str, Any]) -> dict[str, Any]:
        fields = ("id", "text", "location", "source_dataset", "content_hash", "sentiment", "sentiment_confidence", "topic", "topic_confidence", "toxic", "urgency", "status")
        with self._connect() as connection:
            connection.execute(f"INSERT INTO admin_feedbacks({', '.join(fields)}) VALUES ({', '.join('?' for _ in fields)})", tuple(record[field] for field in fields))
            row = connection.execute("SELECT * FROM admin_feedbacks WHERE id = ?", (record["id"],)).fetchone()
        return dict(row)

    def list_admin_feedbacks(self, status: str | None = None, topic: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        clauses, values = [], []
        if status: clauses.append("status = ?"); values.append(status)
        if topic: clauses.append("topic = ?"); values.append(topic)
        sql = "SELECT * FROM admin_feedbacks" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY created_at DESC, id DESC LIMIT ?"
        with self._connect() as connection:
            rows = connection.execute(sql, (*values, limit)).fetchall()
        return [dict(row) for row in rows]

    def feedback_frame(self) -> pd.DataFrame:
        rows = self.list_admin_feedbacks(limit=100_000)
        if not rows:
            return pd.DataFrame(columns=["row_id", "dataset", "sentiment", "topic", "urgency", "toxic"])
        return pd.DataFrame(rows).rename(columns={"id": "row_id", "source_dataset": "dataset"})[["row_id", "dataset", "sentiment", "topic", "urgency", "toxic"]]

    def update_admin_feedback_status(self, feedback_id: str, status: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            connection.execute("UPDATE admin_feedbacks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (status, feedback_id))
            row = connection.execute("SELECT * FROM admin_feedbacks WHERE id = ?", (feedback_id,)).fetchone()
        return dict(row) if row else None

    def delete_admin_feedback(self, feedback_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM admin_feedbacks WHERE id = ?", (feedback_id,))
        return cursor.rowcount > 0
