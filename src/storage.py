from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
