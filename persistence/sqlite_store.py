from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .base import BaseStore
from .models import StoredProject, StoredSession, StoredArtifact, RequestLog


class SQLiteStore(BaseStore):
    """SQLite-backed persistent storage implementation."""

    def __init__(self, db_path: str | Path = "neural_kernel.db") -> None:
        self.db_path = Path(db_path)
        self._lock = threading.RLock()

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper configuration."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema and run migrations."""
        with self._lock:
            conn = self._get_connection()
            try:
                # Read and execute base schema
                schema_path = Path(__file__).parent / "migrations" / "schema.sql"
                with open(schema_path, "r") as f:
                    schema = f.read()

                conn.executescript(schema)
                conn.commit()

                # Run incremental migrations (ALTER TABLE statements may fail
                # with "duplicate column" if already applied — that's OK).
                self._run_migrations(conn)
                conn.commit()
            finally:
                conn.close()

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply incremental SQL migrations that add columns etc."""
        migrations_dir = Path(__file__).parent / "migrations"
        migration_files = sorted(migrations_dir.glob("[0-9]*.sql"))
        for mf in migration_files:
            with open(mf, "r") as f:
                stmts = f.read()
            for raw_stmt in stmts.split(";"):
                # Strip comment lines and whitespace
                lines = [
                    ln for ln in raw_stmt.strip().splitlines()
                    if ln.strip() and not ln.strip().startswith("--")
                ]
                stmt = "\n".join(lines).strip()
                if not stmt:
                    continue
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as exc:
                    # "duplicate column name" is expected for re-runs
                    if "duplicate column" in str(exc).lower():
                        continue
                    raise

    # ========== Projects ==========

    def save_project(self, project: StoredProject) -> str:
        """Save or update a project."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO projects
                    (project_id, user_id, session_id, raw_text, project_name, summary,
                     project_type, features_json, tech_stack_json, status, artifact_path,
                     error_message, created_at, updated_at,
                     owner_client_id, owner_user_id, org_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project.project_id,
                        project.user_id,
                        project.session_id,
                        project.raw_text,
                        project.project_name,
                        project.summary,
                        project.project_type,
                        project.features_json,
                        project.tech_stack_json,
                        project.status,
                        project.artifact_path,
                        project.error_message,
                        project.created_at.isoformat(),
                        project.updated_at.isoformat(),
                        project.owner_client_id,
                        project.owner_user_id,
                        project.org_id,
                    ),
                )
                conn.commit()
                return project.project_id
            finally:
                conn.close()

    def get_project(
        self,
        project_id: str,
        *,
        owner_client_id: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> Optional[StoredProject]:
        """Retrieve a project by ID, optionally scoped to owner/org."""
        with self._lock:
            conn = self._get_connection()
            try:
                clauses = ["project_id = ?"]
                params: list[object] = [project_id]
                if owner_client_id is not None:
                    clauses.append("(owner_client_id = ? OR owner_client_id IS NULL)")
                    params.append(owner_client_id)
                if org_id is not None:
                    clauses.append("(org_id = ? OR org_id IS NULL)")
                    params.append(org_id)

                query = "SELECT * FROM projects WHERE " + " AND ".join(clauses)
                row = conn.execute(query, params).fetchone()

                if row:
                    return self._row_to_project(row)
                return None
            finally:
                conn.close()

    def update_project_status(
        self, project_id: str, status: str, error: Optional[str] = None
    ) -> None:
        """Update a project's status."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    UPDATE projects
                    SET status = ?, error_message = ?, updated_at = ?
                    WHERE project_id = ?
                    """,
                    (status, error, datetime.utcnow().isoformat(), project_id),
                )
                conn.commit()
            finally:
                conn.close()

    def list_projects(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        *,
        owner_client_id: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> List[StoredProject]:
        """List projects with optional filtering."""
        with self._lock:
            conn = self._get_connection()
            try:
                clauses: list[str] = []
                params: list[object] = []

                if user_id:
                    clauses.append("user_id = ?")
                    params.append(user_id)
                if owner_client_id is not None:
                    clauses.append(
                        "(owner_client_id = ? OR owner_client_id IS NULL)"
                    )
                    params.append(owner_client_id)
                if org_id is not None:
                    clauses.append("(org_id = ? OR org_id IS NULL)")
                    params.append(org_id)

                where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
                query = f"""
                    SELECT * FROM projects
                    {where}
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """
                params.extend([limit, offset])

                rows = conn.execute(query, params).fetchall()
                return [self._row_to_project(row) for row in rows]
            finally:
                conn.close()

    # ========== Sessions ==========

    def save_session(self, session: StoredSession) -> str:
        """Save or update a session."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO sessions
                    (session_id, user_id, messages, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.session_id,
                        session.user_id,
                        json.dumps(session.messages),
                        json.dumps(session.metadata),
                        session.created_at.isoformat(),
                        session.updated_at.isoformat(),
                    ),
                )
                conn.commit()
                return session.session_id
            finally:
                conn.close()

    def get_session(self, session_id: str) -> Optional[StoredSession]:
        """Retrieve a session by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

                if row:
                    return self._row_to_session(row)
                return None
            finally:
                conn.close()

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to a session's message history (with timestamp)."""
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT messages FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

                messages = json.loads(row["messages"]) if row else []
                messages.append({
                    "role": role,
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat(),
                })

                conn.execute(
                    """
                    UPDATE sessions
                    SET messages = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (json.dumps(messages), datetime.utcnow().isoformat(), session_id),
                )
                conn.commit()
            finally:
                conn.close()

    def list_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[StoredSession]:
        """List sessions with optional user filtering and pagination."""
        with self._lock:
            conn = self._get_connection()
            try:
                params: list[object] = []
                where = ""
                if user_id:
                    where = "WHERE user_id = ?"
                    params.append(user_id)

                params.extend([limit, offset])
                rows = conn.execute(
                    f"""
                    SELECT * FROM sessions
                    {where}
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    params,
                ).fetchall()

                return [self._row_to_session(row) for row in rows]
            finally:
                conn.close()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if found and deleted."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def get_session_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """Get paginated messages from a session."""
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT messages FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

                if not row:
                    return []

                messages = json.loads(row["messages"])
                return messages[offset : offset + limit]
            finally:
                conn.close()

    # ========== Memory ==========

    def save_memory_entry(
        self, key: str, answer: str, hits: int = 0, tier: str = "short",
        query_text: str = "", source_query_id: str = "",
    ) -> None:
        """Persist a memory entry to the database."""
        now = datetime.utcnow().isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memory
                    (key, query_text, answer, hits, tier, source_query_id,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?,
                            COALESCE((SELECT created_at FROM memory WHERE key = ?), ?),
                            ?)
                    """,
                    (key, query_text or key, answer, hits, tier,
                     source_query_id, key, now, now),
                )
                conn.commit()
            finally:
                conn.close()

    def get_memory_entry(self, key: str) -> Optional[dict]:
        """Retrieve a memory entry by key."""
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM memory WHERE key = ?", (key,),
                ).fetchone()
                if row:
                    return {
                        "key": row["key"],
                        "query_text": row["query_text"],
                        "answer": row["answer"],
                        "hits": row["hits"],
                        "tier": row["tier"],
                        "source_query_id": row["source_query_id"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                return None
            finally:
                conn.close()

    def list_memory(self, tier: Optional[str] = None, limit: int = 200) -> List[dict]:
        """List all memory entries, optionally filtered by tier."""
        with self._lock:
            conn = self._get_connection()
            try:
                params: list[object] = []
                where = ""
                if tier:
                    where = "WHERE tier = ?"
                    params.append(tier)
                params.append(limit)

                rows = conn.execute(
                    f"""
                    SELECT * FROM memory
                    {where}
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()

                return [
                    {
                        "key": r["key"],
                        "answer": r["answer"],
                        "hits": r["hits"],
                        "tier": r["tier"],
                        "updated_at": r["updated_at"],
                    }
                    for r in rows
                ]
            finally:
                conn.close()

    def delete_memory_entry(self, key: str) -> bool:
        """Delete a memory entry. Returns True if found."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("DELETE FROM memory WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    # ========== Artifacts ==========

    def save_artifact(self, artifact: StoredArtifact) -> str:
        """Save an artifact."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO artifacts
                    (artifact_id, project_id, filename, file_path, file_size_bytes,
                     format, created_at, owner_client_id, owner_user_id, org_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact.artifact_id,
                        artifact.project_id,
                        artifact.filename,
                        artifact.file_path,
                        artifact.file_size_bytes,
                        artifact.format,
                        artifact.created_at.isoformat(),
                        artifact.owner_client_id,
                        artifact.owner_user_id,
                        artifact.org_id,
                    ),
                )
                conn.commit()
                return artifact.artifact_id
            finally:
                conn.close()

    def get_artifact(self, artifact_id: str) -> Optional[StoredArtifact]:
        """Retrieve an artifact by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM artifacts WHERE artifact_id = ?",
                    (artifact_id,),
                ).fetchone()

                if row:
                    return self._row_to_artifact(row)
                return None
            finally:
                conn.close()

    def get_artifacts_by_project(self, project_id: str) -> List[StoredArtifact]:
        """Retrieve all artifacts for a project."""
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM artifacts
                    WHERE project_id = ?
                    ORDER BY created_at DESC
                    """,
                    (project_id,),
                ).fetchall()

                return [self._row_to_artifact(row) for row in rows]
            finally:
                conn.close()

    # ========== Request Logs ==========

    def log_request(self, log: RequestLog) -> str:
        """Log a request."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT INTO request_logs
                    (log_id, project_id, user_id, raw_text, parsed_features,
                     parsed_tech_stack, processing_time_ms, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log.log_id,
                        log.project_id,
                        log.user_id,
                        log.raw_text,
                        json.dumps(log.parsed_features),
                        json.dumps(log.parsed_tech_stack),
                        log.processing_time_ms,
                        log.status,
                        log.created_at.isoformat(),
                    ),
                )
                conn.commit()
                return log.log_id
            finally:
                conn.close()

    def get_request_logs(
        self, project_id: Optional[str] = None, limit: int = 100
    ) -> List[RequestLog]:
        """Retrieve request logs."""
        with self._lock:
            conn = self._get_connection()
            try:
                if project_id:
                    rows = conn.execute(
                        """
                        SELECT * FROM request_logs
                        WHERE project_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (project_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM request_logs
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()

                return [self._row_to_request_log(row) for row in rows]
            finally:
                conn.close()

    # ========== Helper methods ==========

    @staticmethod
    def _row_to_project(row: sqlite3.Row) -> StoredProject:
        """Convert a database row to StoredProject."""
        # owner columns may not exist in very old DBs — access safely
        keys = row.keys() if hasattr(row, "keys") else []
        return StoredProject(
            project_id=row["project_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            raw_text=row["raw_text"],
            project_name=row["project_name"],
            summary=row["summary"],
            project_type=row["project_type"],
            features_json=row["features_json"],
            tech_stack_json=row["tech_stack_json"],
            status=row["status"],
            artifact_path=row["artifact_path"],
            error_message=row["error_message"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            owner_client_id=row["owner_client_id"] if "owner_client_id" in keys else None,
            owner_user_id=row["owner_user_id"] if "owner_user_id" in keys else None,
            org_id=row["org_id"] if "org_id" in keys else None,
        )

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> StoredSession:
        """Convert a database row to StoredSession."""
        return StoredSession(
            session_id=row["session_id"],
            user_id=row["user_id"],
            messages=json.loads(row["messages"]),
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_artifact(row: sqlite3.Row) -> StoredArtifact:
        """Convert a database row to StoredArtifact."""
        keys = row.keys() if hasattr(row, "keys") else []
        return StoredArtifact(
            artifact_id=row["artifact_id"],
            project_id=row["project_id"],
            filename=row["filename"],
            file_path=row["file_path"],
            file_size_bytes=row["file_size_bytes"],
            format=row["format"],
            created_at=datetime.fromisoformat(row["created_at"]),
            owner_client_id=row["owner_client_id"] if "owner_client_id" in keys else None,
            owner_user_id=row["owner_user_id"] if "owner_user_id" in keys else None,
            org_id=row["org_id"] if "org_id" in keys else None,
        )

    @staticmethod
    def _row_to_request_log(row: sqlite3.Row) -> RequestLog:
        """Convert a database row to RequestLog."""
        return RequestLog(
            log_id=row["log_id"],
            project_id=row["project_id"],
            user_id=row["user_id"],
            raw_text=row["raw_text"],
            parsed_features=json.loads(row["parsed_features"]),
            parsed_tech_stack=json.loads(row["parsed_tech_stack"]),
            processing_time_ms=row["processing_time_ms"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
