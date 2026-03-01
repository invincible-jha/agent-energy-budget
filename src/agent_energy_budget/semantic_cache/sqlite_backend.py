"""SQLite-backed persistent cache backend.

Uses stdlib sqlite3 for storage, JSON-encodes embeddings, and computes
cosine similarity on the application side.  Suitable for single-process
deployments that need persistence across restarts without an external
service.

Thread safety: sqlite3 connections are created per-thread via a
threading.local so that Connection objects are never shared across
threads.  The database file itself handles its own locking.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time

from agent_energy_budget.semantic_cache.base import (
    CacheBackend,
    SimilarityMatch,
    StoredEntry,
)
from agent_energy_budget.semantic_cache.similarity import cosine_similarity

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS semantic_cache (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    embedding  TEXT,
    created_at REAL NOT NULL,
    expiry_at  REAL NOT NULL
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_expiry ON semantic_cache (expiry_at);
"""


class SQLiteCacheBackend(CacheBackend):
    """SQLite-backed cache backend with TTL and LRU eviction.

    Parameters
    ----------
    db_path:
        Filesystem path for the SQLite database file.
        Use ``":memory:"`` for an in-process, non-persistent database
        (useful in tests).
    max_entries:
        Maximum number of rows to keep.  When the limit is exceeded,
        the oldest entries (by creation time) are evicted.

    Thread safety
    -------------
    A threading.local is used so each thread gets its own sqlite3
    Connection.  SQLite WAL mode is enabled to improve read concurrency.
    """

    def __init__(self, db_path: str = ":memory:", max_entries: int = 10000) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}.")
        self._db_path = db_path
        self._max_entries = max_entries
        self._local = threading.local()
        # Initialise the schema on the calling thread
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.execute(_CREATE_INDEX_SQL)
        self._conn.commit()

    @property
    def _conn(self) -> sqlite3.Connection:
        """Return (or lazily create) the per-thread sqlite3 connection."""
        if not hasattr(self._local, "connection"):
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
            conn.commit()
            self._local.connection = conn
        return self._local.connection  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expiry_timestamp(self, ttl: int) -> float:
        """Return absolute expiry time; 0.0 means no expiration."""
        return (time.time() + ttl) if ttl > 0 else 0.0

    def _is_expired(self, expiry_at: float) -> bool:
        """Return True when the entry has passed its wall-clock expiry."""
        return expiry_at > 0.0 and time.time() > expiry_at

    def _evict_oldest_if_needed(self) -> None:
        """Delete the oldest rows when max_entries is exceeded."""
        row = self._conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()
        count: int = row[0]
        if count <= self._max_entries:
            return
        overflow = count - self._max_entries
        self._conn.execute(
            """
            DELETE FROM semantic_cache
            WHERE key IN (
                SELECT key FROM semantic_cache
                ORDER BY created_at ASC
                LIMIT ?
            )
            """,
            (overflow,),
        )

    # ------------------------------------------------------------------
    # CacheBackend interface
    # ------------------------------------------------------------------

    def get(self, key: str) -> StoredEntry | None:
        """Retrieve an entry by exact key, returning None on miss or expiry.

        Parameters
        ----------
        key:
            SHA-256 hex digest to look up.

        Returns
        -------
        StoredEntry | None
        """
        row = self._conn.execute(
            "SELECT key, value, embedding, created_at, expiry_at "
            "FROM semantic_cache WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if self._is_expired(row["expiry_at"]):
            self._conn.execute("DELETE FROM semantic_cache WHERE key = ?", (key,))
            self._conn.commit()
            return None
        embedding: list[float] | None = None
        if row["embedding"] is not None:
            try:
                embedding = json.loads(row["embedding"])
            except (json.JSONDecodeError, ValueError):
                embedding = None
        return StoredEntry(
            key=row["key"],
            value=row["value"],
            embedding=embedding,
            created_at=row["created_at"],
        )

    def put(
        self,
        key: str,
        value: str,
        embedding: list[float] | None,
        ttl: int,
    ) -> None:
        """Store or overwrite a cache entry.

        Parameters
        ----------
        key:
            SHA-256 hex digest of the prompt.
        value:
            LLM response string.
        embedding:
            Optional embedding vector.
        ttl:
            Time-to-live in seconds; 0 means no expiration.
        """
        embedding_json: str | None = json.dumps(embedding) if embedding is not None else None
        expiry_at = self._expiry_timestamp(ttl)
        created_at = time.time()
        self._conn.execute(
            """
            INSERT INTO semantic_cache (key, value, embedding, created_at, expiry_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value      = excluded.value,
                embedding  = excluded.embedding,
                created_at = excluded.created_at,
                expiry_at  = excluded.expiry_at
            """,
            (key, value, embedding_json, created_at, expiry_at),
        )
        self._evict_oldest_if_needed()
        self._conn.commit()

    def delete(self, key: str) -> None:
        """Remove a single entry.  No-op if the key does not exist.

        Parameters
        ----------
        key:
            SHA-256 hex digest to remove.
        """
        self._conn.execute("DELETE FROM semantic_cache WHERE key = ?", (key,))
        self._conn.commit()

    def clear(self) -> None:
        """Remove all entries from the database."""
        self._conn.execute("DELETE FROM semantic_cache")
        self._conn.commit()

    def count(self) -> int:
        """Return the number of non-expired entries.

        Performs a DELETE of all expired rows before counting.

        Returns
        -------
        int
        """
        now = time.time()
        self._conn.execute(
            "DELETE FROM semantic_cache WHERE expiry_at > 0 AND expiry_at <= ?",
            (now,),
        )
        self._conn.commit()
        row = self._conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()
        return int(row[0])

    def similarity_search(
        self,
        embedding: list[float],
        threshold: float,
        limit: int,
    ) -> list[SimilarityMatch]:
        """Scan all non-expired embeddings for cosine similarity above threshold.

        Cosine similarity is computed on the application side after loading
        all stored embeddings.  Expired entries are excluded and lazily
        cleaned up during the scan.

        Parameters
        ----------
        embedding:
            Query embedding vector.
        threshold:
            Minimum cosine similarity to include in results.
        limit:
            Maximum number of results.

        Returns
        -------
        list[SimilarityMatch]
            Sorted by similarity score descending.
        """
        now = time.time()
        rows = self._conn.execute(
            """
            SELECT key, value, embedding
            FROM semantic_cache
            WHERE embedding IS NOT NULL
              AND (expiry_at = 0 OR expiry_at > ?)
            """,
            (now,),
        ).fetchall()

        matches: list[SimilarityMatch] = []
        for row in rows:
            try:
                stored_embedding: list[float] = json.loads(row["embedding"])
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            try:
                score = cosine_similarity(embedding, stored_embedding)
            except ValueError:
                continue
            if score >= threshold:
                matches.append(
                    SimilarityMatch(
                        key=row["key"],
                        value=row["value"],
                        similarity_score=score,
                    )
                )

        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches[:limit]
