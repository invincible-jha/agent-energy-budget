"""Thread-safe in-memory LRU cache backend.

Uses an OrderedDict for O(1) LRU eviction, threading.Lock for thread
safety, and lazy TTL expiration on read.  Suitable for single-process
deployments where persistence across restarts is not required.
"""
from __future__ import annotations

import time
import threading
from collections import OrderedDict

from agent_energy_budget.semantic_cache.base import (
    CacheBackend,
    SimilarityMatch,
    StoredEntry,
)
from agent_energy_budget.semantic_cache.similarity import cosine_similarity


class InMemoryCacheBackend(CacheBackend):
    """OrderedDict-based LRU cache backend with per-entry TTL.

    Parameters
    ----------
    max_entries:
        Maximum number of entries to store before evicting the LRU entry.
        Must be >= 1.

    Thread safety
    -------------
    All public methods acquire ``_lock`` (a threading.Lock) before
    accessing ``_store``.  This makes the backend safe to share across
    multiple threads within the same process.
    """

    def __init__(self, max_entries: int = 10000) -> None:
        if max_entries < 1:
            raise ValueError(
                f"max_entries must be >= 1, got {max_entries}."
            )
        self._max_entries: int = max_entries
        self._store: OrderedDict[str, tuple[StoredEntry, float]] = OrderedDict()
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_expired(self, expiry_at: float) -> bool:
        """Return True when the entry has passed its expiry timestamp.

        An expiry_at value of 0.0 means the entry never expires.
        """
        return expiry_at > 0.0 and time.monotonic() > expiry_at

    def _expiry_timestamp(self, ttl: int) -> float:
        """Compute the absolute expiry timestamp from a TTL in seconds.

        Returns 0.0 when ttl is 0 (meaning no expiration).
        """
        return (time.monotonic() + ttl) if ttl > 0 else 0.0

    # ------------------------------------------------------------------
    # CacheBackend interface
    # ------------------------------------------------------------------

    def get(self, key: str) -> StoredEntry | None:
        """Retrieve an entry by exact key.

        Moves the entry to the end of the OrderedDict (marking it as
        recently used) if found and not expired.  Deletes expired entries
        lazily.

        Parameters
        ----------
        key:
            SHA-256 hex digest to look up.

        Returns
        -------
        StoredEntry | None
            The stored entry, or None on miss or expiry.
        """
        with self._lock:
            if key not in self._store:
                return None
            entry, expiry_at = self._store[key]
            if self._is_expired(expiry_at):
                del self._store[key]
                return None
            # Move to end to mark as recently used
            self._store.move_to_end(key)
            return entry

    def put(
        self,
        key: str,
        value: str,
        embedding: list[float] | None,
        ttl: int,
    ) -> None:
        """Store or overwrite a cache entry.

        Evicts the least-recently-used entry when max_entries is exceeded.

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
        expiry_at = self._expiry_timestamp(ttl)
        entry = StoredEntry(
            key=key,
            value=value,
            embedding=embedding,
            created_at=time.monotonic(),
        )
        with self._lock:
            if key in self._store:
                # Overwrite existing entry and refresh LRU position
                self._store.move_to_end(key)
                self._store[key] = (entry, expiry_at)
                return
            self._store[key] = (entry, expiry_at)
            self._store.move_to_end(key)
            # Evict LRU entries until within limit
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def delete(self, key: str) -> None:
        """Remove an entry by key.  No-op if the key does not exist.

        Parameters
        ----------
        key:
            SHA-256 hex digest to remove.
        """
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries from the in-memory store."""
        with self._lock:
            self._store.clear()

    def count(self) -> int:
        """Return the number of non-expired entries.

        Performs a full scan with lazy expiry removal.  This is O(n) but
        acceptable for periodic monitoring calls.

        Returns
        -------
        int
            Count of live (non-expired) entries.
        """
        with self._lock:
            now = time.monotonic()
            expired_keys = [
                k for k, (_, expiry_at) in self._store.items()
                if expiry_at > 0.0 and now > expiry_at
            ]
            for k in expired_keys:
                del self._store[k]
            return len(self._store)

    def similarity_search(
        self,
        embedding: list[float],
        threshold: float,
        limit: int,
    ) -> list[SimilarityMatch]:
        """Scan all stored embeddings for cosine similarity above threshold.

        Expired entries are excluded from the search and cleaned up
        lazily during the scan.

        Parameters
        ----------
        embedding:
            Query embedding vector.
        threshold:
            Minimum cosine similarity to include in results.
        limit:
            Maximum number of results to return.

        Returns
        -------
        list[SimilarityMatch]
            Matches sorted by similarity score descending.
        """
        matches: list[SimilarityMatch] = []
        now = time.monotonic()

        with self._lock:
            expired_keys: list[str] = []
            for key, (entry, expiry_at) in self._store.items():
                if expiry_at > 0.0 and now > expiry_at:
                    expired_keys.append(key)
                    continue
                if entry.embedding is None:
                    continue
                try:
                    score = cosine_similarity(embedding, entry.embedding)
                except ValueError:
                    continue
                if score >= threshold:
                    matches.append(
                        SimilarityMatch(key=key, value=entry.value, similarity_score=score)
                    )
            for k in expired_keys:
                del self._store[k]

        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches[:limit]
