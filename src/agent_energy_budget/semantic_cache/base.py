"""Abstract base class for all semantic cache backends.

New backends must subclass CacheBackend and implement every abstract
method.  The interface is intentionally small so backends remain easy
to implement and test in isolation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class StoredEntry:
    """A single entry as stored by a backend.

    Parameters
    ----------
    key:
        The cache key (SHA-256 hex digest of the prompt).
    value:
        The cached response as a string.
    embedding:
        Optional embedding vector stored alongside the value for
        similarity search.  None when the embedder was not provided
        at write time.
    created_at:
        Unix timestamp (float) when this entry was written.
    """

    key: str
    value: str
    embedding: list[float] | None
    created_at: float


@dataclass(frozen=True)
class SimilarityMatch:
    """A candidate returned by a similarity search.

    Parameters
    ----------
    key:
        Cache key of the matching entry.
    value:
        Cached response of the matching entry.
    similarity_score:
        Cosine similarity between the query embedding and the stored
        embedding.  Always in [-1.0, 1.0].
    """

    key: str
    value: str
    similarity_score: float


class CacheBackend(ABC):
    """Abstract interface for semantic cache storage backends.

    Implementations are responsible for:
    - Persistent or ephemeral storage of (key, value, embedding) triples.
    - TTL enforcement (at minimum on read).
    - LRU eviction when max_entries is exceeded.
    - Thread-safe access when used from multiple threads.
    """

    @abstractmethod
    def get(self, key: str) -> StoredEntry | None:
        """Retrieve a stored entry by exact key.

        Implementations must return None when the key does not exist
        or when the entry has expired (TTL exceeded).

        Parameters
        ----------
        key:
            The SHA-256 hex digest to look up.

        Returns
        -------
        StoredEntry | None
            The stored entry, or None on miss / expiry.
        """
        ...

    @abstractmethod
    def put(
        self,
        key: str,
        value: str,
        embedding: list[float] | None,
        ttl: int,
    ) -> None:
        """Store or overwrite an entry.

        Parameters
        ----------
        key:
            The SHA-256 hex digest of the prompt.
        value:
            The LLM response to cache.
        embedding:
            Optional embedding vector for similarity search.
        ttl:
            Time-to-live in seconds.  0 means the entry never expires.
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a single entry.  No-op if the key does not exist.

        Parameters
        ----------
        key:
            The SHA-256 hex digest to remove.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the backend."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of currently stored (non-expired) entries.

        Returns
        -------
        int
            Number of live cache entries.
        """
        ...

    @abstractmethod
    def similarity_search(
        self,
        embedding: list[float],
        threshold: float,
        limit: int,
    ) -> list[SimilarityMatch]:
        """Find stored entries whose embeddings exceed *threshold* similarity.

        Parameters
        ----------
        embedding:
            Query embedding vector to compare against stored embeddings.
        threshold:
            Minimum cosine similarity score to include in results.
        limit:
            Maximum number of results to return.

        Returns
        -------
        list[SimilarityMatch]
            Matches sorted by similarity score descending.  Empty list
            when no stored embeddings meet the threshold.
        """
        ...
