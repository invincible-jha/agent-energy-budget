"""Pydantic v2 configuration schema for the semantic cache.

CacheConfig is validated at system boundaries (constructor args, YAML load).
All fields have sensible defaults so minimal configuration is required.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    """Configuration for the semantic cache.

    Parameters
    ----------
    exact_match_only:
        When True, only exact hash matches are returned and no embedding
        similarity search is performed.  Useful when no embedder is
        provided or when deterministic caching is required.
    similarity_threshold:
        Minimum cosine similarity score [0.0, 1.0] required to consider
        two prompts semantically equivalent.
    ttl_seconds:
        Time-to-live in seconds for each cache entry.  Entries older than
        this value are treated as misses and removed lazily on access.
        Set to 0 to disable TTL expiration.
    max_entries:
        Maximum number of entries the backend may hold.  Oldest (LRU)
        entries are evicted when the limit is reached.
    """

    exact_match_only: bool = False
    similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    ttl_seconds: int = Field(default=3600, ge=0)
    max_entries: int = Field(default=10000, ge=1)
