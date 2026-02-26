"""Cached fallback degradation strategy.

When budget is exhausted, signals to the caller that a cached response
should be returned instead of making a new LLM API call. Manages an
in-memory LRU response cache keyed on (model, prompt_hash).
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Union

from agent_energy_budget.degradation.base import DegradationResult, DegradationStrategyBase
from agent_energy_budget.pricing.tables import get_pricing


@dataclass
class CachedResponse:
    """A single cached LLM response.

    Parameters
    ----------
    model:
        Model that produced this response.
    prompt_hash:
        SHA-256 of the normalised prompt text.
    response:
        The cached response text or structured output.
    cached_at:
        UTC timestamp when this entry was stored.
    hit_count:
        Number of times this entry has been served from cache.
    original_cost_usd:
        Cost of the original API call.
    """

    model: str
    prompt_hash: str
    response: str
    cached_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hit_count: int = 0
    original_cost_usd: float = 0.0


class CachedFallbackStrategy(DegradationStrategyBase):
    """Return cached responses when budget is exhausted.

    Uses an LRU eviction policy. Entries are keyed by ``(model, prompt_hash)``
    so that similar prompts for different models are cached independently.

    Parameters
    ----------
    max_cache_size:
        Maximum number of entries before LRU eviction occurs (default 256).
    """

    def __init__(self, max_cache_size: int = 256) -> None:
        if max_cache_size < 1:
            raise ValueError("max_cache_size must be >= 1")
        self._max_size = max_cache_size
        self._cache: OrderedDict[str, CachedResponse] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Degradation strategy interface
    # ------------------------------------------------------------------

    def apply(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        remaining_budget: float,
    ) -> DegradationResult:
        """Signal that a cached fallback should be used.

        This strategy does not inspect the cache itself — it just returns
        a ``use_cache`` result. The caller should then invoke
        :meth:`get_cached_response` with the actual prompt text.

        Parameters
        ----------
        model:
            Requested model.
        input_tokens:
            Expected input tokens (informational only).
        output_tokens:
            Expected output tokens (informational only).
        remaining_budget:
            Available USD.

        Returns
        -------
        DegradationResult
            ``can_proceed=False`` with ``action="use_cache"``.
        """
        try:
            pricing = get_pricing(model)
            estimated_cost = pricing.cost_for_tokens(input_tokens, output_tokens)
            message = (
                f"Budget exhausted (remaining ${remaining_budget:.6f}, "
                f"estimated ${estimated_cost:.6f}). Returning cached response."
            )
        except KeyError:
            message = (
                f"Budget exhausted (remaining ${remaining_budget:.6f}). "
                "Returning cached response."
            )

        return DegradationResult(
            can_proceed=False,
            recommended_model=model,
            max_tokens=0,
            action="use_cache",
            message=message,
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(model: str, prompt_hash: str) -> str:
        return f"{model}:{prompt_hash}"

    @staticmethod
    def _hash_prompt(prompt: Union[str, list[dict[str, str]]]) -> str:
        """Compute a stable SHA-256 hash of a prompt string or messages list."""
        if isinstance(prompt, list):
            normalised = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
        else:
            normalised = prompt.strip()
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    def store(
        self,
        model: str,
        prompt: Union[str, list[dict[str, str]]],
        response: str,
        original_cost_usd: float = 0.0,
    ) -> str:
        """Store a response in the cache.

        Parameters
        ----------
        model:
            Model that produced the response.
        prompt:
            The input prompt (string or messages list) used to generate it.
        response:
            The response text to cache.
        original_cost_usd:
            Cost of the original API call for auditing.

        Returns
        -------
        str
            The prompt hash (can be used for direct lookup).
        """
        prompt_hash = self._hash_prompt(prompt)
        key = self._cache_key(model, prompt_hash)

        with self._lock:
            if key in self._cache:
                # Refresh position in LRU
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    # Evict least recently used
                    self._cache.popitem(last=False)
                self._cache[key] = CachedResponse(
                    model=model,
                    prompt_hash=prompt_hash,
                    response=response,
                    original_cost_usd=original_cost_usd,
                )

        return prompt_hash

    def get_cached_response(
        self,
        model: str,
        prompt: Union[str, list[dict[str, str]]],
    ) -> CachedResponse | None:
        """Retrieve a cached response.

        Parameters
        ----------
        model:
            Model identifier.
        prompt:
            The prompt to look up.

        Returns
        -------
        CachedResponse | None
            The cached response entry, or None on cache miss.
        """
        prompt_hash = self._hash_prompt(prompt)
        key = self._cache_key(model, prompt_hash)

        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                entry.hit_count += 1
                self._cache.move_to_end(key)
            return entry

    def invalidate(self, model: str, prompt: Union[str, list[dict[str, str]]]) -> bool:
        """Remove a specific cache entry.

        Parameters
        ----------
        model:
            Model identifier.
        prompt:
            The prompt whose entry should be removed.

        Returns
        -------
        bool
            True if an entry was found and removed; False on cache miss.
        """
        prompt_hash = self._hash_prompt(prompt)
        key = self._cache_key(model, prompt_hash)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Remove all cache entries.

        Returns
        -------
        int
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return len(self._cache)

    def stats(self) -> dict[str, object]:
        """Return cache statistics.

        Returns
        -------
        dict[str, object]
            Dictionary with size, max_size, and total hit count.
        """
        with self._lock:
            total_hits = sum(e.hit_count for e in self._cache.values())
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "total_hits": total_hits,
            }
