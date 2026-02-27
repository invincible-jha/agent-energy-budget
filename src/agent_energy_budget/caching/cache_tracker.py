"""Cached token cost tracker for prompt-caching APIs.

Many LLM providers (Anthropic, OpenAI) offer prompt-caching at a
significant discount (50–90% off input token prices).  This module
provides :class:`CacheTokenTracker` to detect cached vs. non-cached
tokens from API response metadata and compute the actual cost applying
provider-specific cache discounts.

Cache detection
---------------
Responses include a ``usage`` dict with fields such as:

* ``cache_read_input_tokens``   — tokens served from cache (Anthropic-style)
* ``cached_tokens``             — tokens served from cache (OpenAI-style)
* ``cache_creation_input_tokens`` — tokens written to cache (Anthropic)

The tracker normalises these fields to a unified representation.

Usage
-----
::

    from agent_energy_budget.caching import CacheTokenTracker, CachePricingConfig

    config = CachePricingConfig(
        base_input_price_per_million=3.00,
        cache_read_discount=0.90,    # 90% off
        cache_write_premium=1.25,    # 25% premium for cache writes
    )
    tracker = CacheTokenTracker(config)

    stats = tracker.record_response(
        usage={"input_tokens": 1000, "cache_read_input_tokens": 800,
               "output_tokens": 200},
        base_output_price_per_million=15.00,
    )
    print(stats.cache_hit_rate)      # 0.80
    print(stats.total_cost_usd)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CachePricingConfig:
    """Per-provider cache pricing configuration.

    Parameters
    ----------
    base_input_price_per_million:
        Cost in USD per 1M non-cached input tokens.
    cache_read_discount:
        Fractional discount applied to cache-read tokens.
        A value of 0.90 means cache reads cost 10% of the full input price.
    cache_write_premium:
        Multiplier applied to cache-write (creation) tokens.
        A value of 1.25 means writing costs 125% of the full input price.
    output_price_override_per_million:
        If provided, overrides the per-call output price for cost calculations.
        Use when the tracker is the sole source of pricing data.
    """

    base_input_price_per_million: float = 3.00
    cache_read_discount: float = 0.10  # cache read = 10% of full price (90% off)
    cache_write_premium: float = 1.25  # cache write = 125% of full price
    output_price_override_per_million: float | None = None

    def __post_init__(self) -> None:
        if self.base_input_price_per_million < 0:
            raise ValueError("base_input_price_per_million must be non-negative.")
        if not (0.0 <= self.cache_read_discount <= 1.0):
            raise ValueError(
                f"cache_read_discount must be in [0, 1], "
                f"got {self.cache_read_discount!r}."
            )
        if self.cache_write_premium < 0:
            raise ValueError("cache_write_premium must be non-negative.")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheUsageRecord:
    """Normalized token usage extracted from a single API response.

    Parameters
    ----------
    total_input_tokens:
        Total input tokens (including any cached portions).
    cache_read_tokens:
        Tokens served from cache (discounted).
    cache_write_tokens:
        Tokens written to cache (may carry a premium).
    non_cached_input_tokens:
        Input tokens that were NOT served from cache.
    output_tokens:
        Output tokens generated.
    was_cached:
        True when at least one token was served from cache.
    cost_usd:
        Computed cost for this response.
    full_price_cost_usd:
        What this response would have cost at full price (no cache).
    savings_usd:
        Savings achieved through caching.
    recorded_at:
        ISO-8601 timestamp when this record was created.
    """

    total_input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    non_cached_input_tokens: int
    output_tokens: int
    was_cached: bool
    cost_usd: float
    full_price_cost_usd: float
    savings_usd: float
    recorded_at: str


@dataclass
class CacheTrackerStats:
    """Aggregate statistics across all recorded responses.

    Parameters
    ----------
    total_responses:
        Number of API responses recorded.
    cached_responses:
        Responses where at least one token was cache-served.
    cache_hit_rate:
        Fraction of responses that used cached tokens.
    total_input_tokens:
        Sum of all input tokens across all responses.
    total_cache_read_tokens:
        Total tokens served from cache.
    total_cache_write_tokens:
        Total tokens written to cache.
    total_output_tokens:
        Sum of all output tokens.
    total_cost_usd:
        Actual total cost after cache discounts.
    total_full_price_cost_usd:
        What the total would have cost at full price.
    total_savings_usd:
        Total savings achieved through caching.
    token_cache_hit_rate:
        Fraction of input tokens served from cache.
    """

    total_responses: int
    cached_responses: int
    cache_hit_rate: float
    total_input_tokens: int
    total_cache_read_tokens: int
    total_cache_write_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    total_full_price_cost_usd: float
    total_savings_usd: float
    token_cache_hit_rate: float


# ---------------------------------------------------------------------------
# CacheTokenTracker
# ---------------------------------------------------------------------------


class CacheTokenTracker:
    """Track cached vs. non-cached token usage and compute cache-adjusted costs.

    Thread-safe.  All state mutations are protected by a
    :class:`threading.Lock`.

    Parameters
    ----------
    config:
        :class:`CachePricingConfig` controlling cache discount rates.

    Example
    -------
    ::

        tracker = CacheTokenTracker()
        record = tracker.record_response(
            usage={"input_tokens": 1000, "cache_read_input_tokens": 800,
                   "output_tokens": 300},
            base_output_price_per_million=15.00,
        )
        print(f"Cache hit rate: {tracker.stats().cache_hit_rate:.0%}")
    """

    def __init__(self, config: CachePricingConfig | None = None) -> None:
        self._config = config or CachePricingConfig()
        self._lock = threading.Lock()
        self._records: list[CacheUsageRecord] = []

    @property
    def config(self) -> CachePricingConfig:
        """The pricing configuration for this tracker."""
        return self._config

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record_response(
        self,
        usage: dict[str, object],
        base_output_price_per_million: float | None = None,
    ) -> CacheUsageRecord:
        """Parse *usage* dict, compute costs, and record the result.

        Supports Anthropic-style keys:
        * ``input_tokens``
        * ``cache_read_input_tokens``
        * ``cache_creation_input_tokens``
        * ``output_tokens``

        And OpenAI-style keys:
        * ``prompt_tokens`` (alias for input_tokens)
        * ``cached_tokens`` (alias for cache_read_input_tokens)
        * ``completion_tokens`` (alias for output_tokens)

        Parameters
        ----------
        usage:
            Token usage dict from the API response.
        base_output_price_per_million:
            Output token price per 1M tokens.  Overrides the config
            ``output_price_override_per_million`` if provided.

        Returns
        -------
        CacheUsageRecord
            The normalized and costed record.
        """
        # Normalise token counts
        input_tokens = int(
            usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        )
        cache_read_tokens = int(
            usage.get("cache_read_input_tokens") or usage.get("cached_tokens") or 0
        )
        cache_write_tokens = int(
            usage.get("cache_creation_input_tokens") or 0
        )
        output_tokens = int(
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )

        # Non-cached input = total input minus what was cache-read
        non_cached_input = max(0, input_tokens - cache_read_tokens)

        was_cached = cache_read_tokens > 0

        # Compute costs
        output_price = (
            base_output_price_per_million
            or self._config.output_price_override_per_million
            or 0.0
        )
        record = self._compute_cost(
            non_cached_input=non_cached_input,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            output_tokens=output_tokens,
            total_input_tokens=input_tokens,
            output_price_per_million=output_price,
            was_cached=was_cached,
        )

        with self._lock:
            self._records.append(record)

        return record

    def stats(self) -> CacheTrackerStats:
        """Return aggregate statistics across all recorded responses.

        Returns
        -------
        CacheTrackerStats
            Thread-safe snapshot of cumulative metrics.
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return CacheTrackerStats(
                total_responses=0,
                cached_responses=0,
                cache_hit_rate=0.0,
                total_input_tokens=0,
                total_cache_read_tokens=0,
                total_cache_write_tokens=0,
                total_output_tokens=0,
                total_cost_usd=0.0,
                total_full_price_cost_usd=0.0,
                total_savings_usd=0.0,
                token_cache_hit_rate=0.0,
            )

        total_responses = len(records)
        cached_responses = sum(1 for r in records if r.was_cached)
        total_input = sum(r.total_input_tokens for r in records)
        total_cache_read = sum(r.cache_read_tokens for r in records)
        total_cache_write = sum(r.cache_write_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_cost = sum(r.cost_usd for r in records)
        total_full_price = sum(r.full_price_cost_usd for r in records)
        total_savings = sum(r.savings_usd for r in records)

        cache_hit_rate = cached_responses / total_responses
        token_cache_hit_rate = (
            total_cache_read / total_input if total_input > 0 else 0.0
        )

        return CacheTrackerStats(
            total_responses=total_responses,
            cached_responses=cached_responses,
            cache_hit_rate=round(cache_hit_rate, 6),
            total_input_tokens=total_input,
            total_cache_read_tokens=total_cache_read,
            total_cache_write_tokens=total_cache_write,
            total_output_tokens=total_output,
            total_cost_usd=round(total_cost, 8),
            total_full_price_cost_usd=round(total_full_price, 8),
            total_savings_usd=round(total_savings, 8),
            token_cache_hit_rate=round(token_cache_hit_rate, 6),
        )

    def reset(self) -> None:
        """Clear all recorded data."""
        with self._lock:
            self._records.clear()

    def record_count(self) -> int:
        """Return the number of recorded responses."""
        with self._lock:
            return len(self._records)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_cost(
        self,
        non_cached_input: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        output_tokens: int,
        total_input_tokens: int,
        output_price_per_million: float,
        was_cached: bool,
    ) -> CacheUsageRecord:
        """Compute cache-adjusted and full-price costs for a single response."""
        base_input_price = self._config.base_input_price_per_million
        cache_read_price = base_input_price * self._config.cache_read_discount
        cache_write_price = base_input_price * self._config.cache_write_premium

        # Actual cost
        non_cached_cost = (non_cached_input / 1_000_000) * base_input_price
        cache_read_cost = (cache_read_tokens / 1_000_000) * cache_read_price
        cache_write_cost = (cache_write_tokens / 1_000_000) * cache_write_price
        output_cost = (output_tokens / 1_000_000) * output_price_per_million
        actual_cost = non_cached_cost + cache_read_cost + cache_write_cost + output_cost

        # Full-price cost (no cache discount)
        full_input_cost = (total_input_tokens / 1_000_000) * base_input_price
        full_cost = full_input_cost + output_cost

        savings = full_cost - actual_cost

        return CacheUsageRecord(
            total_input_tokens=total_input_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            non_cached_input_tokens=non_cached_input,
            output_tokens=output_tokens,
            was_cached=was_cached,
            cost_usd=round(actual_cost, 8),
            full_price_cost_usd=round(full_cost, 8),
            savings_usd=round(savings, 8),
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )
