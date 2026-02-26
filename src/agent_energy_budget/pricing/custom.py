"""Custom pricing management for fine-tuned models and volume discounts.

Allows operators to register custom model pricing entries (fine-tuned
models, private deployments, volume-discounted rates) that are layered
on top of the standard PROVIDER_PRICING table.
"""
from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Union

from agent_energy_budget.pricing.tables import (
    MODEL_TIERS,
    PROVIDER_PRICING,
    ModelPricing,
    ModelTier,
    ProviderName,
)

logger = logging.getLogger(__name__)


@dataclass
class VolumeDiscount:
    """Volume discount rule for a specific model.

    Parameters
    ----------
    model:
        Model identifier this discount applies to.
    monthly_token_threshold:
        Minimum monthly token count to qualify (input + output combined).
    input_discount_pct:
        Percentage reduction on input token costs (0–100).
    output_discount_pct:
        Percentage reduction on output token costs (0–100).
    """

    model: str
    monthly_token_threshold: int
    input_discount_pct: float
    output_discount_pct: float

    def apply_to(self, pricing: ModelPricing, monthly_tokens_used: int) -> ModelPricing:
        """Return a new ModelPricing with discounted rates if threshold is met.

        Parameters
        ----------
        pricing:
            Base pricing to apply discounts to.
        monthly_tokens_used:
            Actual tokens consumed this month.

        Returns
        -------
        ModelPricing
            Discounted pricing (or original if threshold not met).
        """
        if monthly_tokens_used < self.monthly_token_threshold:
            return pricing

        new_input = pricing.input_per_million * (1 - self.input_discount_pct / 100)
        new_output = pricing.output_per_million * (1 - self.output_discount_pct / 100)
        return ModelPricing(
            model=pricing.model,
            provider=pricing.provider,
            tier=pricing.tier,
            input_per_million=max(0.0, new_input),
            output_per_million=max(0.0, new_output),
            context_window=pricing.context_window,
            supports_vision=pricing.supports_vision,
        )


@dataclass
class CustomPricingManager:
    """Manage custom and volume-discounted model pricing.

    Custom entries are kept in an isolated registry so they do not pollute
    the canonical PROVIDER_PRICING table until explicitly committed.

    Parameters
    ----------
    persist_path:
        Optional filesystem path for JSON persistence. When provided,
        custom entries are saved/loaded automatically.
    """

    persist_path: pathlib.Path | None = None
    _custom: dict[str, ModelPricing] = field(default_factory=dict, init=False)
    _discounts: list[VolumeDiscount] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.persist_path is not None:
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Custom model registration
    # ------------------------------------------------------------------

    def register(
        self,
        model: str,
        provider: str,
        tier: str,
        input_per_million: float,
        output_per_million: float,
        context_window: int = 0,
        supports_vision: bool = False,
    ) -> ModelPricing:
        """Register a custom model pricing entry.

        Parameters
        ----------
        model:
            Unique identifier for the custom model.
        provider:
            Provider name string (will be coerced to ProviderName.CUSTOM if
            not a recognised provider).
        tier:
            Tier string ("nano", "efficient", "standard", "premium").
        input_per_million:
            Input cost in USD per million tokens.
        output_per_million:
            Output cost in USD per million tokens.
        context_window:
            Maximum context window. 0 = unspecified.
        supports_vision:
            Whether the model accepts images.

        Returns
        -------
        ModelPricing
            The created pricing record.
        """
        try:
            provider_enum = ProviderName(provider.lower())
        except ValueError:
            provider_enum = ProviderName.CUSTOM

        try:
            tier_enum = ModelTier(tier.lower())
        except ValueError:
            tier_enum = ModelTier.EFFICIENT

        pricing = ModelPricing(
            model=model,
            provider=provider_enum,
            tier=tier_enum,
            input_per_million=input_per_million,
            output_per_million=output_per_million,
            context_window=context_window,
            supports_vision=supports_vision,
        )
        self._custom[model] = pricing
        logger.info("Registered custom pricing for model %r", model)
        if self.persist_path is not None:
            self._save_to_disk()
        return pricing

    def remove(self, model: str) -> bool:
        """Remove a custom pricing entry.

        Parameters
        ----------
        model:
            Model identifier to remove.

        Returns
        -------
        bool
            True if the entry existed and was removed; False otherwise.
        """
        existed = model in self._custom
        if existed:
            del self._custom[model]
            logger.info("Removed custom pricing for model %r", model)
            if self.persist_path is not None:
                self._save_to_disk()
        return existed

    def get(self, model: str) -> ModelPricing | None:
        """Retrieve a custom pricing entry by model name.

        Parameters
        ----------
        model:
            Model identifier to look up.

        Returns
        -------
        ModelPricing | None
            The custom pricing record, or None if not found.
        """
        return self._custom.get(model)

    def list_custom(self) -> list[ModelPricing]:
        """Return all registered custom pricing entries."""
        return list(self._custom.values())

    # ------------------------------------------------------------------
    # Volume discounts
    # ------------------------------------------------------------------

    def add_volume_discount(self, discount: VolumeDiscount) -> None:
        """Register a volume discount rule.

        Parameters
        ----------
        discount:
            The VolumeDiscount configuration to add.
        """
        self._discounts.append(discount)
        logger.info(
            "Added volume discount for %r at %d tokens/month",
            discount.model,
            discount.monthly_token_threshold,
        )

    def get_effective_pricing(
        self, model: str, monthly_tokens_used: int = 0
    ) -> ModelPricing | None:
        """Resolve effective pricing for a model, applying applicable discounts.

        Checks custom registry first, then falls back to PROVIDER_PRICING.
        Volume discounts are applied if threshold is met.

        Parameters
        ----------
        model:
            Model identifier.
        monthly_tokens_used:
            Tokens consumed this month (for discount evaluation).

        Returns
        -------
        ModelPricing | None
            Effective pricing with discounts applied, or None if not found.
        """
        pricing = self._custom.get(model) or PROVIDER_PRICING.get(model)
        if pricing is None:
            return None

        for discount in self._discounts:
            if discount.model == model:
                pricing = discount.apply_to(pricing, monthly_tokens_used)

        return pricing

    # ------------------------------------------------------------------
    # Commit to global table
    # ------------------------------------------------------------------

    def commit_to_global(self, *, overwrite_existing: bool = True) -> list[str]:
        """Merge all custom pricing entries into the global PROVIDER_PRICING.

        Parameters
        ----------
        overwrite_existing:
            When True, existing entries in PROVIDER_PRICING are replaced.

        Returns
        -------
        list[str]
            Model names that were committed.
        """
        committed: list[str] = []
        for model_name, pricing in self._custom.items():
            if model_name in PROVIDER_PRICING and not overwrite_existing:
                continue
            PROVIDER_PRICING[model_name] = pricing
            tier_list = MODEL_TIERS.get(pricing.tier, [])
            if model_name not in tier_list:
                tier_list.append(model_name)
                MODEL_TIERS[pricing.tier] = tier_list
            committed.append(model_name)
        logger.info("Committed %d custom pricing entries to global table", len(committed))
        return committed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_to_disk(self) -> None:
        """Persist custom pricing to JSON file."""
        if self.persist_path is None:
            return
        serialisable = {
            name: {
                "provider": p.provider.value,
                "tier": p.tier.value,
                "input_per_million": p.input_per_million,
                "output_per_million": p.output_per_million,
                "context_window": p.context_window,
                "supports_vision": p.supports_vision,
            }
            for name, p in self._custom.items()
        }
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(
            json.dumps(serialisable, indent=2), encoding="utf-8"
        )

    def _load_from_disk(self) -> None:
        """Load persisted custom pricing from JSON file."""
        if self.persist_path is None or not self.persist_path.exists():
            return
        try:
            raw: object = json.loads(
                self.persist_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            logger.warning("Failed to load custom pricing from %s: %s", self.persist_path, exc)
            return

        if not isinstance(raw, dict):
            logger.warning("Custom pricing file has unexpected format; skipping")
            return

        for model_name, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            try:
                self.register(
                    model=model_name,
                    provider=str(entry.get("provider", "custom")),
                    tier=str(entry.get("tier", "efficient")),
                    input_per_million=float(entry["input_per_million"]),
                    output_per_million=float(entry["output_per_million"]),
                    context_window=int(entry.get("context_window", 0)),
                    supports_vision=bool(entry.get("supports_vision", False)),
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping invalid custom entry %r: %s", model_name, exc)

    def export_to_file(self, file_path: Union[str, pathlib.Path]) -> None:
        """Export custom pricing entries to a JSON file.

        Parameters
        ----------
        file_path:
            Destination file path.
        """
        path = pathlib.Path(file_path)
        old_path = self.persist_path
        self.persist_path = path
        self._save_to_disk()
        self.persist_path = old_path
