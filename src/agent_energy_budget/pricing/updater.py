"""Remote and local pricing update utilities.

PricingUpdater can fetch a JSON pricing file from a URL and merge new
entries into PROVIDER_PRICING at runtime. This is useful for keeping
rates current without releasing a new package version.

Expected JSON schema
--------------------
A mapping of model name -> pricing fields::

    {
        "claude-opus-4": {
            "provider": "anthropic",
            "tier": "premium",
            "input_per_million": 15.0,
            "output_per_million": 75.0,
            "context_window": 200000,
            "supports_vision": true
        }
    }
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Union

from agent_energy_budget.pricing.tables import (
    MODEL_TIERS,
    PROVIDER_PRICING,
    ModelPricing,
    ModelTier,
    ProviderName,
)

logger = logging.getLogger(__name__)

_DEFAULT_PRICING_URL = (
    "https://raw.githubusercontent.com/aumos-ai/agent-energy-budget/main"
    "/pricing/current.json"
)


class PricingUpdateError(RuntimeError):
    """Raised when a remote pricing fetch or parse fails."""


class PricingUpdater:
    """Fetch and apply pricing updates from a remote JSON source.

    Parameters
    ----------
    pricing_url:
        URL of the JSON pricing file. Defaults to the official repo URL.
    timeout_seconds:
        HTTP request timeout. Defaults to 10.
    """

    def __init__(
        self,
        pricing_url: str = _DEFAULT_PRICING_URL,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._pricing_url = pricing_url
        self._timeout = timeout_seconds

    # ------------------------------------------------------------------
    # Remote fetch
    # ------------------------------------------------------------------

    def fetch_remote_pricing(self) -> dict[str, ModelPricing]:
        """Fetch and parse pricing data from the configured URL.

        Returns
        -------
        dict[str, ModelPricing]
            Parsed pricing records keyed by model name.

        Raises
        ------
        PricingUpdateError
            On network error, non-200 status, or JSON parse failure.
        """
        try:
            import urllib.request

            req = urllib.request.Request(
                self._pricing_url,
                headers={"User-Agent": "agent-energy-budget/0.1.0"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                raw_bytes: bytes = response.read()
        except Exception as exc:
            raise PricingUpdateError(
                f"Failed to fetch pricing from {self._pricing_url}: {exc}"
            ) from exc

        try:
            raw_data: object = json.loads(raw_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise PricingUpdateError(f"Invalid JSON in pricing response: {exc}") from exc

        if not isinstance(raw_data, dict):
            raise PricingUpdateError(
                f"Expected a JSON object at root, got {type(raw_data).__name__}"
            )

        return _parse_pricing_dict(raw_data)

    def apply_updates(
        self,
        updates: dict[str, ModelPricing],
        *,
        overwrite_existing: bool = True,
    ) -> list[str]:
        """Merge *updates* into the in-process PROVIDER_PRICING table.

        Parameters
        ----------
        updates:
            New or updated pricing records.
        overwrite_existing:
            When True (default), updated records replace existing ones.
            When False, existing records are preserved.

        Returns
        -------
        list[str]
            List of model names that were actually updated/added.
        """
        applied: list[str] = []
        for model_name, pricing in updates.items():
            if model_name in PROVIDER_PRICING and not overwrite_existing:
                logger.debug("Skipping existing pricing for %r", model_name)
                continue
            PROVIDER_PRICING[model_name] = pricing
            # Register in tier table if not already present
            tier_list = MODEL_TIERS.get(pricing.tier, [])
            if model_name not in tier_list:
                tier_list.append(model_name)
                MODEL_TIERS[pricing.tier] = tier_list
            applied.append(model_name)
            logger.info("Applied pricing update for model %r", model_name)
        return applied

    def refresh(self, *, overwrite_existing: bool = True) -> list[str]:
        """Fetch remote pricing and apply it in one step.

        Returns
        -------
        list[str]
            Model names that were updated/added.

        Raises
        ------
        PricingUpdateError
            On network or parse errors.
        """
        updates = self.fetch_remote_pricing()
        return self.apply_updates(updates, overwrite_existing=overwrite_existing)

    # ------------------------------------------------------------------
    # Local file loading
    # ------------------------------------------------------------------

    def load_custom_pricing(
        self,
        file_path: Union[str, pathlib.Path],
        *,
        overwrite_existing: bool = True,
    ) -> list[str]:
        """Load pricing overrides from a local JSON file.

        The file format is identical to the remote pricing schema.

        Parameters
        ----------
        file_path:
            Path to the local JSON pricing file.
        overwrite_existing:
            Same semantics as :meth:`apply_updates`.

        Returns
        -------
        list[str]
            Model names that were updated/added.

        Raises
        ------
        PricingUpdateError
            On file read or parse errors.
        """
        path = pathlib.Path(file_path)
        try:
            raw_data: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PricingUpdateError(
                f"Failed to load pricing file {path}: {exc}"
            ) from exc

        if not isinstance(raw_data, dict):
            raise PricingUpdateError(
                f"Expected a JSON object, got {type(raw_data).__name__}"
            )

        updates = _parse_pricing_dict(raw_data)
        return self.apply_updates(updates, overwrite_existing=overwrite_existing)


def _parse_pricing_dict(data: dict[str, object]) -> dict[str, ModelPricing]:
    """Parse a raw JSON dict into a model_name -> ModelPricing mapping.

    Parameters
    ----------
    data:
        Raw dict from JSON.

    Returns
    -------
    dict[str, ModelPricing]
        Parsed records (invalid entries are logged and skipped).
    """
    result: dict[str, ModelPricing] = {}
    for model_name, raw_entry in data.items():
        if not isinstance(raw_entry, dict):
            logger.warning("Skipping non-dict entry for model %r", model_name)
            continue
        try:
            pricing = ModelPricing(
                model=model_name,
                provider=ProviderName(raw_entry.get("provider", "custom")),
                tier=ModelTier(raw_entry.get("tier", "efficient")),
                input_per_million=float(raw_entry["input_per_million"]),
                output_per_million=float(raw_entry["output_per_million"]),
                context_window=int(raw_entry.get("context_window", 0)),
                supports_vision=bool(raw_entry.get("supports_vision", False)),
            )
            result[model_name] = pricing
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping invalid pricing entry for %r: %s", model_name, exc)

    return result
