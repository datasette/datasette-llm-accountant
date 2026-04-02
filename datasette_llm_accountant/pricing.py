"""
LLM pricing lookup and cost calculation.
"""

from __future__ import annotations

from typing import Optional
from abc import ABC, abstractmethod
from llm.models import AsyncResponse, Usage
import httpx


_PRICING_URL = "https://simonw.github.io/llm-prices/current-v1.json"
_PRICING_TIMEOUT = 10.0


class Nanocents(int):
    """
    An integer amount in nanocents (1/1,000,000,000 of a cent).

    Wraps int to make units explicit in APIs and prevent accidentally
    passing raw dollar amounts or cents where nanocents are expected.
    """

    _NANOCENTS_PER_USD = 100_000_000_000
    _NANOCENTS_PER_CENT = 1_000_000_000

    @classmethod
    def from_usd(cls, usd: float) -> Nanocents:
        return cls(int(usd * cls._NANOCENTS_PER_USD))

    @classmethod
    def from_cents(cls, cents: float) -> Nanocents:
        return cls(int(cents * cls._NANOCENTS_PER_CENT))

    def to_usd(self) -> float:
        return int(self) / self._NANOCENTS_PER_USD

    def to_cents(self) -> float:
        return int(self) / self._NANOCENTS_PER_CENT


class PricingProvider(ABC):
    """
    Abstract base class for pricing providers.

    Implement this class to create custom pricing providers that can be used
    with the AccountedModel wrapper.
    """

    @abstractmethod
    async def calculate_cost_from_response(
        self, model_id: str, usage: Usage, response: Optional[AsyncResponse]
    ) -> Nanocents:
        """
        Calculate the cost for a given model usage.

        Args:
            model_id: The model identifier
            usage: An llm.Usage object containing token usage information
            response: The llm.AsyncResponse object for accessing provider-specific metadata
        Returns:
            Cost as a Nanocents value
        """
        pass

    async def supported_models(self) -> Optional[set[str]]:
        """
        Return the set of model IDs this provider can price, or None to
        indicate all models are supported.

        Returns None by default. Override to restrict to known models.
        """
        return None


class ModelPricingNotFoundError(Exception):
    """Raised when pricing for a model cannot be found."""

    pass


class DefaultPricingProvider(PricingProvider):
    """
    Default pricing provider that fetches pricing data from the remote endpoint.
    """

    def __init__(self):
        self._pricing_cache: Optional[dict] = None

    def _load_pricing_data(self) -> dict:
        """
        Load pricing data from the remote endpoint (and cache it in memory).

        Returns a dict mapping model IDs to pricing information.
        """
        if self._pricing_cache is not None:
            return self._pricing_cache

        response = httpx.get(_PRICING_URL, timeout=_PRICING_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        prices = data.get("prices")

        if not isinstance(prices, list):
            raise RuntimeError("Unexpected pricing payload structure")

        # Convert list to dict for faster lookups
        self._pricing_cache = {item["id"]: item for item in prices}
        return self._pricing_cache

    async def supported_models(self) -> Optional[set[str]]:
        pricing_data = self._load_pricing_data()
        return set(pricing_data.keys())

    async def calculate_cost_from_response(
        self, model_id: str, usage: Usage, response: Optional[AsyncResponse]
    ) -> Nanocents:
        pricing_data = self._load_pricing_data()

        if model_id not in pricing_data:
            raise ModelPricingNotFoundError(
                f"Pricing not found for model '{model_id}'. "
                f"Available models: {', '.join(sorted(pricing_data.keys()))}"
            )

        pricing = pricing_data[model_id]

        input_tokens = usage.input or 0
        output_tokens = usage.output or 0

        # Pricing is in USD per million tokens
        # 1 USD per million tokens = 100,000,000,000 / 1,000,000 = 100,000 nanocents per token
        nanocents_per_token_multiplier = 100_000

        input_cost = int(
            input_tokens * pricing["input"] * nanocents_per_token_multiplier
        )
        output_cost = int(
            output_tokens * pricing["output"] * nanocents_per_token_multiplier
        )

        return Nanocents(input_cost + output_cost)


