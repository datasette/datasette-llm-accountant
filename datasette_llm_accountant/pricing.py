"""
LLM pricing lookup and cost calculation.
"""

from typing import Optional
from abc import ABC, abstractmethod

import httpx


# Cache the pricing data globally
_pricing_cache: Optional[dict] = None
_PRICING_URL = "https://simonw.github.io/llm-prices/current-v1.json"
_PRICING_TIMEOUT = 10.0


class PricingProvider(ABC):
    """
    Abstract base class for pricing providers.
    
    Implement this class to create custom pricing providers that can be used
    with the AccountedModel wrapper.
    """
    
    @abstractmethod
    def get_model_pricing(self, model_id: str) -> dict:
        """
        Get pricing information for a specific model.
        
        Args:
            model_id: The model identifier (e.g., "gpt-4o", "claude-3.5-sonnet")
        
        Returns:
            A dict with keys: id, vendor, name, input, output, input_cached
            - id: The model identifier
            - vendor: The model vendor
            - name: The model name
            - input: Price per million input tokens in USD
            - output: Price per million output tokens in USD
            - input_cached: (Optional) Price per million cached input tokens in USD
        
        Raises:
            ModelPricingNotFoundError: If the model is not found in pricing data
        """
        pass
    
    def calculate_cost_nanocents(
        self, model_id: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0
    ) -> int:
        """
        Calculate the cost in nanocents for a given token usage.
        
        Args:
            model_id: The model identifier
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
            cached_input_tokens: Number of cached input tokens (if applicable)
        
        Returns:
            Cost in nanocents (1/1,000,000,000 of a cent)
        
        Raises:
            ModelPricingNotFoundError: If the model pricing is not available
        """
        pricing = self.get_model_pricing(model_id)
        
        # Pricing is in USD per million tokens
        # We need to convert to nanocents (1 USD = 100 cents = 100,000,000,000 nanocents)
        # So 1 USD per million tokens = 100,000,000,000 / 1,000,000 = 100,000 nanocents per token
        
        nanocents_per_token_multiplier = 100_000
        
        # Calculate uncached input cost
        uncached_input_tokens = input_tokens - cached_input_tokens
        input_cost = int(
            uncached_input_tokens * pricing["input"] * nanocents_per_token_multiplier
        )
        
        # Calculate output cost
        output_cost = int(
            output_tokens * pricing["output"] * nanocents_per_token_multiplier
        )
        
        # Calculate cached input cost if applicable
        cached_cost = 0
        if cached_input_tokens > 0 and pricing.get("input_cached") is not None:
            cached_cost = int(
                cached_input_tokens
                * pricing["input_cached"]
                * nanocents_per_token_multiplier
            )
        
        total_cost = input_cost + output_cost + cached_cost
        return total_cost


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
    
    def get_model_pricing(self, model_id: str) -> dict:
        """
        Get pricing information for a specific model.
        
        Args:
            model_id: The model identifier (e.g., "gpt-4o", "claude-3.5-sonnet")
        
        Returns:
            A dict with keys: id, vendor, name, input, output, input_cached
        
        Raises:
            ModelPricingNotFoundError: If the model is not found in pricing data
        """
        pricing_data = self._load_pricing_data()
        
        if model_id not in pricing_data:
            raise ModelPricingNotFoundError(
                f"Pricing not found for model '{model_id}'. "
                f"Available models: {', '.join(sorted(pricing_data.keys()))}"
            )
        
        return pricing_data[model_id]


# Global default provider instance
_default_provider = DefaultPricingProvider()


def load_pricing_data() -> dict:
    """
    Load pricing data from the remote endpoint (and cache it in memory).

    Returns a dict mapping model IDs to pricing information.
    
    This function uses the global default pricing provider.
    """
    return _default_provider._load_pricing_data()


def get_model_pricing(model_id: str) -> dict:
    """
    Get pricing information for a specific model.

    Args:
        model_id: The model identifier (e.g., "gpt-4o", "claude-3.5-sonnet")

    Returns:
        A dict with keys: id, vendor, name, input, output, input_cached

    Raises:
        ModelPricingNotFoundError: If the model is not found in pricing data
    
    This function uses the global default pricing provider.
    """
    return _default_provider.get_model_pricing(model_id)


def calculate_cost_nanocents(
    model_id: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0
) -> int:
    """
    Calculate the cost in nanocents for a given token usage.

    Args:
        model_id: The model identifier
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        cached_input_tokens: Number of cached input tokens (if applicable)

    Returns:
        Cost in nanocents (1/1,000,000,000 of a cent)

    Raises:
        ModelPricingNotFoundError: If the model pricing is not available
    
    This function uses the global default pricing provider.
    """
    return _default_provider.calculate_cost_nanocents(
        model_id, input_tokens, output_tokens, cached_input_tokens
    )


def usd_to_nanocents(usd: float) -> int:
    """
    Convert USD to nanocents.

    Args:
        usd: Amount in US dollars

    Returns:
        Amount in nanocents (1/1,000,000,000 of a cent)
    """
    # 1 USD = 100 cents = 100,000,000,000 nanocents
    return int(usd * 100_000_000_000)


def nanocents_to_usd(nanocents: int) -> float:
    """
    Convert nanocents to USD.

    Args:
        nanocents: Amount in nanocents

    Returns:
        Amount in US dollars
    """
    return nanocents / 100_000_000_000
