"""
LLM pricing lookup and cost calculation.
"""

from typing import Optional

import httpx


# Cache the pricing data globally
_pricing_cache: Optional[dict] = None
_PRICING_URL = "https://www.llm-prices.com/current-v1.json"
_PRICING_TIMEOUT = 10.0


class ModelPricingNotFoundError(Exception):
    """Raised when pricing for a model cannot be found."""

    pass


def load_pricing_data() -> dict:
    """
    Load pricing data from the remote endpoint (and cache it in memory).

    Returns a dict mapping model IDs to pricing information.
    """
    global _pricing_cache

    if _pricing_cache is not None:
        return _pricing_cache

    response = httpx.get(_PRICING_URL, timeout=_PRICING_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    prices = data.get("prices")

    if not isinstance(prices, list):
        raise RuntimeError("Unexpected pricing payload structure")

    # Convert list to dict for faster lookups
    _pricing_cache = {item["id"]: item for item in prices}
    return _pricing_cache


def get_model_pricing(model_id: str) -> dict:
    """
    Get pricing information for a specific model.

    Args:
        model_id: The model identifier (e.g., "gpt-4o", "claude-3.5-sonnet")

    Returns:
        A dict with keys: id, vendor, name, input, output, input_cached

    Raises:
        ModelPricingNotFoundError: If the model is not found in pricing data
    """
    pricing_data = load_pricing_data()

    if model_id not in pricing_data:
        raise ModelPricingNotFoundError(
            f"Pricing not found for model '{model_id}'. "
            f"Available models: {', '.join(sorted(pricing_data.keys()))}"
        )

    return pricing_data[model_id]


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
    """
    pricing = get_model_pricing(model_id)

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
