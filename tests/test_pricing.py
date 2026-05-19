"""
Tests for pricing lookup and cost calculation.
"""

import pytest
from llm.models import Usage
from datasette_llm_accountant import (
    Nanocents,
    PricingProvider,
    DefaultPricingProvider,
    ModelPricingNotFoundError,
)


@pytest.mark.asyncio
async def test_calculate_cost_from_response():
    """Test cost calculation via the default provider."""
    provider = DefaultPricingProvider()
    # gpt-4o-mini: input 0.15, output 0.6 USD per million tokens
    usage = Usage(input=1000, output=500)
    cost = await provider.calculate_cost_from_response("gpt-4o-mini", usage, None)
    # Input: 1000 * 0.15 * 100_000 = 15,000,000
    # Output: 500 * 0.6 * 100_000 = 30,000,000
    assert cost == 45_000_000


@pytest.mark.asyncio
async def test_calculate_cost_not_found():
    """Test that ModelPricingNotFoundError is raised for unknown models."""
    provider = DefaultPricingProvider()
    usage = Usage(input=100, output=50)
    with pytest.raises(ModelPricingNotFoundError) as exc_info:
        await provider.calculate_cost_from_response(
            "nonexistent-model-xyz", usage, None
        )
    assert "nonexistent-model-xyz" in str(exc_info.value)


@pytest.mark.asyncio
async def test_pricing_provider_abc_contract():
    """Test that PricingProvider enforces the abstract contract."""
    with pytest.raises(TypeError):
        PricingProvider()

    class MyProvider(PricingProvider):
        async def calculate_cost_from_response(self, model_id, usage, response):
            return 42

    provider = MyProvider()
    assert await provider.supported_models() is None


@pytest.mark.asyncio
async def test_custom_provider_calculate_cost():
    """Test that a custom provider's calculate_cost_from_response works."""

    class FixedProvider(PricingProvider):
        async def calculate_cost_from_response(self, model_id, usage, response):
            input_tokens = usage.input or 0
            output_tokens = usage.output or 0
            # $10 per million input, $20 per million output
            return int(input_tokens * 10.0 * 100_000 + output_tokens * 20.0 * 100_000)

    provider = FixedProvider()

    cost = await provider.calculate_cost_from_response(
        "any-model", Usage(input=1000, output=500), None
    )
    # Input: 1000 * 10.0 * 100_000 = 1,000,000,000
    # Output: 500 * 20.0 * 100_000 = 1,000,000,000
    assert cost == 2_000_000_000


@pytest.mark.asyncio
async def test_supported_models():
    """Test supported_models on DefaultPricingProvider."""
    provider = DefaultPricingProvider()
    supported = await provider.supported_models()
    assert "gpt-4o-mini" in supported
    assert "nonexistent-model-xyz" not in supported


def test_nanocents_class():
    """Test the Nanocents wrapper type."""
    n = Nanocents(100_000_000_000)
    assert n == 100_000_000_000
    assert isinstance(n, int)
    assert isinstance(n, Nanocents)

    # to_usd / to_cents
    assert n.to_usd() == 1.0
    assert n.to_cents() == 100.0
    assert Nanocents(50_000_000_000).to_usd() == 0.5
    assert Nanocents(1_000_000_000).to_usd() == 0.01

    # from_usd / from_cents
    assert Nanocents.from_usd(1.0) == 100_000_000_000
    assert Nanocents.from_usd(0.5) == 50_000_000_000
    assert Nanocents.from_usd(0.01) == 1_000_000_000
    assert Nanocents.from_usd(10.0) == 1_000_000_000_000
    assert Nanocents.from_usd(1.50) == 150_000_000_000
    assert Nanocents.from_cents(50) == 50_000_000_000
    assert isinstance(Nanocents.from_usd(1.0), Nanocents)

    # Arithmetic works like int
    assert Nanocents(10) + Nanocents(20) == 30
    assert Nanocents(10) + 5 == 15


@pytest.mark.asyncio
async def test_calculate_cost_returns_nanocents():
    """Test that calculate_cost_from_response returns a Nanocents instance."""
    provider = DefaultPricingProvider()
    usage = Usage(input=1000, output=500)
    cost = await provider.calculate_cost_from_response("gpt-4o-mini", usage, None)
    assert isinstance(cost, Nanocents)


def test_default_pricing_provider_is_pricing_provider():
    """Test that DefaultPricingProvider is a proper PricingProvider subclass."""
    provider = DefaultPricingProvider()
    assert isinstance(provider, PricingProvider)
