"""
Tests for the LLM wrapper with accounting.
"""
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from datasette.app import Datasette
from datasette import hookimpl

from datasette_llm_accountant import (
    LlmWrapper,
    Accountant,
    Tx,
    InsufficientBalanceError,
    PricingProvider,
    ModelPricingNotFoundError,
)
from datasette_llm_accountant.wrapper import ReservationExceededError


class HardcodedPricingProvider(PricingProvider):
    """Test pricing provider with hardcoded pricing data."""

    def __init__(self):
        # Hardcoded pricing for testing
        self._pricing = {
            "gpt-4o-mini": {
                "id": "gpt-4o-mini",
                "vendor": "openai",
                "name": "GPT-4o mini",
                "input": 0.15,  # USD per million tokens
                "output": 0.6,   # USD per million tokens
                "input_cached": None,
            },
            "test-model": {
                "id": "test-model",
                "vendor": "test",
                "name": "Test Model",
                "input": 1.0,   # USD per million tokens
                "output": 2.0,  # USD per million tokens
                "input_cached": 0.5,  # USD per million tokens
            },
        }

    def get_model_pricing(self, model_id: str) -> dict:
        """Get pricing information for a specific model."""
        if model_id not in self._pricing:
            raise ModelPricingNotFoundError(
                f"Pricing not found for model '{model_id}'. "
                f"Available models: {', '.join(sorted(self._pricing.keys()))}"
            )
        return self._pricing[model_id]


class AccountantTest(Accountant):
    """Test accountant that tracks reserve/settle calls."""

    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.reservations = []
        self.settlements = []
        self.rollbacks = []

    async def reserve(self, nanocents: int) -> Tx:
        if self.should_fail:
            raise InsufficientBalanceError("Insufficient balance")
        tx = Tx(f"tx-{len(self.reservations)}")
        self.reservations.append((tx, nanocents))
        return tx

    async def settle(self, tx: Tx, nanocents: int):
        self.settlements.append((tx, nanocents))

    async def rollback(self, tx: Tx):
        self.rollbacks.append(tx)


def create_mock_response(input_tokens, output_tokens, model_id="gpt-4o-mini"):
    """Create a mock llm response with usage data."""
    response = AsyncMock()

    # Mock text() method
    response.text = AsyncMock(return_value="Mock response text")

    # Mock usage() method
    usage = Mock()
    usage.input = input_tokens
    usage.output = output_tokens
    response.usage = AsyncMock(return_value=usage)

    return response


def create_mock_model(model_id="gpt-4o-mini", responses=None):
    """Create a mock llm.AsyncModel."""
    model = Mock()
    model.model_id = model_id

    # Make prompt an AsyncMock
    if responses:
        # If multiple responses provided, return them in sequence (as async)
        async def prompt_side_effect(*args, **kwargs):
            return responses.pop(0)
        model.prompt = AsyncMock(side_effect=prompt_side_effect)
    else:
        # Default single response (as async)
        model.prompt = AsyncMock(return_value=create_mock_response(100, 50, model_id))

    return model


@pytest.mark.asyncio
async def test_accounted_model_reserve_and_prompt():
    """Test basic reserve and prompt flow."""
    from datasette_llm_accountant import AccountedModel

    accountant = AccountantTest()
    mock_model = create_mock_model()

    accounted = AccountedModel(mock_model, [accountant])

    # Reserve and prompt
    async with accounted.reserve(usd=1.0) as tx:
        result = await tx.prompt("Test prompt")

    # Check that reservation was made
    assert len(accountant.reservations) == 1
    assert accountant.reservations[0][1] == 100_000_000_000  # 1 USD in nanocents

    # Check that settlement was made
    assert len(accountant.settlements) == 1
    # Settlement should be for actual cost (100 input + 50 output tokens for gpt-4o-mini)
    # gpt-4o-mini: input=0.15, output=0.6 per million
    # Cost: (100 * 0.15 + 50 * 0.6) * 100,000 = 4,500,000 nanocents
    assert accountant.settlements[0][1] == 4_500_000

    # Check response
    assert await result.text() == "Mock response text"


@pytest.mark.asyncio
async def test_accounted_model_direct_prompt():
    """Test direct prompt with automatic reservation."""
    from datasette_llm_accountant import AccountedModel

    accountant = AccountantTest()
    mock_model = create_mock_model()

    accounted = AccountedModel(mock_model, [accountant])

    # Direct prompt with default reservation (0.5 USD)
    result = await accounted.prompt("Test prompt")

    # Check that reservation was made for default 0.5 USD
    assert len(accountant.reservations) == 1
    assert accountant.reservations[0][1] == 50_000_000_000  # 0.5 USD in nanocents

    # Check settlement
    assert len(accountant.settlements) == 1
    assert await result.text() == "Mock response text"


@pytest.mark.asyncio
async def test_accounted_model_multiple_prompts_in_reservation():
    """Test multiple prompts within a single reservation."""
    from datasette_llm_accountant import AccountedModel

    accountant = AccountantTest()

    # Create multiple mock responses
    responses = [
        create_mock_response(100, 50),  # First prompt
        create_mock_response(200, 100),  # Second prompt
    ]
    mock_model = create_mock_model(responses=responses)

    accounted = AccountedModel(mock_model, [accountant])

    # Reserve and make multiple prompts
    async with accounted.reserve(usd=1.0) as tx:
        result1 = await tx.prompt("First prompt")
        result2 = await tx.prompt("Second prompt")

    # Only one reservation should be made
    assert len(accountant.reservations) == 1

    # Settlement should be for cumulative cost
    # First: (100 * 0.15 + 50 * 0.6) * 100,000 = 4,500,000
    # Second: (200 * 0.15 + 100 * 0.6) * 100,000 = 9,000,000
    # Total: 13,500,000
    assert len(accountant.settlements) == 1
    assert accountant.settlements[0][1] == 13_500_000


@pytest.mark.asyncio
async def test_rollback_on_insufficient_balance():
    """Test that reservations are rolled back when an accountant fails."""
    from datasette_llm_accountant import AccountedModel

    accountant1 = AccountantTest()
    accountant2 = AccountantTest(should_fail=True)  # This one will fail
    accountant3 = AccountantTest()  # Should never be called

    mock_model = create_mock_model()
    accounted = AccountedModel(mock_model, [accountant1, accountant2, accountant3])

    # Try to reserve - should fail
    with pytest.raises(InsufficientBalanceError):
        async with accounted.reserve(usd=1.0) as tx:
            await tx.prompt("Test prompt")

    # First accountant should have reserved and then rolled back
    assert len(accountant1.reservations) == 1
    assert len(accountant1.rollbacks) == 1

    # Second accountant should have attempted reservation (and failed)
    assert len(accountant2.reservations) == 0  # Failed before recording

    # Third accountant should not have been called at all
    assert len(accountant3.reservations) == 0


@pytest.mark.asyncio
async def test_rollback_on_exception():
    """Test that reservations are rolled back when an exception occurs."""
    from datasette_llm_accountant import AccountedModel

    accountant = AccountantTest()
    mock_model = create_mock_model()

    accounted = AccountedModel(mock_model, [accountant])

    # Reserve and raise an exception
    with pytest.raises(ValueError):
        async with accounted.reserve(usd=1.0) as tx:
            raise ValueError("Something went wrong")

    # Should have rolled back instead of settling
    assert len(accountant.reservations) == 1
    assert len(accountant.rollbacks) == 1
    assert len(accountant.settlements) == 0


@pytest.mark.asyncio
async def test_reservation_exceeded():
    """Test that exceeding reservation raises an error."""
    from datasette_llm_accountant import AccountedModel

    accountant = AccountantTest()

    # Create a response with high token usage
    # gpt-4o-mini costs: input=0.15, output=0.6 per million tokens
    # For 1M input + 1M output: (1M * 0.15 + 1M * 0.6) * 100,000 = 75,000,000,000 nanocents (~$0.75)
    expensive_response = create_mock_response(1_000_000, 1_000_000)
    mock_model = create_mock_model(responses=[expensive_response])

    accounted = AccountedModel(mock_model, [accountant])

    # Try to use with a small reservation (0.1 USD = 10,000,000,000 nanocents)
    with pytest.raises(ReservationExceededError) as exc_info:
        async with accounted.reserve(usd=0.1) as tx:
            await tx.prompt("Expensive prompt")

    assert "exceeds reservation" in str(exc_info.value)

    # Should have rolled back
    assert len(accountant.rollbacks) == 1
    assert len(accountant.settlements) == 0


@pytest.mark.asyncio
async def test_llm_wrapper_integration():
    """Test LlmWrapper integration with Datasette plugin system."""
    # Create a test plugin that provides an accountant
    class TestPlugin:
        __name__ = "TestPlugin"

        @hookimpl
        def register_llm_accountants(self, datasette):
            return [AccountantTest()]

    datasette = Datasette(memory=True)

    # Register the test plugin
    try:
        datasette.pm.register(TestPlugin(), name="test-accountant-plugin")

        # Create wrapper
        wrapper = LlmWrapper(datasette)

        # Get accountants - should include our test accountant
        accountants = wrapper._get_accountants()
        assert len(accountants) == 1
        assert isinstance(accountants[0], AccountantTest)

    finally:
        datasette.pm.unregister(name="test-accountant-plugin")


@pytest.mark.asyncio
async def test_llm_wrapper_multiple_accountants():
    """Test LlmWrapper with multiple accountants from different plugins."""

    class TestPlugin1:
        __name__ = "TestPlugin1"

        @hookimpl
        def register_llm_accountants(self, datasette):
            return AccountantTest()

    class TestPlugin2:
        __name__ = "TestPlugin2"

        @hookimpl
        def register_llm_accountants(self, datasette):
            return [AccountantTest(), AccountantTest()]

    datasette = Datasette(memory=True)

    try:
        datasette.pm.register(TestPlugin1(), name="test-plugin-1")
        datasette.pm.register(TestPlugin2(), name="test-plugin-2")

        wrapper = LlmWrapper(datasette)
        accountants = wrapper._get_accountants()

        # Should have 3 accountants total (1 from plugin1, 2 from plugin2)
        assert len(accountants) == 3

    finally:
        datasette.pm.unregister(name="test-plugin-1")
        datasette.pm.unregister(name="test-plugin-2")


@pytest.mark.asyncio
async def test_reserve_with_nanocents():
    """Test reserving with nanocents instead of USD."""
    from datasette_llm_accountant import AccountedModel

    accountant = AccountantTest()
    mock_model = create_mock_model()

    accounted = AccountedModel(mock_model, [accountant])

    # Reserve with nanocents
    async with accounted.reserve(nanocents=50_000_000_000) as tx:
        result = await tx.prompt("Test prompt")

    # Check that reservation was made for exact nanocents amount
    assert accountant.reservations[0][1] == 50_000_000_000


@pytest.mark.asyncio
async def test_reserve_validation():
    """Test that reserve validates parameters correctly."""
    from datasette_llm_accountant import AccountedModel

    accountant = AccountantTest()
    mock_model = create_mock_model()
    accounted = AccountedModel(mock_model, [accountant])

    # Should raise error if neither usd nor nanocents specified
    with pytest.raises(ValueError) as exc_info:
        accounted.reserve()
    assert "exactly one" in str(exc_info.value)

    # Should raise error if both specified
    with pytest.raises(ValueError) as exc_info:
        accounted.reserve(usd=1.0, nanocents=100_000_000_000)
    assert "exactly one" in str(exc_info.value)


@pytest.mark.asyncio
async def test_custom_pricing_provider():
    """Test using a custom pricing provider with AccountedModel."""
    from datasette_llm_accountant import AccountedModel

    accountant = AccountantTest()
    pricing_provider = HardcodedPricingProvider()
    
    # Create a mock model with test-model id
    mock_model = create_mock_model(model_id="test-model")
    
    # Create AccountedModel with custom pricing provider
    accounted = AccountedModel(mock_model, [accountant], pricing_provider)
    
    # Reserve and prompt
    async with accounted.reserve(usd=1.0) as tx:
        result = await tx.prompt("Test prompt")
    
    # Check settlement cost
    # test-model pricing: input=1.0, output=2.0 per million tokens
    # 100 input + 50 output tokens
    # Cost: (100 * 1.0 + 50 * 2.0) * 100,000 = 20,000,000 nanocents
    assert len(accountant.settlements) == 1
    assert accountant.settlements[0][1] == 20_000_000


@pytest.mark.asyncio
async def test_custom_pricing_provider_with_cached_tokens():
    """Test custom pricing provider with cached input tokens."""
    from datasette_llm_accountant import AccountedModel

    accountant = AccountantTest()
    pricing_provider = HardcodedPricingProvider()
    
    # Create a mock response with cached tokens (requires updating usage)
    response = AsyncMock()
    response.text = AsyncMock(return_value="Mock response text")
    usage = Mock()
    usage.input = 100
    usage.output = 50
    # Note: LLM library doesn't expose cached tokens in usage yet,
    # but the pricing provider supports it
    response.usage = AsyncMock(return_value=usage)
    
    mock_model = create_mock_model(model_id="test-model", responses=[response])
    accounted = AccountedModel(mock_model, [accountant], pricing_provider)
    
    async with accounted.reserve(usd=1.0) as tx:
        result = await tx.prompt("Test prompt")
    
    # Should use regular pricing (no cached tokens in usage)
    assert accountant.settlements[0][1] == 20_000_000


@pytest.mark.asyncio
async def test_pricing_provider_model_not_found():
    """Test that custom pricing provider raises error for unknown models."""
    from datasette_llm_accountant import AccountedModel

    accountant = AccountantTest()
    pricing_provider = HardcodedPricingProvider()
    
    # Create model with unknown id
    mock_model = create_mock_model(model_id="unknown-model")
    accounted = AccountedModel(mock_model, [accountant], pricing_provider)
    
    # Should raise ModelPricingNotFoundError when trying to calculate cost
    with pytest.raises(ModelPricingNotFoundError) as exc_info:
        async with accounted.reserve(usd=1.0) as tx:
            await tx.prompt("Test prompt")
    
    assert "unknown-model" in str(exc_info.value)
    
    # Should have rolled back the reservation
    assert len(accountant.rollbacks) == 1


@pytest.mark.asyncio
async def test_llm_wrapper_with_pricing_provider():
    """Test LlmWrapper with a custom pricing provider."""
    from datasette_llm_accountant import AccountedModel

    accountant = AccountantTest()
    pricing_provider = HardcodedPricingProvider()
    
    # Create an AccountedModel directly with the pricing provider
    mock_model = create_mock_model(model_id="test-model")
    
    accounted = AccountedModel(mock_model, [accountant], pricing_provider=pricing_provider)
    
    # Verify the pricing provider is set
    assert accounted.pricing_provider is pricing_provider
    
    # Test that it uses the custom pricing provider
    async with accounted.reserve(usd=1.0) as tx:
        result = await tx.prompt("Test prompt")
    
    # With test-model pricing (input=1.0, output=2.0 per million tokens)
    # For 100 input + 50 output tokens:
    # Cost: (100 * 1.0 + 50 * 2.0) * 100,000 = 20,000,000 nanocents
    assert len(accountant.settlements) == 1
    assert accountant.settlements[0][1] == 20_000_000


@pytest.mark.asyncio
async def test_accounted_model_without_pricing_provider():
    """Test that AccountedModel works without a pricing provider (uses default)."""
    from datasette_llm_accountant import AccountedModel

    accountant = AccountantTest()
    mock_model = create_mock_model()
    
    # Create AccountedModel without pricing provider
    accounted = AccountedModel(mock_model, [accountant], pricing_provider=None)
    
    # Should still work using the default pricing
    async with accounted.reserve(usd=1.0) as tx:
        result = await tx.prompt("Test prompt")
    
    # Should have settled with default pricing
    # gpt-4o-mini: input=0.15, output=0.6 per million
    # (100 * 0.15 + 50 * 0.6) * 100,000 = 4,500,000
    assert len(accountant.settlements) == 1
    assert accountant.settlements[0][1] == 4_500_000

