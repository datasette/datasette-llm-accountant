"""
LLM wrapper with accounting support.
"""

import llm
from typing import Optional, List
from .accountant import Accountant, Tx, InsufficientBalanceError
from .pricing import PricingProvider, DefaultPricingProvider


class ReservationExceededError(Exception):
    """Raised when actual LLM cost exceeds the reserved amount."""

    pass


class AccountedTransaction:
    """
    Context manager for a reserved LLM transaction.

    Handles multi-accountant reservations with rollback on failure.
    """

    def __init__(
        self,
        model: "AccountedModel",
        nanocents: int,
        accountants: List[Accountant],
    ):
        self.model = model
        self.nanocents = nanocents
        self.accountants = accountants
        self.transactions: List[tuple[Accountant, Tx]] = []
        self.spent_nanocents = 0
        self.pricing_provider = model.pricing_provider

    async def __aenter__(self):
        """Reserve from all accountants, rolling back on failure."""
        for accountant in self.accountants:
            try:
                tx = await accountant.reserve(self.nanocents)
                self.transactions.append((accountant, tx))
            except InsufficientBalanceError:
                # Rollback all previous reservations
                await self._rollback()
                raise
            except Exception as e:
                # Rollback on any error
                await self._rollback()
                raise Exception(f"Error reserving from accountant: {e}") from e

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Settle all accountants for the actual amount spent."""
        if exc_type is not None:
            # If there was an exception, rollback instead of settling
            await self._rollback()
            return False

        # Settle with all accountants for the actual amount spent
        for accountant, tx in self.transactions:
            await accountant.settle(tx, self.spent_nanocents)

        return False

    async def _rollback(self):
        """Rollback all successful reservations."""
        for accountant, tx in self.transactions:
            try:
                await accountant.rollback(tx)
            except Exception as e:
                # Log but don't raise - we want to rollback all
                print(f"Error rolling back transaction: {e}")

    async def prompt(self, prompt_text: str, **kwargs) -> llm.AsyncResponse:
        """
        Execute a prompt and track the cost against this reservation.

        Args:
            prompt_text: The prompt text
            **kwargs: Additional arguments to pass to model.prompt()

        Returns:
            The response text

        Raises:
            ReservationExceededError: If the cost exceeds the reservation
        """
        response = await self.model._async_model.prompt(prompt_text, **kwargs)

        # Calculate the cost
        usage = await response.usage()
        model_id = self.model._async_model.model_id

        # Use pricing provider to calculate cost
        cost_nanocents = self.pricing_provider.calculate_cost_nanocents(
            model_id,
            input_tokens=usage.input or 0,
            output_tokens=usage.output or 0,
        )

        # Check if we've exceeded the reservation
        new_total = self.spent_nanocents + cost_nanocents
        if new_total > self.nanocents:
            raise ReservationExceededError(
                f"Cost {new_total} nanocents exceeds reservation of {self.nanocents} nanocents"
            )

        self.spent_nanocents = new_total
        return response


class AccountedModel:
    """
    Wraps an llm.AsyncModel with accounting support.
    """

    def __init__(
        self,
        async_model: llm.models.AsyncModel,
        accountants: List[Accountant],
        pricing_provider: Optional[PricingProvider] = None,
    ):
        self._async_model = async_model
        self._accountants = accountants
        self.pricing_provider = pricing_provider or DefaultPricingProvider()

    def reserve(
        self, usd: Optional[float] = None, nanocents: Optional[int] = None
    ) -> AccountedTransaction:
        """
        Reserve an amount for LLM usage.

        Args:
            usd: Amount to reserve in USD (mutually exclusive with nanocents)
            nanocents: Amount to reserve in nanocents (mutually exclusive with usd)

        Returns:
            An AccountedTransaction context manager

        Raises:
            ValueError: If neither or both usd/nanocents are specified
            InsufficientBalanceError: If any accountant cannot reserve
        """
        if (usd is None) == (nanocents is None):
            raise ValueError("Must specify exactly one of usd or nanocents")

        if usd is not None:
            # Convert USD to nanocents: 1 USD = 100 cents = 100,000,000,000 nanocents
            nanocents = int(usd * 100_000_000_000)
        
        # At this point nanocents is guaranteed to be int
        assert nanocents is not None
        return AccountedTransaction(self, nanocents, self._accountants)

    async def prompt(self, prompt_text: str, usd: float = 0.5, **kwargs) -> llm.AsyncResponse:
        """
        Execute a prompt with automatic reservation.

        Args:
            prompt_text: The prompt text
            usd: Amount to reserve (default: 0.5 USD / 50 cents)
            **kwargs: Additional arguments to pass to model.prompt()

        Returns:
            The response text

        Raises:
            InsufficientBalanceError: If reservation fails
            ReservationExceededError: If cost exceeds reservation
        """
        async with self.reserve(usd=usd) as tx:
            return await tx.prompt(prompt_text, **kwargs)


class LlmWrapper:
    """
    Wrapper for llm that integrates with Datasette's accountant system.
    """

    def __init__(self, datasette):
        self.datasette = datasette
        self._accountants: Optional[List[Accountant]] = None
        self._pricing_provider: Optional[PricingProvider] = None

    def _get_accountants(self) -> List[Accountant]:
        """Get all registered accountants via the plugin hook."""
        if self._accountants is not None:
            return self._accountants

        # Import here to avoid circular imports
        from datasette.plugins import pm

        accountants = []
        for plugin_accountants in pm.hook.register_llm_accountants(
            datasette=self.datasette
        ):
            if plugin_accountants:
                if isinstance(plugin_accountants, list):
                    accountants.extend(plugin_accountants)
                else:
                    accountants.append(plugin_accountants)

        self._accountants = accountants
        return accountants

    def _get_pricing_provider(self) -> PricingProvider:
        """Get the pricing provider via the plugin hook, or default."""
        if self._pricing_provider is not None:
            return self._pricing_provider

        # Import here to avoid circular imports
        from datasette.plugins import pm

        # Use firstresult=True hook to get a single pricing provider
        pricing_provider = pm.hook.register_llm_accountant_pricing(
            datasette=self.datasette
        )

        # If no plugin provided a pricing provider, use default
        if pricing_provider is None:
            pricing_provider = DefaultPricingProvider()
        
        self._pricing_provider = pricing_provider
        assert self._pricing_provider is not None
        return self._pricing_provider

    def get_async_model(self, model_id: str) -> AccountedModel:
        """
        Get an async model wrapped with accounting.

        Args:
            model_id: The model identifier (e.g., "gpt-4o-mini")

        Returns:
            An AccountedModel instance
        """
        async_model = llm.get_async_model(model_id)
        accountants = self._get_accountants()
        pricing_provider = self._get_pricing_provider()
        return AccountedModel(async_model, accountants, pricing_provider)

    def get_async_models(self):
        return llm.get_async_models()
