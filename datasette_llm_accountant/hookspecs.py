"""
Plugin hook specifications for datasette-llm-accountant.
"""

from __future__ import annotations

from typing import Optional, Union

from pluggy import HookspecMarker

from datasette_llm_accountant.accountant import Accountant
from datasette_llm_accountant.pricing import PricingProvider
hookspec = HookspecMarker("datasette")


@hookspec
def register_llm_accountants(datasette) -> Optional[Union[list[Accountant], Accountant]]:
    """
    Register accountants for tracking LLM token usage costs.

    Args:
        datasette: The Datasette instance

    Returns:
        A list of Accountant subclass instances, a single Accountant instance,
        or None if this plugin doesn't provide any accountants.

    Example:
        @hookimpl
        def register_llm_accountants(datasette):
            return [MyAccountant(), AnotherAccountant()]
    """
    pass


@hookspec(firstresult=True)
def register_llm_accountant_pricing(datasette) -> Optional[PricingProvider]:
    """
    Register a custom pricing provider for LLM cost calculations.

    Args:
        datasette: The Datasette instance

    Returns:
        A PricingProvider subclass instance, or None to use the default provider.
        Only the first registered pricing provider will be used.

    Example:
        from datasette_llm_accountant import PricingProvider, Nanocents

        class CustomPricingProvider(PricingProvider):
            async def calculate_cost_from_response(self, model_id, usage, response):
                # Return cost as Nanocents
                return Nanocents(...)

            async def supported_models(self):
                # Return set of model IDs, or None for all
                return MY_KNOWN_MODELS

        @hookimpl
        def register_llm_accountant_pricing(datasette):
            return CustomPricingProvider()
    """
    pass
