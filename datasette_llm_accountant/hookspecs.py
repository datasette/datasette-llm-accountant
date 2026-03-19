"""
Plugin hook specifications for datasette-llm-accountant.
"""

from pluggy import HookspecMarker

hookspec = HookspecMarker("datasette")


@hookspec
def register_llm_accountants(datasette):
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
def register_llm_accountant_pricing(datasette):
    """
    Register a custom pricing provider for LLM cost calculations.

    Args:
        datasette: The Datasette instance

    Returns:
        A PricingProvider subclass instance, or None to use the default provider.
        Only the first registered pricing provider will be used.

    Example:
        from datasette_llm_accountant import PricingProvider

        class CustomPricingProvider(PricingProvider):
            def get_model_pricing(self, model_id: str) -> dict:
                # Return custom pricing data
                ...

        @hookimpl
        def register_llm_accountant_pricing(datasette):
            return CustomPricingProvider()
    """
    pass
