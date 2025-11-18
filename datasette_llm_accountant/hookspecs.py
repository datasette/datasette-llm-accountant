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
