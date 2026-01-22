from datasette import hookimpl
from datasette.plugins import pm

from . import hookspecs

pm.add_hookspecs(hookspecs)

from . import hooks

from .accountant import Accountant, Tx, InsufficientBalanceError
from .hooks import ReservationExceededError, GroupReservation
from .pricing import (
    calculate_cost_nanocents,
    get_model_pricing,
    usd_to_nanocents,
    nanocents_to_usd,
    ModelPricingNotFoundError,
)

__all__ = [
    # Accountant base class (for implementing custom accountants)
    "Accountant",
    "Tx",
    "InsufficientBalanceError",
    # Errors
    "ReservationExceededError",
    # Pricing utilities
    "calculate_cost_nanocents",
    "get_model_pricing",
    "usd_to_nanocents",
    "nanocents_to_usd",
    "ModelPricingNotFoundError",
]


@hookimpl
def llm_prompt_context(datasette, model_id, prompt, purpose):
    return hooks.llm_prompt_context(datasette, model_id, prompt, purpose)


@hookimpl
def llm_group_exit(datasette, group):
    return hooks.llm_group_exit(datasette, group)
