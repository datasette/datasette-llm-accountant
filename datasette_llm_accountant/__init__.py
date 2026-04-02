from datasette import hookimpl
from datasette.plugins import pm

from . import hookspecs

pm.add_hookspecs(hookspecs)

from . import hooks  # noqa: E402

from .accountant import Accountant, Tx, InsufficientBalanceError  # noqa: E402
from .hooks import ReservationExceededError, GroupReservation  # noqa: E402
from .pricing import (  # noqa: E402
    Nanocents,
    PricingProvider,
    DefaultPricingProvider,
    ModelPricingNotFoundError,
)

__all__ = [
    # Accountant base class (for implementing custom accountants)
    "Accountant",
    "Tx",
    "InsufficientBalanceError",
    # Errors
    "ReservationExceededError",
    "GroupReservation",
    # Pricing providers
    "Nanocents",
    "PricingProvider",
    "DefaultPricingProvider",
    "ModelPricingNotFoundError",
]


@hookimpl
def llm_prompt_context(datasette, model_id, prompt, purpose, actor):
    return hooks.llm_prompt_context(datasette, model_id, prompt, purpose, actor)


@hookimpl
def llm_group_exit(datasette, group):
    return hooks.llm_group_exit(datasette, group)


@hookimpl
def llm_filter_models(datasette, models, actor, purpose):
    return hooks.llm_filter_models(datasette, models, actor, purpose)
