from datasette import hookimpl
from datasette.plugins import pm

# Register our plugin hooks
from . import hookspecs

pm.add_hookspecs(hookspecs)

# Export main classes for use by other plugins and applications
from .wrapper import LlmWrapper, AccountedModel, AccountedTransaction
from .accountant import Accountant, Tx, InsufficientBalanceError
from .pricing import (
    calculate_cost_nanocents,
    get_model_pricing,
    usd_to_nanocents,
    nanocents_to_usd,
    ModelPricingNotFoundError,
)

__all__ = [
    "LlmWrapper",
    "AccountedModel",
    "AccountedTransaction",
    "Accountant",
    "Tx",
    "InsufficientBalanceError",
    "calculate_cost_nanocents",
    "get_model_pricing",
    "usd_to_nanocents",
    "nanocents_to_usd",
    "ModelPricingNotFoundError",
]

from datasette import Response

def dbg():
  return Response.json(
        {"ok": True, "foo": "bar1"}
    )

@hookimpl
def register_routes():
  return [(r"^/-/dbg$", dbg)]