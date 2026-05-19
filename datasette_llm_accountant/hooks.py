"""
Hook implementations for datasette-llm integration.

Implements the hooks from datasette-llm to provide:
- Auto-reservation for ungrouped prompts
- Group reservation and settlement
- Usage tracking and cost calculation
"""

from contextlib import asynccontextmanager
from typing import Optional, List, Dict
from datasette.plugins import pm

from .accountant import Accountant, Tx, InsufficientBalanceError
from .pricing import (
    Nanocents,
    DefaultPricingProvider,
    PricingProvider,
)


class ReservationExceededError(Exception):
    """Raised when actual LLM cost exceeds the reserved amount."""

    pass


class GroupReservation:
    """
    Tracks a reservation for a group of prompts.

    Manages the reserve/settle lifecycle across multiple accountants.
    """

    def __init__(
        self,
        nanocents: Nanocents,
        accountants: List[Accountant],
        model_id: Optional[str] = None,
        purpose: Optional[str] = None,
        actor_id: Optional[str] = None,
    ):
        self.nanocents = nanocents
        self.accountants = accountants
        self.model_id = model_id
        self.purpose = purpose
        self.actor_id = actor_id
        self.transactions: List[tuple[Accountant, Tx]] = []
        self.spent_nanocents = Nanocents(0)
        self._settled = False

    async def reserve_all(self):
        """Reserve from all accountants, rolling back on failure."""
        for accountant in self.accountants:
            try:
                tx = await accountant.reserve(
                    self.nanocents,
                    model_id=self.model_id,
                    purpose=self.purpose,
                    actor_id=self.actor_id,
                )
                self.transactions.append((accountant, tx))
            except InsufficientBalanceError:
                await self._rollback()
                raise
            except Exception as e:
                await self._rollback()
                raise Exception(f"Error reserving from accountant: {e}") from e

    async def settle_all(self):
        """Settle all accountants for the actual amount spent."""
        if self._settled:
            return
        self._settled = True

        for accountant, tx in self.transactions:
            await accountant.settle(
                tx,
                self.spent_nanocents,
                model_id=self.model_id,
                purpose=self.purpose,
                actor_id=self.actor_id,
            )

    async def _rollback(self):
        """Rollback all successful reservations."""
        for accountant, tx in self.transactions:
            try:
                await accountant.rollback(tx)
            except Exception:
                pass  # Log but continue rolling back others

    def add_usage(self, nanocents: Nanocents):
        """Add usage to this reservation."""
        self.spent_nanocents += nanocents

    def exceeded(self) -> bool:
        """Check if the reservation has been exceeded."""
        return self.spent_nanocents > self.nanocents


# Track active group reservations by group object id
_active_reservations: Dict[int, GroupReservation] = {}


def _get_accountants(datasette) -> List[Accountant]:
    """Get all registered accountants via the plugin hook."""
    accountants = []
    for plugin_accountants in pm.hook.register_llm_accountants(datasette=datasette):
        if plugin_accountants:
            if isinstance(plugin_accountants, list):
                accountants.extend(plugin_accountants)
            else:
                accountants.append(plugin_accountants)
    return accountants


def _get_pricing_provider(datasette) -> PricingProvider:
    """Get the pricing provider via the plugin hook, or default."""
    result = pm.hook.register_llm_accountant_pricing(datasette=datasette)
    if result is not None:
        return result
    return DefaultPricingProvider()


def _get_config(datasette) -> dict:
    """Get plugin configuration."""
    return datasette.plugin_config("datasette-llm-accountant") or {}


def _calculate_reservation_nanocents(datasette, model_id, purpose) -> Nanocents:
    """Calculate reservation amount from configuration."""
    config = _get_config(datasette)

    # Check purpose-specific config first
    purposes = config.get("purposes", {})
    if purpose and purpose in purposes:
        purpose_config = purposes[purpose]
        if "reservation_nanocents" in purpose_config:
            return Nanocents(purpose_config["reservation_nanocents"])
        elif "reservation_usd" in purpose_config:
            return Nanocents.from_usd(purpose_config["reservation_usd"])

    # Fall back to model-specific config
    models = config.get("models", {})
    if model_id in models:
        model_config = models[model_id]
        if "reservation_nanocents" in model_config:
            return Nanocents(model_config["reservation_nanocents"])
        elif "reservation_usd" in model_config:
            return Nanocents.from_usd(model_config["reservation_usd"])

    # Fall back to global default
    if "default_reservation_nanocents" in config:
        return Nanocents(config["default_reservation_nanocents"])
    elif "default_reservation_usd" in config:
        return Nanocents.from_usd(config["default_reservation_usd"])

    # Default: $0.50
    return Nanocents.from_usd(0.50)


async def _response_cost(provider, model_id, response) -> Nanocents:
    usage = await response.usage()
    return await provider.calculate_cost_from_response(model_id, usage, response)


def llm_prompt_context(datasette, model_id, prompt, purpose, actor):
    """
    Wrap prompt execution with accounting.

    For grouped prompts: creates reservation on first prompt, tracks usage.
    For ungrouped prompts: auto-reserves, executes, settles.
    """
    accountants = _get_accountants(datasette)

    if not accountants:
        # No accountants registered, just pass through
        return None

    provider = _get_pricing_provider(datasette)

    @asynccontextmanager
    async def accounting_wrapper(result):
        group = result.group
        actor_id = actor.get("id") if actor else None

        if group is not None:
            # Part of a group - use/create group's reservation
            group_id = id(group)

            # Create reservation on first prompt in group
            if group_id not in _active_reservations:
                nanocents = _calculate_reservation_nanocents(
                    datasette, model_id, purpose
                )
                reservation = GroupReservation(
                    nanocents,
                    accountants,
                    model_id=model_id,
                    purpose=purpose,
                    actor_id=actor_id,
                )
                await reservation.reserve_all()
                _active_reservations[group_id] = reservation

            reservation = _active_reservations.get(group_id)

            yield

            # Track usage via on_done callback. result.on_response_done()
            # handles both direct prompts and every response in a chain.
            if reservation:

                async def track_group_usage(response):
                    cost = await _response_cost(provider, model_id, response)
                    reservation.add_usage(cost)

                    if reservation.exceeded():
                        raise ReservationExceededError(
                            f"Cost {reservation.spent_nanocents} nanocents "
                            f"exceeds reservation of {reservation.nanocents} nanocents"
                        )

                await result.on_response_done(track_group_usage)

        else:
            # Not part of a group - auto-reserve for this single prompt
            config = _get_config(datasette)

            # Get auto-reservation amount (smaller default for single prompts)
            raw = config.get("auto_reservation_nanocents")
            if raw is not None:
                auto_nanocents = Nanocents(raw)
            else:
                auto_usd = config.get("auto_reservation_usd", 0.10)  # Default: $0.10
                auto_nanocents = Nanocents.from_usd(auto_usd)

            # Create a single-prompt reservation
            reservation = GroupReservation(
                auto_nanocents,
                accountants,
                model_id=model_id,
                purpose=purpose,
                actor_id=actor_id,
            )
            await reservation.reserve_all()

            try:
                yield

                # For model.chain(), datasette-llm keeps this context open
                # until the chain iterator finishes. At that point all chain
                # responses are available, so settle once for their aggregate
                # usage. Direct prompts still settle via on_done so returning a
                # streaming response is not blocked by accounting.
                if result.is_chain:
                    try:
                        for response in result.responses:
                            reservation.add_usage(
                                await _response_cost(provider, model_id, response)
                            )
                    finally:
                        await reservation.settle_all()
                elif result.response:

                    async def track_and_settle(response):
                        try:
                            reservation.add_usage(
                                await _response_cost(provider, model_id, response)
                            )
                        finally:
                            await reservation.settle_all()

                    await result.response.on_done(track_and_settle)
                else:
                    await reservation.settle_all()

            except Exception:
                await reservation._rollback()
                raise

    return accounting_wrapper


def llm_group_exit(datasette, group):
    """
    Settle the reservation when a group exits.

    Called by datasette-llm after all responses in the group have been
    forced to complete. Returns a coroutine that datasette-llm will await.
    """
    group_id = id(group)
    reservation = _active_reservations.pop(group_id, None)
    if reservation:
        # Return coroutine for datasette-llm to await
        return reservation.settle_all()
    return None


async def llm_filter_models(datasette, models, actor, purpose):
    """
    Filter out models that don't have pricing data.

    When accountants are registered, models without pricing can't be
    accounted for, so they shouldn't be available.
    """
    accountants = _get_accountants(datasette)
    if not accountants:
        return None

    provider = _get_pricing_provider(datasette)
    supported = await provider.supported_models()
    if supported is None:
        return models
    return [model for model in models if model.model_id in supported]
