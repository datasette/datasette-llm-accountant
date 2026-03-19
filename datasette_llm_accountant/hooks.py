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
    DefaultPricingProvider,
    PricingProvider,
    calculate_cost_nanocents,
    usd_to_nanocents,
    ModelPricingNotFoundError,
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
        nanocents: int,
        accountants: List[Accountant],
        model_id: str = None,
        purpose: str = None,
    ):
        self.nanocents = nanocents
        self.accountants = accountants
        self.model_id = model_id
        self.purpose = purpose
        self.transactions: List[tuple[Accountant, Tx]] = []
        self.spent_nanocents = 0
        self._settled = False

    async def reserve_all(self):
        """Reserve from all accountants, rolling back on failure."""
        for accountant in self.accountants:
            try:
                tx = await accountant.reserve(
                    self.nanocents,
                    model_id=self.model_id,
                    purpose=self.purpose,
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
            )

    async def _rollback(self):
        """Rollback all successful reservations."""
        for accountant, tx in self.transactions:
            try:
                await accountant.rollback(tx)
            except Exception:
                pass  # Log but continue rolling back others

    def add_usage(self, nanocents: int):
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


def _calculate_reservation_nanocents(datasette, model_id, purpose) -> int:
    """Calculate reservation amount from configuration."""
    config = _get_config(datasette)

    # Check purpose-specific config first
    purposes = config.get("purposes", {})
    if purpose and purpose in purposes:
        purpose_config = purposes[purpose]
        if "reservation_nanocents" in purpose_config:
            return purpose_config["reservation_nanocents"]
        elif "reservation_usd" in purpose_config:
            return usd_to_nanocents(purpose_config["reservation_usd"])

    # Fall back to model-specific config
    models = config.get("models", {})
    if model_id in models:
        model_config = models[model_id]
        if "reservation_nanocents" in model_config:
            return model_config["reservation_nanocents"]
        elif "reservation_usd" in model_config:
            return usd_to_nanocents(model_config["reservation_usd"])

    # Fall back to global default
    if "default_reservation_nanocents" in config:
        return config["default_reservation_nanocents"]
    elif "default_reservation_usd" in config:
        return usd_to_nanocents(config["default_reservation_usd"])

    # Default: $0.50
    return usd_to_nanocents(0.50)


def llm_prompt_context(datasette, model_id, prompt, purpose):
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

        if group is not None:
            # Part of a group - use/create group's reservation
            group_id = id(group)

            # Create reservation on first prompt in group
            if group_id not in _active_reservations:
                nanocents = _calculate_reservation_nanocents(
                    datasette, model_id, purpose
                )
                reservation = GroupReservation(
                    nanocents, accountants, model_id=model_id, purpose=purpose
                )
                await reservation.reserve_all()
                _active_reservations[group_id] = reservation

            reservation = _active_reservations.get(group_id)

            yield

            # Track usage via on_done callback
            if reservation and result.response:

                async def track_group_usage(response):
                    try:
                        usage = await response.usage()
                        cost = provider.calculate_cost_nanocents(
                            model_id,
                            input_tokens=usage.input or 0,
                            output_tokens=usage.output or 0,
                        )
                        reservation.add_usage(cost)

                        if reservation.exceeded():
                            raise ReservationExceededError(
                                f"Cost {reservation.spent_nanocents} nanocents "
                                f"exceeds reservation of {reservation.nanocents} nanocents"
                            )
                    except ModelPricingNotFoundError:
                        pass  # Model doesn't have pricing data
                    except Exception:
                        pass  # Other errors (e.g., model doesn't support usage)

                await result.response.on_done(track_group_usage)

        else:
            # Not part of a group - auto-reserve for this single prompt
            config = _get_config(datasette)

            # Get auto-reservation amount (smaller default for single prompts)
            auto_nanocents = config.get("auto_reservation_nanocents")
            if auto_nanocents is None:
                auto_usd = config.get("auto_reservation_usd", 0.10)  # Default: $0.10
                auto_nanocents = usd_to_nanocents(auto_usd)

            # Create a single-prompt reservation
            reservation = GroupReservation(
                auto_nanocents, accountants, model_id=model_id, purpose=purpose
            )
            await reservation.reserve_all()

            try:
                yield

                # Track usage and settle via on_done
                if result.response:

                    async def track_and_settle(response):
                        try:
                            usage = await response.usage()
                            cost = provider.calculate_cost_nanocents(
                                model_id,
                                input_tokens=usage.input or 0,
                                output_tokens=usage.output or 0,
                            )
                            reservation.add_usage(cost)
                        except (ModelPricingNotFoundError, Exception):
                            pass  # Settle with whatever we tracked
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
