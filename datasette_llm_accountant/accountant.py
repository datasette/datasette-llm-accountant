"""
Base classes for LLM accounting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .pricing import Nanocents


class Tx(str):
    """
    A transaction ID string.

    Accountants return this from reserve() and use it to track settlements.
    """

    pass


class InsufficientBalanceError(Exception):
    """Raised when an accountant cannot reserve the requested amount."""

    pass


class Accountant(ABC):
    """
    Base class for accountant implementations.

    Accountants track LLM token usage costs and enforce spending limits.
    They are registered via the register_llm_accountants() plugin hook.
    """

    @abstractmethod
    async def reserve(
        self,
        nanocents: Nanocents,
        model_id: Optional[str] = None,
        purpose: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> Tx:
        """
        Reserve the specified amount in nanocents.

        Args:
            nanocents: Amount to reserve in nanocents (1/1,000,000,000 of a cent)
            model_id: The model being used (e.g., "gpt-4o-mini")
            purpose: The purpose of the request (e.g., "query-assistant")
            actor_id: The ID of the actor making the request

        Returns:
            A transaction ID that will be used for settlement

        Raises:
            InsufficientBalanceError: If the reservation cannot be made
        """
        pass

    @abstractmethod
    async def settle(
        self,
        tx: Tx,
        nanocents: Nanocents,
        model_id: Optional[str] = None,
        purpose: Optional[str] = None,
        actor_id: Optional[str] = None,
    ):
        """
        Settle a transaction for the actual amount spent.

        Args:
            tx: Transaction ID returned from reserve()
            nanocents: Actual amount spent in nanocents
            model_id: The model that was used
            purpose: The purpose of the request
            actor_id: The ID of the actor making the request
        """
        pass

    async def rollback(self, tx: Tx):
        """
        Rollback/release a reservation.

        Called when a reservation needs to be cancelled (e.g., if another
        accountant fails to reserve, or if the LLM call fails).

        The default implementation settles for 0, but accountants may override
        this for different behavior.

        Args:
            tx: Transaction ID to rollback
        """
        from .pricing import Nanocents

        await self.settle(tx, Nanocents(0))
