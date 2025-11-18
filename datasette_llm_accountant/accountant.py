"""
Base classes for LLM accounting.
"""

from abc import ABC, abstractmethod


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
    async def reserve(self, nanocents: int) -> Tx:
        """
        Reserve the specified amount in nanocents.

        Args:
            nanocents: Amount to reserve in nanocents (1/1,000,000,000 of a cent)

        Returns:
            A transaction ID that will be used for settlement

        Raises:
            InsufficientBalanceError: If the reservation cannot be made
        """
        pass

    @abstractmethod
    async def settle(self, tx: Tx, nanocents: int):
        """
        Settle a transaction for the actual amount spent.

        Args:
            tx: Transaction ID returned from reserve()
            nanocents: Actual amount spent in nanocents
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
        await self.settle(tx, 0)
