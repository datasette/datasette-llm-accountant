# datasette-llm-accountant

[![PyPI](https://img.shields.io/pypi/v/datasette-llm-accountant.svg)](https://pypi.org/project/datasette-llm-accountant/)
[![Changelog](https://img.shields.io/github/v/release/datasette/datasette-llm-accountant?include_prereleases&label=changelog)](https://github.com/datasette/datasette-llm-accountant/releases)
[![Tests](https://github.com/datasette/datasette-llm-accountant/actions/workflows/test.yml/badge.svg)](https://github.com/datasette/datasette-llm-accountant/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/datasette/datasette-llm-accountant/blob/main/LICENSE)

Budget management and cost tracking for LLM usage in Datasette.

## Installation

Install this plugin in the same environment as Datasette:

```bash
datasette install datasette-llm-accountant
```

This plugin works alongside [datasette-llm](https://github.com/datasette/datasette-llm) to provide automatic cost tracking and budget enforcement for LLM prompts.

## Overview

This plugin provides:

- **Automatic cost calculation** based on token usage and model pricing
- **Reserve/settle pattern** for budget enforcement
- **Accountant plugin system** for custom spending trackers
- **Hook integration** with datasette-llm for transparent accounting

When installed, all prompts made through `datasette-llm` are automatically wrapped with accounting logic. Accountants can enforce spending limits, log usage, and track costs.

## How It Works

1. When a prompt is made via `datasette-llm`, this plugin's hooks intercept the call
2. A reservation is made with all registered accountants for the estimated cost
3. The prompt executes
4. The actual cost is calculated from token usage
5. Accountants are settled with the real cost (refunding any unused reservation)

## Configuration

Configure reservation amounts in `datasette.yaml`:

```yaml
plugins:
  datasette-llm-accountant:
    # Default reservation for single prompts
    auto_reservation_usd: 0.10

    # Default reservation for grouped prompts
    default_reservation_usd: 0.50

    # Purpose-specific reservations
    purposes:
      enrichments:
        reservation_usd: 5.00
      query-assistant:
        reservation_usd: 0.25
```

## Creating an Accountant Plugin

Accountants track and enforce LLM spending. Create a plugin that implements the `register_llm_accountants` hook:

```python
from datasette import hookimpl
from datasette_llm_accountant import Accountant, Tx, InsufficientBalanceError

class MyAccountant(Accountant):
    """Custom accountant that tracks spending."""

    def __init__(self, datasette):
        self.datasette = datasette

    async def reserve(
        self,
        nanocents: int,
        model_id: str = None,
        purpose: str = None,
    ) -> Tx:
        """
        Reserve the specified amount.

        Args:
            nanocents: Amount to reserve (1 USD = 100,000,000,000 nanocents)
            model_id: The model being used (e.g., "gpt-4o-mini")
            purpose: The purpose of the request (e.g., "enrichments")

        Returns:
            A transaction ID for settlement

        Raises:
            InsufficientBalanceError: If reservation cannot be made
        """
        # Check balance, create reservation, return transaction ID
        if not await self.has_sufficient_balance(nanocents):
            raise InsufficientBalanceError("Insufficient balance")

        tx_id = await self.create_reservation(nanocents, model_id, purpose)
        return Tx(tx_id)

    async def settle(
        self,
        tx: Tx,
        nanocents: int,
        model_id: str = None,
        purpose: str = None,
    ):
        """
        Settle a transaction for the actual amount spent.

        Args:
            tx: Transaction ID from reserve()
            nanocents: Actual amount spent
            model_id: The model that was used
            purpose: The purpose of the request
        """
        await self.record_settlement(tx, nanocents, model_id, purpose)

    async def rollback(self, tx: Tx):
        """Optional: Release a reservation without charging."""
        await self.settle(tx, 0)

@hookimpl
def register_llm_accountants(datasette):
    return [MyAccountant(datasette)]
```

See [datasette-llm-allowance](https://github.com/datasette/datasette-llm-allowance) for a complete implementation that uses Datasette's internal database to track a spending allowance.

## Multiple Accountants

Multiple accountants can be registered. When a reservation is made:

1. All accountants are called in sequence to reserve the amount
2. If any accountant fails (e.g., `InsufficientBalanceError`), previous reservations are rolled back
3. When the prompt completes, all accountants are settled with the actual cost

This enables layered accounting (per-user limits, per-project budgets, global caps, etc.).

## Cost Calculation

Costs are calculated using pricing data from [llm-prices.com](https://www.llm-prices.com/):

```python
from datasette_llm_accountant import calculate_cost_nanocents

cost = calculate_cost_nanocents(
    model_id="gpt-4o-mini",
    input_tokens=1000,
    output_tokens=500,
    cached_input_tokens=200,  # Optional
)
# Returns cost in nanocents
```

### Pricing Utilities

```python
from datasette_llm_accountant import (
    usd_to_nanocents,
    nanocents_to_usd,
    get_model_pricing,
)

# Convert between USD and nanocents
nanocents = usd_to_nanocents(1.50)  # 150,000,000,000
usd = nanocents_to_usd(150_000_000_000)  # 1.5

# Get pricing for a model
pricing = get_model_pricing("gpt-4o-mini")
# Returns: {"input": 0.15, "output": 0.6, "cached_input": 0.075}
# Prices are per million tokens
```

## API Reference

### Accountant Base Class

```python
class Accountant(ABC):
    @abstractmethod
    async def reserve(
        self,
        nanocents: int,
        model_id: str = None,
        purpose: str = None,
    ) -> Tx:
        """Reserve an amount, return transaction ID."""
        pass

    @abstractmethod
    async def settle(
        self,
        tx: Tx,
        nanocents: int,
        model_id: str = None,
        purpose: str = None,
    ):
        """Settle a transaction for the actual amount."""
        pass

    async def rollback(self, tx: Tx):
        """Release a reservation (default: settle for 0)."""
        await self.settle(tx, 0)
```

### Exceptions

- `InsufficientBalanceError` - Raised when an accountant cannot reserve the requested amount
- `ReservationExceededError` - Raised when actual cost exceeds the reserved amount
- `ModelPricingNotFoundError` - Raised when pricing data is not available for a model

### Nanocents

All amounts use nanocents for precision:

- 1 nanocent = 1/1,000,000,000 of a cent
- 1 USD = 100 cents = 100,000,000,000 nanocents
- This allows tracking costs down to fractions of a cent without floating-point errors

## Development

```bash
cd datasette-llm-accountant
python -m venv venv
source venv/bin/activate
pip install -e '.[test]'
python -m pytest
```
