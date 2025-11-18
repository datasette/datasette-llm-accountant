# datasette-llm-accountant

[![PyPI](https://img.shields.io/pypi/v/datasette-llm-accountant.svg)](https://pypi.org/project/datasette-llm-accountant/)
[![Changelog](https://img.shields.io/github/v/release/datasette/datasette-llm-accountant?include_prereleases&label=changelog)](https://github.com/datasette/datasette-llm-accountant/releases)
[![Tests](https://github.com/datasette/datasette-llm-accountant/actions/workflows/test.yml/badge.svg)](https://github.com/datasette/datasette-llm-accountant/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/datasette/datasette-llm-accountant/blob/main/LICENSE)

Accounting for LLM token usage

## Installation

Install this plugin in the same environment as Datasette.
```bash
datasette install datasette-llm-accountant
```
## Overview

This plugin provides a library for other Datasette plugins to interact with LLMs via a layer that does accounting and cost tracking. It wraps the [llm](https://llm.datasette.io/) library with automatic cost calculation and spending limits.

## Usage

### Basic Usage

```python
from datasette_llm_accountant import LlmWrapper

# In your Datasette plugin
llm = LlmWrapper(datasette)

# Get an async model wrapped with accounting
model = llm.get_async_model("gpt-4o-mini")

# Use with automatic reservation (defaults to 50 cents)
response = await model.prompt("Write a poem about a pirate")
```

### Manual Reservations

You can manually reserve an amount and track spending across multiple prompts:

```python
# Reserve $4.50 for multiple prompts
async with model.reserve(usd=4.50) as tx:
    response1 = await tx.prompt("First question")
    response2 = await tx.prompt("Follow-up question")
    # Total spending is tracked cumulatively
```

You can also specify reservations in nanocents (1/1,000,000,000 of a cent):

```python
async with model.reserve(nanocents=50_000_000_000) as tx:  # 50 cents
    response = await tx.prompt("a poem about a pirate")
```

### Cost Protection

If a prompt exceeds your reservation, a `ReservationExceededError` is raised:

```python
from datasette_llm_accountant import ReservationExceededError

try:
    async with model.reserve(usd=0.01) as tx:  # Very small reservation
        # This might exceed the reservation
        response = await tx.prompt("Write a very long story...")
except ReservationExceededError as e:
    print(f"Exceeded budget: {e}")
```

## Creating Accountant Plugins

Other plugins can implement accountants to track and limit LLM spending. Create a plugin that implements the `register_llm_accountants` hook:

```python
from datasette import hookimpl
from datasette_llm_accountant import Accountant, Tx, InsufficientBalanceError

class DatabaseAccountant(Accountant):
    """Tracks spending in a database."""

    async def reserve(self, nanocents: int) -> Tx:
        # Check if user has sufficient balance
        # If not, raise InsufficientBalanceError
        # Otherwise, create and return a transaction ID
        tx_id = await self.create_reservation(nanocents)
        return Tx(tx_id)

    async def settle(self, tx: Tx, nanocents: int):
        # Record the actual amount spent for this transaction
        await self.record_settlement(tx, nanocents)

    async def rollback(self, tx: Tx):
        # Optional: Release/cancel the reservation
        # Default implementation settles for 0
        await self.cancel_reservation(tx)

@hookimpl
def register_llm_accountants(datasette):
    return [DatabaseAccountant(datasette)]
```

### Multiple Accountants

The system supports multiple accountants. When a reservation is made:

1. All accountants are called in sequence to reserve the amount
2. If any accountant fails, all previous reservations are rolled back
3. When the transaction completes, all accountants are settled with the actual cost

This allows for multiple layers of accounting (e.g., per-user limits, per-project limits, global limits).

## Cost Calculation

Costs are automatically calculated based on token usage and current pricing data from [llm-prices.com](https://www.llm-prices.com/). The library:

- Tracks input and output tokens separately
- Handles cached input tokens when available
- Converts costs to nanocents for precise accounting
- Works with 80+ models across multiple providers

## API Reference

### LlmWrapper

```python
wrapper = LlmWrapper(datasette)
model = wrapper.get_async_model("model-id")
```

### AccountedModel

```python
# Direct prompt with auto-reservation
response = await model.prompt("text", usd=0.5)

# Manual reservation
async with model.reserve(usd=1.0) as tx:
    response = await tx.prompt("text")
```

### Accountant Base Class

```python
class Accountant(ABC):
    async def reserve(self, nanocents: int) -> Tx: ...
    async def settle(self, tx: Tx, nanocents: int): ...
    async def rollback(self, tx: Tx): ...  # Optional override
```

### Exceptions

- `InsufficientBalanceError` - Accountant cannot reserve the requested amount
- `ReservationExceededError` - Actual cost exceeds reserved amount
- `ModelPricingNotFoundError` - Pricing data not available for model

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd datasette-llm-accountant
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```
