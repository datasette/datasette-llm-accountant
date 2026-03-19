"""
Sample plugin: LLM demo page with per-actor budget management.

Provides a single page at /-/llm-demo with:
- Budget management: set, reset, delete per-actor budgets
- Prompt form: pick a model, enter a prompt, see the response with cost tracking

Budgets are stored in the internal database.
All mutations use JSON API routes (no CSRF needed).
"""

import json
import uuid

from datasette import hookimpl
from datasette.utils.asgi import Response
from datasette_debug_gotham import ACTORS

from datasette_llm_accountant import (
    Accountant,
    Tx,
    InsufficientBalanceError,
    LlmWrapper,
    AccountedModel,
    PricingProvider,
    DefaultPricingProvider,
    ModelPricingNotFoundError,
    nanocents_to_usd,
    usd_to_nanocents,
)


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS llm_budgets (
    actor_id TEXT PRIMARY KEY,
    budget_nanocents INTEGER NOT NULL,
    spent_nanocents INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS llm_budget_transactions (
    tx_id TEXT PRIMARY KEY,
    actor_id TEXT NOT NULL,
    reserved_nanocents INTEGER NOT NULL,
    settled_nanocents INTEGER,
    status TEXT DEFAULT 'reserved',
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (actor_id) REFERENCES llm_budgets(actor_id)
);
"""


async def ensure_tables(datasette):
    db = datasette.get_internal_database()
    for statement in CREATE_TABLES_SQL.strip().split(";"):
        statement = statement.strip()
        if statement:
            await db.execute_write(statement)


class BudgetAccountant(Accountant):
    """Per-actor accountant backed by the internal DB."""

    def __init__(self, datasette, actor_id: str):
        self.datasette = datasette
        self.actor_id = actor_id

    async def reserve(self, nanocents: int) -> Tx:
        await ensure_tables(self.datasette)
        db = self.datasette.get_internal_database()

        row = (
            await db.execute(
                "SELECT budget_nanocents, spent_nanocents FROM llm_budgets WHERE actor_id = ?",
                [self.actor_id],
            )
        ).first()

        if row is None:
            raise InsufficientBalanceError(
                f"No budget set for actor '{self.actor_id}'"
            )

        remaining = row["budget_nanocents"] - row["spent_nanocents"]
        if nanocents > remaining:
            raise InsufficientBalanceError(
                f"Insufficient budget: need ${nanocents_to_usd(nanocents):.6f}, "
                f"remaining ${nanocents_to_usd(remaining):.6f}"
            )

        tx_id = Tx(str(uuid.uuid4()))
        await db.execute_write(
            "INSERT INTO llm_budget_transactions (tx_id, actor_id, reserved_nanocents) VALUES (?, ?, ?)",
            [tx_id, self.actor_id, nanocents],
        )
        return tx_id

    async def settle(self, tx: Tx, nanocents: int):
        await ensure_tables(self.datasette)
        db = self.datasette.get_internal_database()

        await db.execute_write(
            "UPDATE llm_budget_transactions SET settled_nanocents = ?, status = 'settled' WHERE tx_id = ?",
            [nanocents, tx],
        )
        await db.execute_write(
            "UPDATE llm_budgets SET spent_nanocents = spent_nanocents + ?, updated_at = datetime('now') WHERE actor_id = ?",
            [nanocents, self.actor_id],
        )

    async def rollback(self, tx: Tx):
        await ensure_tables(self.datasette)
        db = self.datasette.get_internal_database()
        await db.execute_write(
            "UPDATE llm_budget_transactions SET settled_nanocents = 0, status = 'rolled_back' WHERE tx_id = ?",
            [tx],
        )


# --- JSON API routes ---

async def api_budgets_list(datasette, request):
    await ensure_tables(datasette)
    db = datasette.get_internal_database()
    rows = (await db.execute("SELECT * FROM llm_budgets ORDER BY actor_id")).rows
    budgets = []
    for row in rows:
        budget_usd = nanocents_to_usd(row["budget_nanocents"])
        spent_usd = nanocents_to_usd(row["spent_nanocents"])
        budgets.append({
            "actor_id": row["actor_id"],
            "budget_usd": budget_usd,
            "spent_usd": spent_usd,
            "remaining_usd": budget_usd - spent_usd,
        })
    return Response.json({"budgets": budgets})


async def api_budget_set(datasette, request):
    if request.method != "POST":
        return Response.json({"error": "POST required"}, status=405)
    body = json.loads(await request.post_body())
    actor_id = str(body.get("actor_id", "")).strip()
    budget_usd = body.get("budget_usd")
    if not actor_id or budget_usd is None:
        return Response.json({"error": "actor_id and budget_usd required"}, status=400)
    try:
        budget_usd = float(budget_usd)
    except (ValueError, TypeError):
        return Response.json({"error": "budget_usd must be a number"}, status=400)
    await ensure_tables(datasette)
    db = datasette.get_internal_database()
    await db.execute_write(
        """INSERT INTO llm_budgets (actor_id, budget_nanocents)
        VALUES (?, ?)
        ON CONFLICT(actor_id) DO UPDATE SET
            budget_nanocents = excluded.budget_nanocents,
            updated_at = datetime('now')""",
        [actor_id, usd_to_nanocents(budget_usd)],
    )
    return Response.json({"ok": True})


async def api_budget_reset(datasette, request):
    if request.method != "POST":
        return Response.json({"error": "POST required"}, status=405)
    body = json.loads(await request.post_body())
    actor_id = str(body.get("actor_id", "")).strip()
    if not actor_id:
        return Response.json({"error": "actor_id required"}, status=400)
    await ensure_tables(datasette)
    db = datasette.get_internal_database()
    await db.execute_write(
        "UPDATE llm_budgets SET spent_nanocents = 0, updated_at = datetime('now') WHERE actor_id = ?",
        [actor_id],
    )
    return Response.json({"ok": True})


async def api_budget_delete(datasette, request):
    if request.method != "POST":
        return Response.json({"error": "POST required"}, status=405)
    body = json.loads(await request.post_body())
    actor_id = str(body.get("actor_id", "")).strip()
    if not actor_id:
        return Response.json({"error": "actor_id required"}, status=400)
    await ensure_tables(datasette)
    db = datasette.get_internal_database()
    await db.execute_write("DELETE FROM llm_budget_transactions WHERE actor_id = ?", [actor_id])
    await db.execute_write("DELETE FROM llm_budgets WHERE actor_id = ?", [actor_id])
    return Response.json({"ok": True})


async def api_models_list(datasette, request):
    wrapper = LlmWrapper(datasette)
    models = [m.model_id for m in wrapper.get_async_models()]
    return Response.json({"models": models})


async def api_prompt(datasette, request):
    if request.method != "POST":
        return Response.json({"error": "POST required"}, status=405)
    body = json.loads(await request.post_body())
    model_id = str(body.get("model", "")).strip()
    prompt_text = str(body.get("prompt", "")).strip()

    if not model_id or not prompt_text:
        return Response.json({"error": "model and prompt required"}, status=400)

    actor = request.actor
    actor_id = (actor.get("id") if actor else None) or "anonymous"
    accountant = BudgetAccountant(datasette, actor_id)
    wrapper = LlmWrapper(datasette)

    try:
        model = AccountedModel(
            wrapper.get_async_model(model_id)._async_model,
            accountants=[accountant],
            pricing_provider=wrapper._get_pricing_provider(),
        )
        response = await model.prompt(prompt_text, usd=0.50)
        response_text = await response.text()
        usage = await response.usage()
        input_tokens = usage.input or 0
        output_tokens = usage.output or 0
        cost_usd = nanocents_to_usd(
            wrapper._get_pricing_provider().calculate_cost_nanocents(
                model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
        return Response.json({
            "response": response_text,
            "model": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        })
    except InsufficientBalanceError as e:
        return Response.json({"error": f"Budget error: {e}"}, status=402)
    except Exception as e:
        return Response.json({"error": str(e)}, status=500)


# --- Page route (GET only, serves template) ---

async def llm_demo_page(datasette, request):
    actor = request.actor
    actor_id = (actor.get("id") if actor else None) or "anonymous"
    return Response.html(
        await datasette.render_template(
            "llm_demo.html",
            {
                "actor_id": actor_id,
                "actors": [{"id": k, "name": v["name"]} for k, v in ACTORS.items()],
            },
            request=request,
        )
    )


# --- Pricing: only allow gpt-5-nano ---

ALLOWED_MODELS = {"gpt-5-nano"}


class DemoPricingProvider(PricingProvider):
    """Wraps the default provider but only exposes allowed models."""

    def __init__(self):
        self._default = DefaultPricingProvider()

    def get_model_pricing(self, model_id: str) -> dict:
        if model_id not in ALLOWED_MODELS:
            raise ModelPricingNotFoundError(
                f"Model '{model_id}' is not enabled in this demo. "
                f"Allowed: {', '.join(sorted(ALLOWED_MODELS))}"
            )
        pricing = dict(self._default.get_model_pricing(model_id))
        pricing["input"] = pricing["input"] * 100
        pricing["output"] = pricing["output"] * 100
        if pricing.get("input_cached") is not None:
            pricing["input_cached"] = pricing["input_cached"] * 100
        return pricing


@hookimpl
def register_llm_accountant_pricing(datasette):
    return DemoPricingProvider()


@hookimpl
def register_routes():
    return [
        (r"^/-/llm-demo$", llm_demo_page),
        (r"^/-/llm-demo/api/budgets$", api_budgets_list),
        (r"^/-/llm-demo/api/budgets/set$", api_budget_set),
        (r"^/-/llm-demo/api/budgets/reset$", api_budget_reset),
        (r"^/-/llm-demo/api/budgets/delete$", api_budget_delete),
        (r"^/-/llm-demo/api/models$", api_models_list),
        (r"^/-/llm-demo/api/prompt$", api_prompt),
    ]
