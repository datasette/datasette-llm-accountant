"""
Sample plugin: LLM demo with per-actor budget management.

Provides:
- /-/demo-admin (root only): manage budgets, model pricing
- /-/demo (any actor with a budget): prompt UI with cost tracking

Budgets and pricing are stored in the internal database.
All mutations use JSON API routes (no CSRF needed).
"""

import json
import uuid
from typing import Optional

from datasette import hookimpl
from datasette.utils.asgi import Response
from datasette_debug_gotham import ACTORS

from datasette_llm_accountant import (
    Accountant,
    Tx,
    InsufficientBalanceError,
    Nanocents,
    PricingProvider,
    ModelPricingNotFoundError,
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

CREATE TABLE IF NOT EXISTS llm_model_pricing (
    model_id TEXT PRIMARY KEY,
    input_price REAL NOT NULL,
    output_price REAL NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
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

    def __init__(self, datasette):
        self.datasette = datasette

    async def reserve(
        self,
        nanocents: int,
        model_id: Optional[str] = None,
        purpose: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> Tx:
        await ensure_tables(self.datasette)
        db = self.datasette.get_internal_database()

        row = (
            await db.execute(
                "SELECT budget_nanocents, spent_nanocents FROM llm_budgets WHERE actor_id = ?",
                [actor_id],
            )
        ).first()

        if row is None:
            raise InsufficientBalanceError(f"No budget set for actor '{actor_id}'")

        remaining = row["budget_nanocents"] - row["spent_nanocents"]
        if nanocents > remaining:
            raise InsufficientBalanceError(
                f"Insufficient budget: need ${Nanocents(nanocents).to_usd():.6f}, "
                f"remaining ${Nanocents(remaining).to_usd():.6f}"
            )

        tx_id = Tx(str(uuid.uuid4()))
        await db.execute_write(
            "INSERT INTO llm_budget_transactions (tx_id, actor_id, reserved_nanocents) VALUES (?, ?, ?)",
            [tx_id, actor_id, nanocents],
        )
        return tx_id

    async def settle(
        self,
        tx: Tx,
        nanocents: int,
        model_id: Optional[str] = None,
        purpose: Optional[str] = None,
        actor_id: Optional[str] = None,
    ):
        if not actor_id:
            return  # No actor — nothing to settle against
        await ensure_tables(self.datasette)
        db = self.datasette.get_internal_database()

        await db.execute_write(
            "UPDATE llm_budget_transactions SET settled_nanocents = ?, status = 'settled' WHERE tx_id = ?",
            [nanocents, tx],
        )
        await db.execute_write(
            "UPDATE llm_budgets SET spent_nanocents = spent_nanocents + ?, updated_at = datetime('now') WHERE actor_id = ?",
            [nanocents, actor_id],
        )

    async def rollback(self, tx: Tx):
        if tx == "no-actor":
            return
        await ensure_tables(self.datasette)
        db = self.datasette.get_internal_database()
        await db.execute_write(
            "UPDATE llm_budget_transactions SET settled_nanocents = 0, status = 'rolled_back' WHERE tx_id = ?",
            [tx],
        )


class DemoPricingProvider(PricingProvider):
    """Provides pricing from admin-configured per-model rates.

    Queries the DB directly — no in-memory cache needed since
    calculate_cost_from_response is async.
    """

    def __init__(self, datasette):
        self.datasette = datasette

    async def _get_pricing(self, model_id: str) -> dict:
        await ensure_tables(self.datasette)
        db = self.datasette.get_internal_database()
        row = (
            await db.execute(
                "SELECT input_price, output_price FROM llm_model_pricing WHERE model_id = ?",
                [model_id],
            )
        ).first()
        if row is None:
            raise ModelPricingNotFoundError(
                f"No pricing configured for '{model_id}'. An admin must set it first."
            )
        return {"input": row["input_price"], "output": row["output_price"]}

    async def supported_models(self):
        await ensure_tables(self.datasette)
        db = self.datasette.get_internal_database()
        rows = (await db.execute("SELECT model_id FROM llm_model_pricing")).rows
        return {row["model_id"] for row in rows}

    async def calculate_cost_from_response(
        self, model_id, usage, response
    ) -> Nanocents:
        pricing = await self._get_pricing(model_id)
        input_tokens = usage.input or 0
        output_tokens = usage.output or 0
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return Nanocents(int((input_cost + output_cost) * 1_000_000_000))


# --- Routes ---


class Routes:
    @staticmethod
    async def budgets_list(datasette, request):
        await ensure_tables(datasette)
        db = datasette.get_internal_database()
        rows = (await db.execute("SELECT * FROM llm_budgets ORDER BY actor_id")).rows
        budgets = []
        for row in rows:
            budget_usd = Nanocents(row["budget_nanocents"]).to_usd()
            spent_usd = Nanocents(row["spent_nanocents"]).to_usd()
            budgets.append(
                {
                    "actor_id": row["actor_id"],
                    "budget_usd": budget_usd,
                    "spent_usd": spent_usd,
                    "remaining_usd": budget_usd - spent_usd,
                }
            )
        return Response.json({"budgets": budgets})

    @staticmethod
    async def budget_set(datasette, request):
        if request.method != "POST":
            return Response.json({"error": "POST required"}, status=405)
        body = json.loads(await request.post_body())
        actor_id = str(body.get("actor_id", "")).strip()
        budget_usd = body.get("budget_usd")
        if not actor_id or budget_usd is None:
            return Response.json(
                {"error": "actor_id and budget_usd required"}, status=400
            )
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
            [actor_id, Nanocents.from_usd(budget_usd)],
        )
        return Response.json({"ok": True})

    @staticmethod
    async def budget_reset(datasette, request):
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

    @staticmethod
    async def budget_delete(datasette, request):
        if request.method != "POST":
            return Response.json({"error": "POST required"}, status=405)
        body = json.loads(await request.post_body())
        actor_id = str(body.get("actor_id", "")).strip()
        if not actor_id:
            return Response.json({"error": "actor_id required"}, status=400)
        await ensure_tables(datasette)
        db = datasette.get_internal_database()
        await db.execute_write(
            "DELETE FROM llm_budget_transactions WHERE actor_id = ?", [actor_id]
        )
        await db.execute_write("DELETE FROM llm_budgets WHERE actor_id = ?", [actor_id])
        return Response.json({"ok": True})

    @staticmethod
    async def models_list(datasette, request):
        import llm as llm_library
        from datasette_llm import LLM

        llm = LLM(datasette)
        all_models = [m.model_id for m in llm_library.get_async_models()]
        available = [m.model_id for m in await llm.models()]
        await ensure_tables(datasette)
        db = datasette.get_internal_database()
        rows = (
            await db.execute(
                "SELECT model_id, input_price, output_price FROM llm_model_pricing"
            )
        ).rows
        pricing = {
            row["model_id"]: {
                "input": row["input_price"],
                "output": row["output_price"],
            }
            for row in rows
        }
        return Response.json(
            {
                "all_models": all_models,
                "models": available,
                "pricing": pricing,
            }
        )

    @staticmethod
    async def model_set_pricing(datasette, request):
        if request.method != "POST":
            return Response.json({"error": "POST required"}, status=405)
        if not request.actor or request.actor.get("id") != "root":
            return Response.json({"error": "root access required"}, status=403)
        body = json.loads(await request.post_body())
        model_id = str(body.get("model_id", "")).strip()
        input_price = body.get("input_price")
        output_price = body.get("output_price")
        if not model_id or input_price is None or output_price is None:
            return Response.json(
                {"error": "model_id, input_price, and output_price required"},
                status=400,
            )
        try:
            input_price = float(input_price)
            output_price = float(output_price)
        except (ValueError, TypeError):
            return Response.json(
                {"error": "input_price and output_price must be numbers"}, status=400
            )
        await ensure_tables(datasette)
        db = datasette.get_internal_database()
        await db.execute_write(
            """INSERT INTO llm_model_pricing (model_id, input_price, output_price)
            VALUES (?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                input_price = excluded.input_price,
                output_price = excluded.output_price""",
            [model_id, input_price, output_price],
        )
        return Response.json({"ok": True})

    @staticmethod
    async def model_remove_pricing(datasette, request):
        if request.method != "POST":
            return Response.json({"error": "POST required"}, status=405)
        if not request.actor or request.actor.get("id") != "root":
            return Response.json({"error": "root access required"}, status=403)
        body = json.loads(await request.post_body())
        model_id = str(body.get("model_id", "")).strip()
        if not model_id:
            return Response.json({"error": "model_id required"}, status=400)
        await ensure_tables(datasette)
        db = datasette.get_internal_database()
        await db.execute_write(
            "DELETE FROM llm_model_pricing WHERE model_id = ?",
            [model_id],
        )
        return Response.json({"ok": True})

    @staticmethod
    async def prompt(datasette, request):
        if request.method != "POST":
            return Response.json({"error": "POST required"}, status=405)
        body = json.loads(await request.post_body())
        model_id = str(body.get("model", "")).strip()
        prompt_text = str(body.get("prompt", "")).strip()

        if not model_id or not prompt_text:
            return Response.json({"error": "model and prompt required"}, status=400)

        actor = request.actor

        from datasette_llm import LLM

        try:
            llm = LLM(datasette)
            model = await llm.model(model_id, purpose="llm-demo", actor=actor)
            response = await model.prompt(prompt_text)
            response_text = await response.text()
            usage = await response.usage()

            # Cost calculation + settlement happens automatically via llm_prompt_context hook
            return Response.json(
                {
                    "response": response_text,
                    "model": model_id,
                    "input_tokens": usage.input or 0,
                    "output_tokens": usage.output or 0,
                }
            )
        except InsufficientBalanceError as e:
            return Response.json({"error": f"Budget error: {e}"}, status=402)
        except Exception as e:
            return Response.json({"error": str(e)}, status=500)

    @staticmethod
    async def my_budget(datasette, request):
        """Return the current actor's budget info."""
        actor = request.actor
        actor_id = actor.get("id") if actor else None
        if not actor_id:
            return Response.json({"error": "Not logged in"}, status=401)
        await ensure_tables(datasette)
        db = datasette.get_internal_database()
        row = (
            await db.execute(
                "SELECT budget_nanocents, spent_nanocents FROM llm_budgets WHERE actor_id = ?",
                [actor_id],
            )
        ).first()
        if row is None:
            return Response.json({"budget": None})
        budget_usd = Nanocents(row["budget_nanocents"]).to_usd()
        spent_usd = Nanocents(row["spent_nanocents"]).to_usd()
        return Response.json(
            {
                "budget": {
                    "actor_id": actor_id,
                    "budget_usd": budget_usd,
                    "spent_usd": spent_usd,
                    "remaining_usd": budget_usd - spent_usd,
                }
            }
        )

    @staticmethod
    async def demo_admin_page(datasette, request):
        actor = request.actor
        if not actor or actor.get("id") != "root":
            return Response.text("Root access required", status=403)
        return Response.html(
            await datasette.render_template(
                "demo_admin.html",
                {
                    "actor_id": actor.get("id"),
                    "actors": [{"id": k, "name": v["name"]} for k, v in ACTORS.items()],
                },
                request=request,
            )
        )

    @staticmethod
    async def demo_page(datasette, request):
        actor = request.actor
        if not actor:
            return Response.text("Login required", status=403)
        actor_id = actor.get("id")
        return Response.html(
            await datasette.render_template(
                "demo.html",
                {"actor_id": actor_id},
                request=request,
            )
        )


# --- Hooks ---


@hookimpl
def register_llm_accountants(datasette):
    return [BudgetAccountant(datasette)]


@hookimpl
def register_llm_accountant_pricing(datasette):
    return DemoPricingProvider(datasette)


@hookimpl
def menu_links(datasette, actor, request):
    links = []
    if actor and actor.get("id") == "root":
        links.append(
            {
                "href": datasette.urls.path("/-/demo-admin"),
                "label": "Demo Admin",
            }
        )
    if actor:
        links.append(
            {
                "href": datasette.urls.path("/-/demo"),
                "label": "Demo",
            }
        )
    return links


@hookimpl
def register_routes():
    return [
        (r"^/-/demo-admin$", Routes.demo_admin_page),
        (r"^/-/demo$", Routes.demo_page),
        (r"^/-/demo/api/budgets$", Routes.budgets_list),
        (r"^/-/demo/api/budgets/set$", Routes.budget_set),
        (r"^/-/demo/api/budgets/reset$", Routes.budget_reset),
        (r"^/-/demo/api/budgets/delete$", Routes.budget_delete),
        (r"^/-/demo/api/models$", Routes.models_list),
        (r"^/-/demo/api/models/set-pricing$", Routes.model_set_pricing),
        (r"^/-/demo/api/models/remove-pricing$", Routes.model_remove_pricing),
        (r"^/-/demo/api/prompt$", Routes.prompt),
        (r"^/-/demo/api/my-budget$", Routes.my_budget),
    ]
