"""
Sample plugin: OpenRouter integration for datasette-llm.

Provides:
- Dynamic model registration from OpenRouter's model catalog
- Pricing from OpenRouter's per-model rates
- Exact cost calculation via OpenRouter's generation API
- Usage logging accountant backed by internal DB
- Admin UI at /-/openrouter

Requires OPENROUTER_API_KEY environment variable (or set via llm keys).
"""

import json
import os
import uuid
from typing import Optional

import httpx
from datasette import hookimpl
from datasette.utils.asgi import Response

from datasette_llm_accountant import (
    Accountant,
    Tx,
    InsufficientBalanceError,
    Nanocents,
    PricingProvider,
    ModelPricingNotFoundError,
)

import llm
from llm.default_plugins.openai_models import Chat, AsyncChat

# --- OpenRouter API client ---

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_API_BASE}/models"
OPENROUTER_GENERATION_URL = f"{OPENROUTER_API_BASE}/generation"
_TIMEOUT = 15.0

# Cached model catalog: {model_id: model_data_dict}
_model_catalog: dict[str, dict] = {}


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or ""
    if not key:
        try:
            key = llm.get_key("", "openrouter") or ""
        except Exception:
            pass
    return key


def _fetch_models() -> dict[str, dict]:
    """Fetch and cache model catalog from OpenRouter.

    Set OPENROUTER_MODEL_FILTER to a comma-separated list of substrings
    to only include matching model IDs (e.g. "gemini,claude").
    """
    if _model_catalog:
        return _model_catalog

    headers = {}
    key = _get_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    resp = httpx.get(OPENROUTER_MODELS_URL, headers=headers, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    model_filter = os.environ.get("OPENROUTER_MODEL_FILTER", "")
    filter_terms = [t.strip().lower() for t in model_filter.split(",") if t.strip()]

    for model in data.get("data", []):
        model_id = model["id"]
        # Only include text-output models
        output_modalities = model.get("architecture", {}).get("output_modalities", [])
        if "text" not in output_modalities:
            continue
        # Apply filter if set
        if filter_terms and not any(t in model_id.lower() for t in filter_terms):
            continue
        _model_catalog[model_id] = model

    return _model_catalog


def _get_generation(generation_id: str) -> Optional[dict]:
    """Look up a generation's metadata from OpenRouter."""
    key = _get_api_key()
    if not key:
        return None
    resp = httpx.get(
        OPENROUTER_GENERATION_URL,
        params={"id": generation_id},
        headers={"Authorization": f"Bearer {key}"},
        timeout=_TIMEOUT,
    )
    if resp.status_code != 200:
        return None
    return resp.json().get("data")


# --- LLM model registration ---


# The --plugins-dir loader only registers with datasette's plugin manager,
# not llm's. We need to manually register this module with llm's PM so
# the register_models hook fires. We use a wrapper class since the module
# object isn't in sys.modules when loaded via --plugins-dir.
class _LlmPluginHooks:
    @staticmethod
    @llm.hookimpl
    def register_models(register):
        _do_register_models(register)


llm.plugins.pm.register(_LlmPluginHooks(), name="openrouter-plugin")


def _do_register_models(register):
    """Register all OpenRouter text models as llm models."""
    catalog = _fetch_models()
    for model_id, model_data in catalog.items():
        pricing = model_data.get("pricing", {})
        # Skip free models with no pricing
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")
        if prompt_price is None and completion_price is None:
            continue

        supports_tools = "tools" in model_data.get("supported_parameters", [])
        supports_schema = "structured_outputs" in model_data.get(
            "supported_parameters", []
        )

        input_modalities = model_data.get("architecture", {}).get(
            "input_modalities", []
        )
        vision = "image" in input_modalities

        shared_kwargs = dict(
            model_id=model_id,
            model_name=model_id,
            api_base=OPENROUTER_API_BASE,
            vision=vision,
            supports_tools=supports_tools,
            supports_schema=supports_schema,
        )

        try:
            sync_model = Chat(**shared_kwargs)
            async_model = AsyncChat(**shared_kwargs)
            # Override key lookup to use OPENROUTER_API_KEY
            sync_model.needs_key = "openrouter"
            sync_model.key_env_var = "OPENROUTER_API_KEY"
            async_model.needs_key = "openrouter"
            async_model.key_env_var = "OPENROUTER_API_KEY"
            register(sync_model, async_model)
        except Exception:
            pass  # Skip models that fail to register


# --- Pricing provider ---


class OpenRouterPricingProvider(PricingProvider):
    """
    Pricing provider using OpenRouter's model catalog and generation API.

    Primary strategy: look up exact cost from OpenRouter's generation endpoint.
    Fallback: calculate from cached per-token pricing.
    """

    async def supported_models(self):
        catalog = _fetch_models()
        return set(catalog.keys())

    async def calculate_cost_from_response(
        self, model_id, usage, response
    ) -> Nanocents:
        # Try exact cost from generation API first
        cost = self._cost_from_generation(response)
        if cost is not None:
            return cost

        # Fallback: calculate from cached pricing
        return self._cost_from_tokens(model_id, usage)

    def _cost_from_generation(self, response) -> Optional[Nanocents]:
        """Try to get exact cost from OpenRouter's generation endpoint."""
        if response is None:
            return None

        # The OpenRouter-compatible response includes an 'id' field
        # accessible via response.response_json
        response_json = getattr(response, "response_json", None)
        if not response_json:
            return None

        generation_id = None
        if isinstance(response_json, dict):
            generation_id = response_json.get("id")
        if not generation_id:
            return None

        gen_data = _get_generation(generation_id)
        if gen_data is None:
            return None

        total_cost_usd = gen_data.get("total_cost")
        if total_cost_usd is not None:
            return Nanocents.from_usd(total_cost_usd)

        return None

    def _cost_from_tokens(self, model_id: str, usage) -> Nanocents:
        """Calculate cost from token counts and cached pricing."""
        catalog = _fetch_models()
        if model_id not in catalog:
            raise ModelPricingNotFoundError(f"No OpenRouter pricing for '{model_id}'.")

        pricing = catalog[model_id].get("pricing", {})
        # OpenRouter prices are strings in USD per token
        prompt_price = float(pricing.get("prompt", 0))
        completion_price = float(pricing.get("completion", 0))

        input_tokens = usage.input or 0
        output_tokens = usage.output or 0

        # Prices are per-token in USD; convert to nanocents
        cost_usd = (input_tokens * prompt_price) + (output_tokens * completion_price)
        return Nanocents.from_usd(cost_usd)


# --- Usage logging accountant ---

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS openrouter_usage_log (
    id TEXT PRIMARY KEY,
    actor_id TEXT,
    model_id TEXT,
    purpose TEXT,
    nanocents INTEGER NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


async def _ensure_tables(datasette):
    db = datasette.get_internal_database()
    for statement in CREATE_TABLES_SQL.strip().split(";"):
        statement = statement.strip()
        if statement:
            await db.execute_write(statement)


class UsageLoggingAccountant(Accountant):
    """Logs all LLM usage to the internal database. No budget enforcement."""

    def __init__(self, datasette):
        self.datasette = datasette

    async def reserve(
        self,
        nanocents: Nanocents,
        model_id: Optional[str] = None,
        purpose: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> Tx:
        # Always approve — this accountant only logs, doesn't enforce
        return Tx(str(uuid.uuid4()))

    async def settle(
        self,
        tx: Tx,
        nanocents: Nanocents,
        model_id: Optional[str] = None,
        purpose: Optional[str] = None,
        actor_id: Optional[str] = None,
    ):
        await _ensure_tables(self.datasette)
        db = self.datasette.get_internal_database()
        await db.execute_write(
            "INSERT INTO openrouter_usage_log (id, actor_id, model_id, purpose, nanocents) "
            "VALUES (?, ?, ?, ?, ?)",
            [tx, actor_id, model_id, purpose, int(nanocents)],
        )


# --- Hook implementations ---


@hookimpl
def register_llm_accountants(datasette):
    return [UsageLoggingAccountant(datasette)]


@hookimpl
def register_llm_accountant_pricing(datasette):
    return OpenRouterPricingProvider()


# --- Admin UI ---


async def openrouter_admin_page(datasette, request):
    actor = request.actor
    if not actor or actor.get("id") != "root":
        return Response.text("Root access required", status=403)

    return Response.html(
        await datasette.render_template("openrouter.html", {}, request=request)
    )


async def api_openrouter_models(datasette, request):
    """Return cached model catalog with pricing."""
    catalog = _fetch_models()
    models = []
    for model_id, data in sorted(catalog.items()):
        pricing = data.get("pricing", {})
        models.append(
            {
                "id": model_id,
                "name": data.get("name", model_id),
                "context_length": data.get("context_length"),
                "prompt_price": pricing.get("prompt"),
                "completion_price": pricing.get("completion"),
            }
        )
    return Response.json({"models": models, "count": len(models)})


async def api_openrouter_usage(datasette, request):
    """Return recent usage log entries."""
    await _ensure_tables(datasette)
    db = datasette.get_internal_database()
    rows = (
        await db.execute(
            "SELECT * FROM openrouter_usage_log ORDER BY created_at DESC LIMIT 100"
        )
    ).rows
    entries = []
    for row in rows:
        entries.append(
            {
                "id": row["id"],
                "actor_id": row["actor_id"],
                "model_id": row["model_id"],
                "purpose": row["purpose"],
                "cost_usd": Nanocents(row["nanocents"]).to_usd(),
                "created_at": row["created_at"],
            }
        )
    return Response.json({"usage": entries})


async def openrouter_prompt_page(datasette, request):
    actor = request.actor
    if not actor:
        return Response.text("Login required", status=403)
    return Response.html(
        await datasette.render_template(
            "openrouter_prompt.html",
            {"actor_id": actor.get("id")},
            request=request,
        )
    )


async def api_openrouter_prompt(datasette, request):
    """Execute a prompt via datasette-llm and return the result with cost."""
    if request.method != "POST":
        return Response.json({"error": "POST required"}, status=405)

    actor = request.actor
    if not actor:
        return Response.json({"error": "Login required"}, status=401)

    body = json.loads(await request.post_body())
    model_id = str(body.get("model", "")).strip()
    prompt_text = str(body.get("prompt", "")).strip()
    if not model_id or not prompt_text:
        return Response.json({"error": "model and prompt required"}, status=400)

    from datasette_llm import LLM

    try:
        llm_client = LLM(datasette)
        model = await llm_client.model(model_id, purpose="openrouter-demo", actor=actor)
        response = await model.prompt(prompt_text)
        response_text = await response.text()
        usage = await response.usage()
        return Response.json(
            {
                "response": response_text,
                "model": model_id,
                "input_tokens": usage.input or 0,
                "output_tokens": usage.output or 0,
            }
        )
    except Exception as e:
        return Response.json({"error": str(e)}, status=500)


async def api_openrouter_available_models(datasette, request):
    """Return models available for prompting (filtered by accountant)."""
    from datasette_llm import LLM

    actor = request.actor
    llm_client = LLM(datasette)
    models = await llm_client.models(actor=actor, purpose="openrouter-demo")
    return Response.json({"models": [{"id": m.model_id} for m in models]})


@hookimpl
def register_routes():
    return [
        (r"^/-/openrouter$", openrouter_admin_page),
        (r"^/-/openrouter/prompt$", openrouter_prompt_page),
        (r"^/-/openrouter/api/models$", api_openrouter_models),
        (r"^/-/openrouter/api/available-models$", api_openrouter_available_models),
        (r"^/-/openrouter/api/usage$", api_openrouter_usage),
        (r"^/-/openrouter/api/prompt$", api_openrouter_prompt),
    ]


@hookimpl
def menu_links(datasette, actor, request):
    links = []
    if actor and actor.get("id") == "root":
        links.append(
            {
                "href": datasette.urls.path("/-/openrouter"),
                "label": "OpenRouter Admin",
            }
        )
    if actor:
        links.append(
            {
                "href": datasette.urls.path("/-/openrouter/prompt"),
                "label": "OpenRouter Prompt",
            }
        )
    return links
