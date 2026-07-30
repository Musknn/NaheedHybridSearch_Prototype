"""
Query Router — Intent classification + structured extraction, via
llm_client.call_llm(). Model names and generation hyperparameters come
from config.ROUTER_*_MODEL_NAME / config.ROUTING — nothing hardcoded here.
"""
import json

from pydantic import BaseModel

from config import ROUTER_EXTRACTION_MODEL_NAME, ROUTER_INTENT_MODEL_NAME, ROUTING
from llm_client import call_llm


class RecipeExtraction(BaseModel):
    dish_name: str
    ingredients: list[str]


class PriceFilter(BaseModel):
    item_name: str
    max_price: float | None = None
    min_price: float | None = None


class ExtractionError(Exception):
    """Raised when structured extraction fails after the model call succeeds
    but the response can't be parsed/validated. Callers (generation.py)
    should catch this and fall back to standard_search."""


def _safe_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned.removeprefix("json").strip()
    return json.loads(cleaned)


ROUTER_PROMPT = """Classify the user's grocery/pharmacy chatbot query into exactly one category:

- "recipe_builder": user wants ingredients/shopping list for a dish (e.g. "recipe of biryani", "chicken karahi banane ke liye kya chahiye", "sabzi kaise banayein")
- "price_filter": user wants items under/above a price (e.g. "atta under 500", "500 se kam wala chawal", "diapers below 1000")
- "standard_search": anything else — direct product lookup, general queries

Respond ONLY with JSON: {"intent": "<category>"}
No explanation, no markdown fences."""

VALID_INTENTS = {"recipe_builder", "price_filter", "standard_search"}


def classify_intent(query: str) -> str:
    """
    Classify intent. Falls back to "standard_search" on ANY failure —
    a bad/deprecated model string, a network error, malformed JSON, or an
    unrecognized intent value — so routing failures never crash the pipeline;
    they just degrade to the original, always-working search path.
    """
    try:
        raw = call_llm(
            prompt=f"{ROUTER_PROMPT}\n\nQuery: {query}",
            model_name=ROUTER_INTENT_MODEL_NAME,
            max_new_tokens=ROUTING.intent_max_new_tokens,
            temperature=ROUTING.intent_temperature,
        )
    except Exception as e:
        print(f"[router] classify_intent call_llm failed ({e}), falling back to standard_search")
        return "standard_search"

    try:
        intent = _safe_json(raw)["intent"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"[router] classify_intent parse failed ({e}), raw={raw!r}, falling back to standard_search")
        return "standard_search"

    if intent not in VALID_INTENTS:
        print(f"[router] classify_intent returned unknown intent {intent!r}, falling back to standard_search")
        return "standard_search"

    return intent


def extract_recipe(query: str) -> RecipeExtraction:
    """
    Extract dish name + ingredients. Raises ExtractionError on any failure
    (model call failure, malformed JSON, schema validation failure) so the
    caller (generation.py) can catch it and fall back to standard_search.
    """
    schema = RecipeExtraction.model_json_schema()
    prompt = f"""Extract the dish name and the full list of ingredients needed to cook it.
Use Roman Urdu spellings as they'd appear in a Pakistani grocery catalog
(e.g. "chawal", "murghi", "biryani masala", "dahi") where applicable.
Respond ONLY with JSON matching this schema exactly:
{json.dumps(schema)}

Query: {query}"""

    try:
        raw = call_llm(
            prompt=prompt,
            model_name=ROUTER_EXTRACTION_MODEL_NAME,
            max_new_tokens=ROUTING.extraction_max_new_tokens,
            temperature=ROUTING.extraction_temperature,
        )
    except Exception as e:
        raise ExtractionError(f"extract_recipe call_llm failed: {e}") from e

    try:
        return RecipeExtraction(**_safe_json(raw))
    except Exception as e:
        raise ExtractionError(f"extract_recipe parse/validation failed: {e}, raw={raw!r}") from e


def extract_price_filter(query: str) -> PriceFilter:
    """
    Extract item name + price bounds. Raises ExtractionError on any failure
    so the caller (generation.py) can catch it and fall back to standard_search.
    """
    schema = PriceFilter.model_json_schema()
    prompt = f"""Extract the item/category being searched for and any price constraint in PKR.
Respond ONLY with JSON matching this schema exactly:
{json.dumps(schema)}

Query: {query}"""

    try:
        raw = call_llm(
            prompt=prompt,
            model_name=ROUTER_EXTRACTION_MODEL_NAME,
            max_new_tokens=150,
            temperature=ROUTING.extraction_temperature,
        )
    except Exception as e:
        raise ExtractionError(f"extract_price_filter call_llm failed: {e}") from e

    try:
        return PriceFilter(**_safe_json(raw))
    except Exception as e:
        raise ExtractionError(f"extract_price_filter parse/validation failed: {e}, raw={raw!r}") from e
    