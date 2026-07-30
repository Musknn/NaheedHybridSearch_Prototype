"""
STEP 2b: RAG Generation
--------------------------
Takes the top-ranked products from retrieval.py's hybrid search pipeline
and uses an LLM (see llm_client.py) to generate a natural-language answer
grounded in that retrieved context.

Pipeline position:

    USER QUERY
      -> router.classify_intent()
      -> retrieval.search()   [BM25 + Vector -> WRRF -> MMR -> Cross-Encoder rerank]
      -> TOP-K PRODUCTS
      -> build_context()      (this module)
      -> llm_client.call_llm()
      -> RAG RESPONSE

Design note on context:
    The context passed to the LLM is built only from the structured
    fields retrieval.py already returns (name, brand, category, price,
    in_stock) rather than raw chunk text — every field is independently
    checkable, which is what faithfulness scoring (evaluation.py) needs.

Usage as a module:
    from generation import generate
    result = generate("pampers diapers for a newborn")
    print(result.response)

Usage as a CLI:
    python generation.py "is panadol in stock"
    python generation.py "sons ka pampers" --top-k 3 --in-stock
"""
from __future__ import annotations

import argparse
import time

from pydantic import BaseModel, Field

from config import GENERATION
from llm_client import call_llm
from retrieval import SearchRequest, SearchResponse, SearchResult, merge_search_responses, search
from router import ExtractionError, classify_intent, extract_price_filter, extract_recipe

# ═══════════════════════════════════════════════════════════════════════════
# Prompt templates
# ═══════════════════════════════════════════════════════════════════════════

GENERATION_PROMPT = """You are a helpful shopping assistant for Naheed, a Pakistani pharmacy 
and supermarket chain. Answer the customer's question using ONLY the product 
information listed below.

CRITICAL INSTRUCTIONS:
1. Bilingual Mapping: The user may ask in Roman Urdu (e.g., "larkon ke glasses", "lal mirch"). You MUST mentally map these terms to the English product names, categories, or bilingual labels found in the retrieved products (e.g., map "larkon" to "Men's", or "lal mirch" to "Red Chilli").
2. Ignore Brand Clutter: Look for the core item being requested, even if it is surrounded by brand names, weights, or packaging sizes (e.g., find "Ginger" within "Fresh Basket Ginger 250g").

If a product matches the user's intent, confidently recommend it using the details provided. 

Do not mention any product that is not listed below. If the listed products 
don't contain the answer, say so honestly instead of guessing.

Retrieved products:
{context}

Customer question: {query}

Answer:"""


RECIPE_PROMPT = """You are a helpful shopping assistant for Naheed, a Pakistani pharmacy 
and supermarket chain. The customer wants to cook {dish_name} and needs a shopping list.

Below are the retrieved products for the requested ingredients. 

CRITICAL MATCHING INSTRUCTIONS:
1. Exhaustive Search: Carefully scan the ENTIRE product list for each required ingredient. 
2. Ignore Brand Clutter: Match the core ingredient name even if it is hidden behind brand names, weights, or packaging types (e.g., match 'Ginger' or 'Adrak' to 'Fresh Basket Ginger (Adrak), 250g').
3. Bilingual Mapping: Products may be listed in English, Roman Urdu, or both. You MUST check for both translations (e.g., check for both 'Mustard Oil' and 'Sarson Ka Tel', or 'Garlic' and 'Lehsan') before declaring an item missing.
4. Do Not Skip: Do not skip any requested ingredients. If a highly relevant product is in the list, you must output it.

If an ingredient truly has no match in the context after applying the rules above, say so honestly instead of guessing a substitute.

Retrieved products (grouped by ingredient, ranked by relevance within each):
{context}

Customer question: {query}

Write a friendly shopping list: one line per ingredient with the matched product 
and price, then a total estimated cost at the end. If an ingredient wasn't found, 
mention it clearly so the customer knows to source it elsewhere.

Answer:"""

# ═══════════════════════════════════════════════════════════════════════════
# Result model
# ═══════════════════════════════════════════════════════════════════════════


class GenerationResult(BaseModel):
    """The full output of a single generate() call."""

    query: str
    response: str = Field(description="The LLM's generated answer")
    context: str = Field(
        description="The retrieved-product context the LLM was grounded in — "
        "pass this directly to evaluation.evaluate_faithfulness()"
    )
    retrieved: SearchResponse = Field(description="Full retrieval response, for inspection/debugging")
    timings: dict[str, float] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Context building
# ═══════════════════════════════════════════════════════════════════════════


def build_context(results: list[SearchResult]) -> str:
    """Format retrieved products into a numbered context block for the LLM."""
    if not results:
        return "No matching products were found in the catalogue."

    lines = []
    for r in results:
        stock = "in stock" if r.in_stock else "out of stock"
        price = f"Rs. {r.price:,.0f}" if r.price is not None else "price unavailable"
        lines.append(
            f"{r.rank}. {r.name} — Brand: {r.brand} | Category: {r.category} | {price} | {stock}"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Intent-specific retrieval
# ═══════════════════════════════════════════════════════════════════════════


def _retrieve_standard_search(query: str, top_k: int, **search_kwargs) -> SearchResponse:
    request = SearchRequest(query=query, top_k=top_k, **search_kwargs)
    return search(request)


def _retrieve_price_filter(query: str, top_k: int, **search_kwargs) -> tuple[SearchResponse, str | None]:
    """Returns (response, dish_name=None) — dish_name kept for a uniform tuple shape upstream."""
    try:
        extraction = extract_price_filter(query)
        request = SearchRequest(
            query=extraction.item_name,
            top_k=top_k,
            min_price=extraction.min_price,
            max_price=extraction.max_price,
            **search_kwargs,
        )
        return search(request), None
    except ExtractionError as e:
        print(f"[generate] price_filter extraction failed ({e}), falling back to standard_search")
        return _retrieve_standard_search(query, top_k, **search_kwargs), None


def _retrieve_recipe(query: str, **search_kwargs) -> tuple[SearchResponse, str]:
    extraction = extract_recipe(query)
    per_ingredient_responses = [
        search(SearchRequest(query=ingredient, top_k=3, **search_kwargs))
        for ingredient in extraction.ingredients
    ]
    merged = merge_search_responses(per_ingredient_responses, combined_query=query)
    return merged, extraction.dish_name


# ═══════════════════════════════════════════════════════════════════════════
# Main generation function
# ═══════════════════════════════════════════════════════════════════════════


def generate(
    query: str,
    top_k: int | None = None,
    max_new_tokens: int = GENERATION.max_new_tokens,
    temperature: float = GENERATION.temperature,
    **search_kwargs,
) -> GenerationResult:
    """
    Run the full RAG pipeline: classify intent -> retrieve -> build context
    -> generate. `top_k` defaults to None, letting retrieval.py resolve it
    from config.RETRIEVAL.default_top_k, same as a direct search() call.
    """
    timings: dict[str, float] = {}
    dish_name: str | None = None

    t0 = time.perf_counter()
    intent = classify_intent(query)
    timings["routing"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if intent == "price_filter":
        retrieved, dish_name = _retrieve_price_filter(query, top_k, **search_kwargs)
    elif intent == "recipe_builder":
        retrieved, dish_name = _retrieve_recipe(query, **search_kwargs)
    else:
        retrieved = _retrieve_standard_search(query, top_k, **search_kwargs)
    timings["retrieval"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    context = build_context(retrieved.results)

    if intent == "recipe_builder":
        prompt = RECIPE_PROMPT.format(dish_name=dish_name, context=context, query=query)
    else:
        prompt = GENERATION_PROMPT.format(context=context, query=query)

    response = call_llm(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    timings["generation"] = time.perf_counter() - t0

    return GenerationResult(
        query=query,
        response=response,
        context=context,
        retrieved=retrieved,
        timings={k: round(v, 4) for k, v in timings.items()},
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG answer generation for the Naheed product search chatbot")
    parser.add_argument("query", type=str, help="Customer question (English or Roman Urdu)")
    parser.add_argument("--top-k", type=int, default=None, help="Number of retrieved products to ground the answer in")
    parser.add_argument("--max-new-tokens", type=int, default=GENERATION.max_new_tokens)
    parser.add_argument("--in-stock", action="store_true", help="Only consider in-stock products")
    parser.add_argument("--category", type=str, default=None, help="Category filter")
    parser.add_argument("--brand", type=str, default=None, help="Brand filter")
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking (faster, less precise)")
    args = parser.parse_args()

    result = generate(
        query=args.query,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        in_stock=True if args.in_stock else None,
        category=args.category,
        brand=args.brand,
        use_reranker=(False if args.no_rerank else None),
    )

    print(f"Query: {result.query}\n")
    print("Retrieved context:")
    print(result.context)
    print("\nAnswer:")
    print(result.response)
    print("\nTimings:")
    for stage, elapsed in result.timings.items():
        print(f"  {stage:>10s}: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

    