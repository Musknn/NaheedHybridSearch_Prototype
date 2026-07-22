"""
STEP 2b: RAG Generation
--------------------------
Takes the top-ranked products from retrieval.py's hybrid search pipeline
and uses an LLM (Tiny Aya — see llm_client.py) to generate a natural-
language answer grounded in that retrieved context.

Pipeline position (per Unit 10's "Complete RAG Pipeline with Evaluation" slide):

    USER QUERY
      -> retrieval.search()   [BM25 + Semantic -> RRF -> MMR -> Cross-Encoder rerank]
      -> TOP-K PRODUCTS
      -> build_context()      (this module)
      -> LLM ("LLM1", this module)
      -> RAG RESPONSE
      -> hand off `result.response` + `result.context` to evaluation.py
         for faithfulness/relevancy scoring

Design note on context:
    The context passed to the LLM (and later to evaluation.evaluate_faithfulness)
    is built only from the structured fields retrieval.py already returns
    (name, brand, category, price, in_stock) rather than the raw chunk text.
    Every one of those fields is independently checkable, which is exactly
    what faithfulness scoring needs: a claim like "it costs Rs. 450 and is in
    stock" can be verified word-for-word against this context, rather than
    against a longer, harder-to-check paragraph of raw catalogue text.

Usage as a module:
    from generation import generate
    result = generate("pampers diapers for a newborn")
    print(result.response)
    print(result.context)   # feed this into evaluation.evaluate_faithfulness()

Usage as a CLI:
    python generation.py "is panadol in stock"
    python generation.py "sons ka pampers" --top-k 3 --in-stock
"""
from __future__ import annotations

import argparse
import time

from pydantic import BaseModel, Field

from llm_client import call_llm
from retrieval import SearchRequest, SearchResponse, SearchResult, search

# ═══════════════════════════════════════════════════════════════════════════
# Prompt template
# ═══════════════════════════════════════════════════════════════════════════
# Explicitly instructed to answer ONLY from the retrieved context and to
# admit when it doesn't have the answer, rather than guessing — this is the
# core RAG motivation from Unit 8 ("LLMs hallucinate when missing context")
# and exactly what evaluation.py's faithfulness metric checks for afterward.

GENERATION_PROMPT = """You are a helpful shopping assistant for Naheed, a Pakistani pharmacy \
and supermarket chain. Answer the customer's question using ONLY the product \
information listed below. Do not mention any product that is not listed. If the \
listed products don't contain the answer, say so honestly instead of guessing.

Retrieved products:
{context}

Customer question: {query}

Answer:"""


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic model for the result
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
    """
    Format retrieved products into a numbered context block for the LLM.

    Each line includes exactly the fields a customer could ask about, so
    every claim the LLM might generate is checkable against this same text.
    """
    if not results:
        return "No matching products were found in the catalogue."

    lines = []
    for r in results:
        stock = "in stock" if r.in_stock else "out of stock"
        price = f"Rs. {r.price:,.0f}" if r.price is not None else "price unavailable"
        lines.append(
            f"{r.rank}. {r.name} — Brand: {r.brand} | Category: {r.category} | "
            f"{price} | {stock}"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main generation function
# ═══════════════════════════════════════════════════════════════════════════


def generate(
    query: str,
    top_k: int = 5,
    max_new_tokens: int = 256,
    **search_kwargs,
) -> GenerationResult:
    """
    Run the full retrieval -> generation pipeline for a single query.

    Args:
        query: The customer's question, in English or Roman Urdu.
        top_k: Number of retrieved products to ground the answer in
               (Unit 10's pipeline diagram uses the top 5 after reranking).
        max_new_tokens: Generation length cap passed to the LLM.
        **search_kwargs: Passed through to SearchRequest — e.g. in_stock,
                         category, brand, min_price, max_price, mmr_lambda,
                         use_reranker. Do not pass `query` or `top_k` here;
                         use the named parameters above instead.

    Returns:
        GenerationResult with the answer, the context it was grounded in,
        the full retrieval response, and per-stage timings.
    """
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    request = SearchRequest(query=query, top_k=top_k, **search_kwargs)
    retrieved = search(request)
    timings["retrieval"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    context = build_context(retrieved.results)
    prompt = GENERATION_PROMPT.format(context=context, query=query)
    response = call_llm(prompt, max_new_tokens=max_new_tokens, temperature=0.3)
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
    parser = argparse.ArgumentParser(
        description="RAG answer generation for the Naheed product search chatbot"
    )
    parser.add_argument("query", type=str, help="Customer question (English or Roman Urdu)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved products to ground the answer in")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens the LLM may generate")
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
        use_reranker=not args.no_rerank,
    )

    print(f"Query: {result.query}\n")
    print("Retrieved context:")
    print(result.context)
    print()
    print("Answer:")
    print(result.response)
    print()
    print("Timings:")
    for stage, elapsed in result.timings.items():
        print(f"  {stage:>10s}: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
