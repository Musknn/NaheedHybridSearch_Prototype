"""
FastAPI Backend for Naheed Product Search
Exposes the hybrid search pipeline as REST endpoints.

Scope: pure hybrid search (BM25 + Vector -> WRRF -> confidence filter)
plus feedback/history review. NO LLM/RAG layer — generation.py, router.py,
and llm_client.py are intentionally not imported here, so this module has
zero dependency on any LLM provider (Groq/OpenAI/Gemini) or API key.
"""

import sys
from pathlib import Path
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

import feedback
import history_logger
from retrieval import SearchRequest, search
import cross_sell 

# ──────────────────────────────────────────────────────────────────────────
# Pydantic Models for API
# ──────────────────────────────────────────────────────────────────────────


class ProductResponse(BaseModel):
    id: str
    rank: int
    name: str
    brand: str
    category: str
    price: Optional[float]
    in_stock: bool
    url_key: str
    score: float
    image_url: Optional[str] = None
    rating: Optional[float] = None
    short_description: Optional[str] = None


class CrossSellSuggestionResponse(BaseModel):
    sku: str
    name: str
    brand: str
    price: Optional[float]
    in_stock: bool
    bought_percent: float
    orders: int


class CrossSellResponse(BaseModel):
    sku: str
    suggestions: List[CrossSellSuggestionResponse]

class SearchResponseAPI(BaseModel):
    query: str
    results: List[ProductResponse]
    total_candidates: int
    timings: dict[str, float]
    suggested_queries: Optional[List[str]] = None
    cross_sell: Optional[List[CrossSellSuggestionResponse]] = None


class AutoCompleteResponse(BaseModel):
    suggestions: List[str]


class FeedbackRequest(BaseModel):
    """
    A single feedback event tied to a (query, product) pair — either
    IMPLICIT (inferred from behavior) or EXPLICIT (a manual label given
    while reviewing /api/history):
      - "click"        : implicit, weak positive signal
      - "add_to_cart"  : implicit, strong positive signal
      - "relevant"     : explicit "yes this is correct" label
      - "irrelevant"   : explicit "no this is wrong" label — HARD-EXCLUDES
                         this product from this query's results going forward
    """
    query: str
    product_id: str
    event_type: str


class FeedbackResponse(BaseModel):
    status: str
    event: dict


class FeedbackHistoryResponse(BaseModel):
    count: int
    events: List[dict]


class SearchHistoryResponse(BaseModel):
    count: int
    entries: List[dict]


# ──────────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Naheed Product Search API",
    description="Hybrid search engine for Naheed product catalogue",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
        "http://localhost:8080",
        "*",  # For development only - remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _to_product_response(r, include_short_description: bool = False) -> ProductResponse:
    return ProductResponse(
        id=r.id,
        rank=r.rank,
        name=r.name,
        brand=r.brand,
        category=r.category,
        price=r.price,
        in_stock=r.in_stock,
        url_key=r.url_key,
        score=r.score,
        short_description=(f"{r.brand} {r.name}" if r.brand else r.name) if include_short_description else None,
    )


def get_suggestions(products: List[ProductResponse]) -> List[str]:
    """Generate related search suggestions from the top results."""
    suggestions = []

    categories = {p.category for p in products[:5] if p.category}
    if categories:
        suggestions.append(f"More in {next(iter(categories))}")

    brands = {p.brand for p in products[:5] if p.brand}
    if brands:
        suggestions.append(f"Shop {next(iter(brands))}")

    return suggestions[:3]


def _get_cross_sell_for_top_result(products: List[ProductResponse]) -> List[CrossSellSuggestionResponse]:
    """
    "Related to your top result" — used by /api/search only, separate
    from the dedicated /api/cross-sell endpoint (which is keyed off
    whatever SKU the frontend explicitly passes, e.g. on Add to Cart).
    Never raises — a cross-sell lookup failing must never take down a
    search request, same philosophy as the history_logger try/except
    right below this function's call site.
    """
    if not products:
        return []
    try:
        top_sku = products[0].id
        suggestions = cross_sell.get_cross_sell(top_sku, top_n=5)
        return [
            CrossSellSuggestionResponse(
                sku=s.sku,
                name=s.name,
                brand=s.brand,
                price=s.price,
                in_stock=s.in_stock,
                bought_percent=s.bought_percent,
                orders=s.orders,
            )
            for s in suggestions
        ]
    except Exception as cs_err:
        print(f"[api] cross-sell lookup failed for top result: {cs_err}")
        return []


# ──────────────────────────────────────────────────────────────────────────
# Search Endpoints
# ──────────────────────────────────────────────────────────────────────────


@app.get("/api/search", response_model=SearchResponseAPI)
async def search_products(
    q: str = Query(..., description="Search query", min_length=1),
    top_k: int = Query(20, description="Number of results", ge=1, le=50),
    in_stock: Optional[bool] = Query(None, description="Filter by stock status"),
    category: Optional[str] = Query(None, description="Filter by category"),
    brand: Optional[str] = Query(None, description="Filter by brand"),
    min_price: Optional[float] = Query(None, description="Minimum price", ge=0),
    max_price: Optional[float] = Query(None, description="Maximum price", ge=0),
):
    """Hybrid search endpoint combining BM25 + Vector search. Every pipeline
    hyperparameter (rrf_alpha, mmr_lambda, use_reranker, confidence
    threshold, etc.) is controlled solely by config.RETRIEVAL — the client
    only ever supplies the query and metadata filters, never tuning knobs."""
    try:
        request = SearchRequest(
            query=q,
            top_k=top_k,
            in_stock=in_stock,
            category=category,
            brand=brand,
            min_price=min_price,
            max_price=max_price,
        )
        response = search(request)

        products = [_to_product_response(r, include_short_description=True) for r in response.results]
        suggested = get_suggestions(products) if products else []
        cross_sell_suggestions = _get_cross_sell_for_top_result(products)

        try:
            history_logger.log_search(
                query=q,
                results=response.results,
                total_candidates=response.total_candidates,
                timings=response.timings,
                from_cache=response.from_cache,
            )
        except Exception as log_err:
            print(f"[api] failed to log search history: {log_err}")  # never fail the request over logging

        return SearchResponseAPI(
            query=q,
            results=products,
            total_candidates=response.total_candidates,
            timings=response.timings,
            suggested_queries=suggested,
            cross_sell=cross_sell_suggestions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/autocomplete", response_model=AutoCompleteResponse)
async def autocomplete(
    q: str = Query(..., description="Partial query", min_length=1),
    limit: int = Query(10, description="Max suggestions", ge=1, le=20),
):
    """Fast suggestions as the user types — deliberately skips the reranker
    (autocomplete needs low latency far more than it needs precision)."""
    try:
        request = SearchRequest(
            query=q,
            top_k=limit,
            use_reranker=False,
        )
        response = search(request)
        suggestions = [r.name for r in response.results[:limit]]

        # Popular related terms (in production, sourced from analytics).
        if q.lower() in ("diaper", "diapers", "pampers"):
            suggestions.extend(["Pampers Baby Dry", "Pampers Active Baby", "Diaper Size 4"])
        elif q.lower() in ("shampoo", "hair", "conditioner"):
            suggestions.extend(["Hair Care", "Shampoo", "Conditioner", "Hair Oil"])

        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)

        return AutoCompleteResponse(suggestions=unique_suggestions[:limit])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cross-sell", response_model=CrossSellResponse)
async def cross_sell_suggestions(
    sku: str = Query(..., description="SKU of the product just added to cart"),
    top_n: int = Query(5, description="Max suggestions to return", ge=1, le=20),
):
    # """
    # "Customers who bought {sku} also bought..." — call this right after
    # a product is added to cart. Returns an EMPTY suggestions list (not
    # an error) if the product has no qualifying co-purchase data — the
    # frontend should treat that as "nothing to show here", not a failure.
    # """
    try:
        suggestions = cross_sell.get_cross_sell(sku, top_n=top_n)
        return CrossSellResponse(
            sku=sku,
            suggestions=[
                CrossSellSuggestionResponse(
                    sku=s.sku,
                    name=s.name,
                    brand=s.brand,
                    price=s.price,
                    in_stock=s.in_stock,
                    bought_percent=s.bought_percent,
                    orders=s.orders,
                )
                for s in suggestions
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Naheed Product Search"}


# ──────────────────────────────────────────────────────────────────────────
# Feedback Endpoints (click / add-to-cart -> ranking signal + audit log)
# ──────────────────────────────────────────────────────────────────────────


@app.get("/api/history", response_model=SearchHistoryResponse)
async def search_history(
    limit: int = Query(200, description="Max search entries to return (most recent first)", ge=1, le=5000),
):
    """
    Review search history: every query, every product it returned, and
    each product's final score — regardless of whether anything was
    clicked/added to cart. Use this to find results worth labeling via
    POST /api/feedback (event_type="relevant" or "irrelevant").
    """
    try:
        entries = history_logger.load_search_history()
        recent = entries[-limit:][::-1]  # most recent first
        return SearchHistoryResponse(count=len(entries), entries=[e.model_dump() for e in recent])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(payload: FeedbackRequest):
    """
    Log a feedback event for a (query, product) pair. This both:
      1. Appends to the append-only feedback log (for manual review — see
         /api/feedback/history), and
      2. Immediately becomes available to retrieval.search() — no restart
         or cache-clear needed (feedback.py rebuilds from disk each call):
         - "click"/"add_to_cart"/"relevant" boost the product's ranking
           for this query and similar future queries.
         - "irrelevant" HARD-EXCLUDES the product from this exact query's
           results going forward, not just a rank demotion.
    """
    try:
        event = feedback.log_event(
            query=payload.query,
            product_id=payload.product_id,
            event_type=payload.event_type,
        )
        return FeedbackResponse(status="logged", event=event.model_dump())
    except feedback.InvalidEventTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feedback/history", response_model=FeedbackHistoryResponse)
async def feedback_history(
    limit: int = Query(200, description="Max events to return (most recent first)", ge=1, le=5000),
):
    """Review the full click/add-to-cart history — to check what the
    pipeline is getting right vs. wrong over time."""
    try:
        events = feedback.load_all_events()
        recent = events[-limit:][::-1]  # most recent first
        return FeedbackHistoryResponse(count=len(events), events=[e.model_dump() for e in recent])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")