"""
FastAPI Backend for Naheed Product Search
Exposes the hybrid search pipeline as REST endpoints.
"""

import os

# Prevent Windows OS Error 1455 (paging file too small) when the embedding
# model loads via safetensors mmap — same workaround app.py uses, needed
# here too since this is the entrypoint the frontend actually talks to.
os.environ["SAFETENSORS_FAST_DISABLE"] = "1"

import sys
from pathlib import Path
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from generation import generate
from retrieval import SearchRequest, search

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


class SearchResponseAPI(BaseModel):
    query: str
    results: List[ProductResponse]
    total_candidates: int
    timings: dict[str, float]
    suggested_queries: Optional[List[str]] = None


class AutoCompleteResponse(BaseModel):
    suggestions: List[str]


class ChatResponseAPI(BaseModel):
    query: str
    answer: str
    products: List[ProductResponse]
    timings: dict[str, float]


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

        return SearchResponseAPI(
            query=q,
            results=products,
            total_candidates=response.total_candidates,
            timings=response.timings,
            suggested_queries=suggested,
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


@app.get("/api/chat", response_model=ChatResponseAPI)
async def chat_query(
    query: str = Query(..., description="User question"),
    top_k: int = Query(5, description="Number of products to ground the answer"),
):
    """RAG-powered chat endpoint. Accepts natural language questions and returns grounded answers."""
    try:
        result = generate(query, top_k=top_k)
        products = [_to_product_response(r) for r in result.retrieved.results]

        return ChatResponseAPI(
            query=query,
            answer=result.response,
            products=products,
            timings=result.timings,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Naheed Product Search"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")