"""
FastAPI Backend for Naheed Product Search
Exposes the hybrid search pipeline as REST endpoints.
"""

import sys
import os
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from retrieval import SearchRequest, search
from generation import generate  # ← NEW: Import generation

# ──────────────────────────────────────────────────────────────────────────
# Pydantic Models for API
# ──────────────────────────────────────────────────────────────────────────

class ProductResponse(BaseModel):
    """Single product in search results."""
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
    """API response wrapper."""
    query: str
    results: List[ProductResponse]
    total_candidates: int
    timings: dict[str, float]
    suggested_queries: Optional[List[str]] = None


class AutoCompleteResponse(BaseModel):
    """Autocomplete suggestions."""
    suggestions: List[str]


# ──────────────────────────────────────────────────────────────────────────
# NEW: Chat Response Model
# ──────────────────────────────────────────────────────────────────────────

class ChatResponseAPI(BaseModel):
    """Response for chat/QA endpoint."""
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
    version="1.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5500",  # VS Code Live Server
        "http://localhost:8080",
        "*",  # For development only - remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    use_reranker: bool = Query(True, description="Use cross-encoder reranking"),
):
    """
    Hybrid search endpoint combining BM25 + Vector search.
    
    Returns top-k products with metadata and ranking scores.
    """
    try:
        request = SearchRequest(
            query=q,
            top_k=top_k,
            in_stock=in_stock,
            category=category,
            brand=brand,
            min_price=min_price,
            max_price=max_price,
            use_reranker=use_reranker,
            bm25_candidates=100,
            vector_candidates=100,
            mmr_candidates=30,
        )
        
        response = search(request)
        
        # Convert to API response format
        products = []
        for r in response.results:
            products.append(ProductResponse(
                id=r.id,
                rank=r.rank,
                name=r.name,
                brand=r.brand,
                category=r.category,
                price=r.price,
                in_stock=r.in_stock,
                url_key=r.url_key,
                score=r.score,
                short_description=f"{r.brand} {r.name}" if r.brand else r.name,
            ))
        
        # Generate suggested queries
        suggested = get_suggestions(q, products) if products else []
        
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
    """
    Autocomplete suggestions as user types.
    """
    try:
        # Quick BM25 search with small top_k for suggestions
        request = SearchRequest(
            query=q,
            top_k=limit,
            use_reranker=False,
            bm25_candidates=50,
            vector_candidates=50,
        )
        response = search(request)
        
        # Extract product names as suggestions
        suggestions = [r.name for r in response.results[:limit]]
        
        # Add popular related terms (in production, from analytics)
        if q.lower() in ["diaper", "diapers", "pampers"]:
            suggestions.extend(["Pampers Baby Dry", "Pampers Active Baby", "Diaper Size 4"])
        elif q.lower() in ["shampoo", "hair", "conditioner"]:
            suggestions.extend(["Hair Care", "Shampoo", "Conditioner", "Hair Oil"])
        
        # Deduplicate and limit
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)
        
        return AutoCompleteResponse(suggestions=unique_suggestions[:limit])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────
# NEW: Chat/QA Endpoint
# ──────────────────────────────────────────────────────────────────────────

@app.get("/api/chat", response_model=ChatResponseAPI)
async def chat_query(
    query: str = Query(..., description="User question"),
    top_k: int = Query(5, description="Number of products to ground the answer"),
):
    """
    RAG-powered chat endpoint.
    Accepts natural language questions and returns grounded answers.
    """
    try:
        # Call the generation pipeline
        result = generate(query, top_k=top_k)
        
        # Convert retrieved products to ProductResponse
        products = []
        for r in result.retrieved.results:
            products.append(ProductResponse(
                id=r.id,
                rank=r.rank,
                name=r.name,
                brand=r.brand,
                category=r.category,
                price=r.price,
                in_stock=r.in_stock,
                url_key=r.url_key,
                score=r.score,
            ))
        
        return ChatResponseAPI(
            query=query,
            answer=result.response,
            products=products,
            timings=result.timings
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────

def get_suggestions(query: str, products: List[ProductResponse]) -> List[str]:
    """Generate related search suggestions."""
    suggestions = []
    
    # Category-based suggestions
    categories = set()
    for p in products[:5]:
        if p.category:
            categories.add(p.category)
    
    if categories:
        suggestions.append(f"More in {list(categories)[0]}")
    
    # Brand-based suggestions
    brands = set()
    for p in products[:5]:
        if p.brand:
            brands.add(p.brand)
    
    if brands:
        suggestions.append(f"Shop {list(brands)[0]}")
    
    return suggestions[:3]


# ──────────────────────────────────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Naheed Product Search"}


# ──────────────────────────────────────────────────────────────────────────
# Run Server
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )