"""
STEP 2: Hybrid Retrieval Pipeline
-----------------------------------
Two-stage retrieval system for the Naheed product search engine.

Stage 1 — Fast Broad Retrieval:
    BM25 (keyword) + Vector (semantic) → Reciprocal Rank Fusion

Stage 2 — Precision Refinement:
    MMR (diversity) → Cross-Encoder Reranking (relevance)

Metadata filtering (in_stock, category, brand, price range) is applied as a
pre-filter before scoring so that irrelevant products never enter the
candidate pool.

Usage as a module:
    from retrieval import search, SearchRequest
    request = SearchRequest(query="pampers diapers", top_k=5, in_stock=True)
    response = search(request)
    for r in response.results:
        print(r.id, r.name, r.score)

Usage as a CLI:
    python retrieval.py "pampers diapers" --top-k 5 --in-stock
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import field
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (
    BM25_INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    EMBED_IDS_PATH,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
)

# Re-use the same tokenizer that built the BM25 index so that query
# tokenization is consistent with corpus tokenization.
from bm25_index import tokenize

# ═══════════════════════════════════════════════════════════════════════════
# Pydantic models
# ═══════════════════════════════════════════════════════════════════════════


class SearchRequest(BaseModel):
    """Encapsulates everything needed to execute a hybrid search."""

    query: str = Field(..., description="The user's search query (any language)")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of final results to return")

    # --- Metadata filters (all optional) ---
    in_stock: Optional[bool] = Field(default=None, description="Filter to in-stock products only")
    category: Optional[str] = Field(default=None, description="Filter by category (matches any level in category_path)")
    brand: Optional[str] = Field(default=None, description="Filter by brand name (case-insensitive)")
    min_price: Optional[float] = Field(default=None, ge=0, description="Minimum price filter")
    max_price: Optional[float] = Field(default=None, ge=0, description="Maximum price filter")

    # --- Pipeline controls ---
    bm25_candidates: int = Field(default=60, description="Number of BM25 candidates for Stage 1")
    vector_candidates: int = Field(default=60, description="Number of vector candidates for Stage 1")
    rrf_k: int = Field(default=40, description="RRF constant k (standard=60)")
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0, description="MMR λ: 1.0=pure relevance, 0.0=pure diversity")
    mmr_candidates: int = Field(default=30, description="Number of candidates after MMR filtering")
    use_reranker: bool = Field(default=True, description="Whether to apply cross-encoder reranking (slower but more precise)")


class SearchResult(BaseModel):
    """A single search result."""

    id: str
    rank: int
    score: float
    name: str
    brand: str
    category: str
    price: Optional[float]
    in_stock: bool
    url_key: str


class SearchResponse(BaseModel):
    """The full search response including timing metadata."""

    query: str
    results: list[SearchResult]
    total_candidates: int = Field(description="Number of candidates after RRF fusion (before MMR/reranking)")
    timings: dict[str, float] = Field(default_factory=dict, description="Time in seconds for each pipeline stage")


# ═══════════════════════════════════════════════════════════════════════════
# Lazy-loaded global resources
# ═══════════════════════════════════════════════════════════════════════════
# Loaded once on first search() call, then cached. This avoids loading
# heavy models on import.

_resources: dict = {}


def _get_resources() -> dict:
    """Lazy-load all indexes and models on first call."""
    if _resources:
        return _resources

    print("Loading retrieval resources (first call only) ...")
    t0 = time.perf_counter()

    # --- BM25 index ---
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    _resources["bm25_index"] = bm25_data["index"]
    _resources["bm25_ids"] = bm25_data["ids"]

    # --- Chunks (full data for metadata filtering & display) ---
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    # Build a lookup dict: id → chunk
    _resources["chunks_by_id"] = {c["id"]: c for c in chunks}

    # --- Dense embeddings ---
    _resources["embeddings"] = np.load(EMBEDDINGS_PATH)
    with open(EMBED_IDS_PATH, encoding="utf-8") as f:
        _resources["embed_ids"] = json.load(f)
    # Build a lookup dict: id → row index in the embedding matrix
    _resources["embed_id_to_idx"] = {
        eid: i for i, eid in enumerate(_resources["embed_ids"])
    }

    # --- Embedding model (for encoding queries) ---
    _resources["embed_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # --- Cross-encoder reranker (loaded lazily on first rerank call) ---
    _resources["reranker"] = None  # loaded on demand

    print(f"  Resources loaded in {time.perf_counter() - t0:.1f}s")
    return _resources


def _get_reranker() -> CrossEncoder:
    """Load the cross-encoder reranker on first use."""
    res = _get_resources()
    if res["reranker"] is None:
        print(f"Loading reranker: {RERANKER_MODEL_NAME} ...")
        res["reranker"] = CrossEncoder(RERANKER_MODEL_NAME)
        print("  Reranker loaded.")
    return res["reranker"]


# ═══════════════════════════════════════════════════════════════════════════
# Metadata filtering
# ═══════════════════════════════════════════════════════════════════════════


def _passes_filters(chunk: dict, request: SearchRequest) -> bool:
    """Check if a chunk passes all metadata filters in the request."""
    meta = chunk["metadata"]

    if request.in_stock is not None and meta.get("in_stock") != request.in_stock:
        return False

    if request.category is not None:
        cat_lower = request.category.lower()
        category_path = meta.get("category_path", [])
        # Match against any level in the category hierarchy
        if not any(cat_lower in level.lower() for level in category_path):
            # Also check the top-level category field
            if cat_lower not in meta.get("category", "").lower():
                return False

    if request.brand is not None:
        if request.brand.lower() not in meta.get("brand", "").lower():
            return False

    price = meta.get("price")
    if request.min_price is not None and (price is None or price < request.min_price):
        return False
    if request.max_price is not None and (price is None or price > request.max_price):
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: BM25 Search
# ═══════════════════════════════════════════════════════════════════════════


def bm25_search(
    query: str,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """
    Score all documents against the query using BM25 and return top-k.

    Returns:
        List of (chunk_id, bm25_score) tuples, sorted by score descending.
    """
    res = _get_resources()
    query_tokens = tokenize(query)
    scores = res["bm25_index"].get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        (res["bm25_ids"][i], float(scores[i]))
        for i in top_indices
        if scores[i] > 0  # skip zero-score documents
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Vector Search
# ═══════════════════════════════════════════════════════════════════════════


def vector_search(
    query: str,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """
    Encode the query and find nearest neighbors by cosine similarity.

    Returns:
        List of (chunk_id, cosine_score) tuples, sorted by score descending.
    """
    res = _get_resources()
    q_emb = res["embed_model"].encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )
    # Embeddings are L2-normalized, so dot product = cosine similarity
    scores = res["embeddings"] @ q_emb.T
    scores = scores.squeeze()

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        (res["embed_ids"][i], float(scores[i]))
        for i in top_indices
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Reciprocal Rank Fusion (RRF)
# ═══════════════════════════════════════════════════════════════════════════


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF_score(d) = Σ  1 / (k + rank_i(d))

    where rank_i(d) is the 1-based rank of document d in ranked list i.
    Documents not present in a list receive no contribution from that list
    (equivalent to rank = infinity).

    Args:
        *ranked_lists: Variable number of ranked lists, each a list of
                       (id, score) tuples sorted by score descending.
        k: RRF constant (standard = 60). Higher k reduces the influence
           of high-ranked documents.

    Returns:
        List of (id, rrf_score) tuples sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Sort by RRF score descending
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused

def weighted_score_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    alpha: float = 0.4,
) -> list[tuple[str, float]]:
    """
    Merge vector and BM25 results via a normalized weighted sum:

        final_score = alpha * norm_vector_score + (1 - alpha) * norm_bm25_score

    Each score list is independently min-max normalized to [0, 1] first —
    raw BM25 scores are unbounded while cosine scores are already roughly
    bounded, so normalizing puts them on a comparable scale before applying
    alpha. Documents missing from one list score 0 on that side.
    """
    def _normalize(results: list[tuple[str, float]]) -> dict[str, float]:
        if not results:
            return {}
        scores = [s for _, s in results]
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return {doc_id: 1.0 for doc_id, _ in results}
        return {doc_id: (s - lo) / (hi - lo) for doc_id, s in results}

    norm_vector = _normalize(vector_results)
    norm_bm25 = _normalize(bm25_results)

    all_ids = set(norm_vector) | set(norm_bm25)
    fused = {
        doc_id: alpha * norm_vector.get(doc_id, 0.0) + (1 - alpha) * norm_bm25.get(doc_id, 0.0)
        for doc_id in all_ids
    }

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Maximal Marginal Relevance (MMR)
# ═══════════════════════════════════════════════════════════════════════════


def mmr_rerank(
    query: str,
    candidate_ids: list[str],
    top_k: int = 50,
    lambda_param: float = 0.5,
) -> list[str]:
    """
    Apply Maximal Marginal Relevance to reduce redundancy.

    MMR = λ · Sim(d, query) − (1−λ) · max{ Sim(d, d') for d' in selected }

    At each iteration, the document with the highest MMR score is added
    to the selected set. This balances relevance (Sim to query) with
    diversity (dissimilarity to already-selected documents).

    Uses the pre-computed embeddings for similarity calculations.

    Args:
        query: The user's search query.
        candidate_ids: List of chunk IDs to consider (from RRF output).
        top_k: Number of diverse documents to select.
        lambda_param: Balance parameter. 1.0 = pure relevance, 0.0 = pure diversity.

    Returns:
        List of selected chunk IDs in MMR-order.
    """
    res = _get_resources()
    id_to_idx = res["embed_id_to_idx"]
    embeddings = res["embeddings"]

    # Encode query
    q_emb = res["embed_model"].encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).squeeze()  # shape: (dim,)

    # Filter to candidates that exist in our embedding index
    valid_ids = [cid for cid in candidate_ids if cid in id_to_idx]
    if not valid_ids:
        return []

    # Get embedding matrix for candidates only
    candidate_indices = [id_to_idx[cid] for cid in valid_ids]
    candidate_embs = embeddings[candidate_indices]  # shape: (n_candidates, dim)

    # Relevance scores: cosine similarity to query (already normalized)
    relevance_scores = candidate_embs @ q_emb  # shape: (n_candidates,)

    # Greedy MMR selection
    selected_indices: list[int] = []  # indices into valid_ids/candidate_embs
    remaining = set(range(len(valid_ids)))
    top_k = min(top_k, len(valid_ids))

    for _ in range(top_k):
        best_idx = -1
        best_mmr = -float("inf")

        for idx in remaining:
            # Relevance component
            rel = relevance_scores[idx]

            # Diversity component: max similarity to any already-selected doc
            if selected_indices:
                selected_embs = candidate_embs[selected_indices]  # (n_selected, dim)
                sims_to_selected = selected_embs @ candidate_embs[idx]  # (n_selected,)
                max_sim = float(np.max(sims_to_selected))
            else:
                max_sim = 0.0

            mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    return [valid_ids[i] for i in selected_indices]


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Cross-Encoder Reranking
# ═══════════════════════════════════════════════════════════════════════════


def cross_encoder_rerank(
    query: str,
    candidate_ids: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Re-score candidates using the cross-encoder (BAAI/bge-reranker-v2-m3).

    Unlike bi-encoders, the cross-encoder jointly processes the (query, doc)
    pair through all transformer layers, enabling cross-attention between
    query and document tokens. This gives much more accurate relevance
    scores but is too slow for the full corpus (~50-100ms per pair on CPU).

    Args:
        query: The user's search query.
        candidate_ids: Chunk IDs to re-score (typically 20-50 from MMR).
        top_k: Number of top results to return after reranking.

    Returns:
        List of (chunk_id, cross_encoder_score) sorted by score descending.
    """
    res = _get_resources()
    reranker = _get_reranker()
    chunks_by_id = res["chunks_by_id"]

    # Build (query, document_text) pairs
    pairs = []
    valid_ids = []
    for cid in candidate_ids:
        chunk = chunks_by_id.get(cid)
        if chunk:
            pairs.append((query, chunk["text"]))
            valid_ids.append(cid)

    if not pairs:
        return []

    # Score all pairs in a single batch
    scores = reranker.predict(pairs)

    # Sort by cross-encoder score descending
    scored = list(zip(valid_ids, [float(s) for s in scores]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# Main search orchestrator
# ═══════════════════════════════════════════════════════════════════════════


def search(request: SearchRequest) -> SearchResponse:
    """
    Execute the full two-stage hybrid search pipeline.

    Pipeline:
        1. BM25 candidates (top N)
        2. Vector candidates (top N)
        3. RRF fusion
        4. Metadata filtering
        5. MMR diversity filtering
        6. Cross-encoder reranking (optional)
        7. Return top_k results

    Args:
        request: A SearchRequest with query, filters, and pipeline controls.

    Returns:
        SearchResponse with ranked results and timing metadata.
    """
    res = _get_resources()
    chunks_by_id = res["chunks_by_id"]
    timings: dict[str, float] = {}

    # ── Stage 1a: BM25 Search ──
    t0 = time.perf_counter()
    bm25_results = bm25_search(request.query, top_k=request.bm25_candidates)
    timings["bm25"] = time.perf_counter() - t0

    # ── Stage 1b: Vector Search ──
    t0 = time.perf_counter()
    vector_results = vector_search(request.query, top_k=request.vector_candidates)
    timings["vector"] = time.perf_counter() - t0

    # ── Stage 1c: RRF Fusion ──
    t0 = time.perf_counter()
    fused = weighted_score_fusion(vector_results, bm25_results, alpha=0.4)
    timings["rrf"] = time.perf_counter() - t0
    fused = fused[:100]          # <-- add this line
    total_candidates = len(fused)

    # ── Metadata Filtering ──
    t0 = time.perf_counter()
    filtered_ids = [
        doc_id
        for doc_id, _score in fused
        if doc_id in chunks_by_id and _passes_filters(chunks_by_id[doc_id], request)
    ]
    timings["filtering"] = time.perf_counter() - t0

    # ── Stage 2a: MMR Diversity ──
    t0 = time.perf_counter()
    mmr_ids = mmr_rerank(
        query=request.query,
        candidate_ids=filtered_ids,
        top_k=request.mmr_candidates,
        lambda_param=request.mmr_lambda,
    )
    timings["mmr"] = time.perf_counter() - t0

    # ── Stage 2b: Cross-Encoder Reranking (optional) ──
    if request.use_reranker and mmr_ids:
        t0 = time.perf_counter()
        reranked = cross_encoder_rerank(
            query=request.query,
            candidate_ids=mmr_ids,
            top_k=request.top_k,
        )
        timings["reranker"] = time.perf_counter() - t0
        final_ids_scores = reranked
    else:
        # Without reranking, just take top_k from MMR order
        final_ids_scores = [(cid, 0.0) for cid in mmr_ids[: request.top_k]]

    # ── Build response ──
    results = []
    for rank, (doc_id, score) in enumerate(final_ids_scores, start=1):
        chunk = chunks_by_id[doc_id]
        meta = chunk["metadata"]
        results.append(
            SearchResult(
                id=doc_id,
                rank=rank,
                score=round(score, 4),
                name=meta.get("name", ""),
                brand=meta.get("brand", ""),
                category=meta.get("category", ""),
                price=meta.get("price"),
                in_stock=meta.get("in_stock", False),
                url_key=meta.get("url_key", ""),
            )
        )

    return SearchResponse(
        query=request.query,
        results=results,
        total_candidates=total_candidates,
        timings={k: round(v, 4) for k, v in timings.items()},
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval search for Naheed product catalogue"
    )
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--in-stock", action="store_true", help="Only in-stock products")
    parser.add_argument("--category", type=str, default=None, help="Category filter")
    parser.add_argument("--brand", type=str, default=None, help="Brand filter")
    parser.add_argument("--min-price", type=float, default=None, help="Minimum price")
    parser.add_argument("--max-price", type=float, default=None, help="Maximum price")
    parser.add_argument("--mmr-lambda", type=float, default=0.5, help="MMR λ (0=diversity, 1=relevance)")
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking")
    args = parser.parse_args()

    request = SearchRequest(
        query=args.query,
        top_k=args.top_k,
        in_stock=True if args.in_stock else None,
        category=args.category,
        brand=args.brand,
        min_price=args.min_price,
        max_price=args.max_price,
        mmr_lambda=args.mmr_lambda,
        use_reranker=not args.no_rerank,
    )

    print(f"Searching for: \"{request.query}\"")
    print(f"  Filters: in_stock={request.in_stock}, category={request.category}, "
          f"brand={request.brand}, price=[{request.min_price}, {request.max_price}]")
    print(f"  MMR λ={request.mmr_lambda}, reranker={'ON' if request.use_reranker else 'OFF'}")
    print()

    response = search(request)

    print(f"Found {response.total_candidates} candidates after RRF fusion")
    print(f"Returning top {len(response.results)} results:\n")

    for r in response.results:
        stock_str = "✓ in stock" if r.in_stock else "✗ out of stock"
        price_str = f"Rs.{r.price:,.0f}" if r.price else "N/A"
        print(f"  {r.rank}. [{r.id}] {r.name}")
        print(f"     Brand: {r.brand} | Category: {r.category} | {price_str} | {stock_str}")
        print(f"     Score: {r.score:.4f}")
        print()

    print("Timings:")
    for stage, elapsed in response.timings.items():
        print(f"  {stage:>12s}: {elapsed:.4f}s")
    total = sum(response.timings.values())
    print(f"  {'TOTAL':>12s}: {total:.4f}s")


if __name__ == "__main__":
    main()
