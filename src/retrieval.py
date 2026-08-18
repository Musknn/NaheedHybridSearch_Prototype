"""
STEP 2: Hybrid Retrieval Pipeline
-----------------------------------
Two-stage retrieval system for the Naheed product search engine.

Stage 1 — Fast Broad Retrieval:
    BM25 (keyword) + Vector (semantic) -> Weighted Reciprocal Rank Fusion
    -> Confidence filter (applied to EVERY WRRF candidate)

Stage 2 — Precision Refinement (currently OFF by default — see config.RETRIEVAL):
    MMR (diversity, mmr_lambda=1.0 -> no-op) -> Cross-Encoder Reranking (use_reranker=False)

Current running configuration: BM25 + Vector -> WRRF -> confidence
threshold applied to all fused candidates. Stage 2 (MMR + cross-encoder)
is left fully implemented and wired in, just toggled off via
config.RETRIEVAL.use_reranker / mmr_lambda — flip those back on in
config.py any time without touching this file.

Metadata filtering (in_stock, category, brand, price range) is applied as a
pre-filter before scoring so that irrelevant products never enter the
candidate pool.

All hyperparameters (alpha, mmr_lambda, reranker toggle, confidence
threshold, cache size, etc.) live in config.RETRIEVAL. `SearchRequest`
fields default to `None`, meaning "use whatever config.RETRIEVAL says" —
callers only need to set a field explicitly when they want to override
the shared default for that one call.

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
import hashlib
import json
import math
import pickle
import time
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    from cachetools import TTLCache
except ImportError:
    raise ImportError("Please install cachetools: pip install cachetools")

from config import (
    BM25_INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    EMBED_IDS_PATH,
    EMBEDDING_MODEL_NAME,
    FEEDBACK,
    RERANKER_MODEL_NAME,
    RETRIEVAL,
    URDU_NORMALIZATION,
)

# Re-use the same tokenizer that built the BM25 index so that query
# tokenization is consistent with corpus tokenization.
from bm25_index import tokenize
from urdu_normalizer import normalize_query 
import feedback

# ═══════════════════════════════════════════════════════════════════════════
# CACHE SETUP  (size/ttl come from config.RETRIEVAL — see config.py)
# ═══════════════════════════════════════════════════════════════════════════

query_cache = TTLCache(maxsize=RETRIEVAL.cache_maxsize, ttl=RETRIEVAL.cache_ttl_seconds)

_cache_stats = {"hits": 0, "misses": 0, "total": 0}


def get_cache_stats() -> dict:
    """Return current cache hit/miss statistics."""
    return {
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "total": _cache_stats["total"],
        "cache_size": len(query_cache),
        "hit_rate": _cache_stats["hits"] / _cache_stats["total"] if _cache_stats["total"] > 0 else 0,
    }


def clear_cache() -> None:
    """Clear the entire query cache."""
    old_size = len(query_cache)
    query_cache.clear()
    print(f"[cache] cleared {old_size} items")


def get_cache_info() -> dict:
    """Get detailed cache information."""
    return {
        "size": len(query_cache),
        "maxsize": query_cache.maxsize,
        "ttl": query_cache.ttl,
        "stats": _cache_stats,
        "hit_rate": f"{_cache_stats['hits'] / _cache_stats['total'] * 100:.1f}%" if _cache_stats["total"] > 0 else "0%",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic models
# ═══════════════════════════════════════════════════════════════════════════


class SearchRequest(BaseModel):
    """
    Encapsulates everything needed to execute a hybrid search.

    Pipeline-control fields default to `None`, which means "resolve from
    config.RETRIEVAL at search time" (see `_resolve_params`). Set a field
    explicitly only when a specific call needs to deviate from the shared
    default (e.g. autocomplete wants a smaller candidate pool).
    """

    query: str = Field(..., description="The user's search query (any language)")
    top_k: Optional[int] = Field(default=None, ge=1, le=200, description="Number of final results")

    # --- Metadata filters ---
    in_stock: Optional[bool] = Field(default=None, description="Filter to in-stock products only")
    category: Optional[str] = Field(default=None)
    brand: Optional[str] = Field(default=None)
    min_price: Optional[float] = Field(default=None, ge=0)
    max_price: Optional[float] = Field(default=None, ge=0)

    # --- Pipeline controls (None = fall back to config.RETRIEVAL) ---
    bm25_candidates: Optional[int] = Field(default=None, description="Stage 1 BM25 candidate count")
    vector_candidates: Optional[int] = Field(default=None, description="Stage 1 vector candidate count")
    rrf_k: Optional[int] = Field(default=None, description="RRF smoothing constant")
    rrf_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Weight for vector results in WRRF")
    mmr_lambda: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mmr_candidates: Optional[int] = Field(default=None, description="Candidates passed to Stage 2")
    use_reranker: Optional[bool] = Field(default=None, description="Master toggle for cross-encoder reranking")
    conditional_rerank: Optional[bool] = Field(default=None, description="Skip reranker if Stage 1 is confident")
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_relevance_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confidence cutoff applied to every WRRF candidate"
    )


class _ResolvedParams(BaseModel):
    """Fully-resolved hyperparameters for a single search() call (no Nones)."""

    top_k: int
    bm25_candidates: int
    vector_candidates: int
    rrf_k: int
    rrf_alpha: float
    mmr_lambda: float
    mmr_candidates: int
    use_reranker: bool
    conditional_rerank: bool
    confidence_threshold: float
    min_relevance_score: float


def _resolve_params(request: SearchRequest) -> _ResolvedParams:
    """Fill in any `None` fields on the request from config.RETRIEVAL."""
    r = RETRIEVAL
    return _ResolvedParams(
        top_k=request.top_k if request.top_k is not None else r.default_top_k,
        bm25_candidates=request.bm25_candidates if request.bm25_candidates is not None else r.bm25_candidates,
        vector_candidates=request.vector_candidates if request.vector_candidates is not None else r.vector_candidates,
        rrf_k=request.rrf_k if request.rrf_k is not None else r.rrf_k,
        rrf_alpha=request.rrf_alpha if request.rrf_alpha is not None else r.rrf_alpha,
        mmr_lambda=request.mmr_lambda if request.mmr_lambda is not None else r.mmr_lambda,
        mmr_candidates=request.mmr_candidates if request.mmr_candidates is not None else r.mmr_candidates,
        use_reranker=request.use_reranker if request.use_reranker is not None else r.use_reranker,
        conditional_rerank=request.conditional_rerank if request.conditional_rerank is not None else r.conditional_rerank,
        confidence_threshold=request.confidence_threshold if request.confidence_threshold is not None else r.confidence_threshold,
        min_relevance_score=request.min_relevance_score if request.min_relevance_score is not None else r.min_relevance_score,
    )


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
    total_candidates: int = Field(
        description="Number of candidates after RRF fusion AND the confidence filter "
        "(before MMR/reranking, which are currently disabled by default)"
    )
    timings: dict[str, float] = Field(default_factory=dict, description="Time in seconds for each pipeline stage")
    from_cache: bool = Field(default=False, description="Whether this response came from cache")


def _generate_cache_key(request: SearchRequest, params: _ResolvedParams) -> str:
    """Deterministic cache key from the query, filters, and resolved hyperparameters."""
    key_parts = [
        request.query.strip().lower(),
        str(request.in_stock),
        str(request.category or "").strip().lower(),
        str(request.brand or "").strip().lower(),
        str(request.min_price or ""),
        str(request.max_price or ""),
        str(params.top_k),
        str(params.bm25_candidates),
        str(params.vector_candidates),
        str(params.rrf_k),
        str(params.rrf_alpha),
        str(params.mmr_lambda),
        str(params.mmr_candidates),
        str(params.use_reranker),
        str(params.conditional_rerank),
        str(params.confidence_threshold),
        str(params.min_relevance_score),
    ]
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# Lazy-loaded global resources
# ═══════════════════════════════════════════════════════════════════════════
# Loaded once on first search() call, then cached. This avoids loading
# heavy models on import.

_resources: dict = {}


def _get_resources() -> dict:
    """Lazy-load core indexes/models on first call. The cross-encoder
    reranker is NOT loaded here — see _get_reranker() below — since
    config.RETRIEVAL.use_reranker is False by default and this pipeline
    is currently BM25 + Vector -> WRRF -> confidence filter only. Loading
    a reranker model that's never actually used would waste startup time
    and an unnecessary Hugging Face Hub download/model load."""
    if _resources:
        return _resources

    print("[retrieval] loading resources (first call only) ...")
    t0 = time.perf_counter()

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    _resources["bm25_index"] = bm25_data["index"]
    _resources["bm25_ids"] = bm25_data["ids"]

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    _resources["chunks_by_id"] = {c["id"]: c for c in chunks}

    _resources["embeddings"] = np.load(EMBEDDINGS_PATH)
    with open(EMBED_IDS_PATH, encoding="utf-8") as f:
        _resources["embed_ids"] = json.load(f)
    _resources["embed_id_to_idx"] = {eid: i for i, eid in enumerate(_resources["embed_ids"])}

    _resources["embed_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"[retrieval] resources loaded in {time.perf_counter() - t0:.1f}s "
          f"(reranker NOT loaded — use_reranker={RETRIEVAL.use_reranker})")
    return _resources


_reranker = None


def _get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder reranker ONLY on first actual use
    (i.e. the first time cross_encoder_rerank() runs). With the current
    default config (use_reranker=False), this never gets called, so the
    reranker model is never downloaded/loaded at all."""
    global _reranker
    if _reranker is None:
        print("[retrieval] loading reranker (first actual use) ...")
        t0 = time.perf_counter()
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
        print(f"[retrieval] reranker loaded in {time.perf_counter() - t0:.1f}s")
    return _reranker


def get_embed_model() -> SentenceTransformer:
    """Shared accessor so other modules (evaluation.py) reuse this loaded
    instance instead of loading a second copy of the same model."""
    return _get_resources()["embed_model"]


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of strings with the shared (fine-tuned) embedding
    model. Used to pass query/product embeddings into feedback.get_boosts()
    so feedback-query similarity is computed in the same embedding space
    as product retrieval."""
    res = _get_resources()
    return res["embed_model"].encode(texts, normalize_embeddings=True, convert_to_numpy=True)


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
        if not any(cat_lower in level.lower() for level in category_path):
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


def bm25_search(query: str, top_k: int) -> list[tuple[str, float]]:
    # \"\"\"Score all documents against the query using BM25 and return top-k.

    # The query is first passed through urdu_normalizer.normalize_query(),
    # which appends the canonical English/roman-urdu term for any roman-urdu
    # word or known misspelling it recognizes (e.g. "kheema" -> also adds
    # "qeema"). BM25 has zero tolerance for spelling drift on its own, so
    # this is the retrieval path that benefits most from expansion — see
    # config.URDU_NORMALIZATION.enable_for_bm25 to toggle it off.
    # \"\"\"
    res = _get_resources()

    if URDU_NORMALIZATION.enable_for_bm25:
        norm_result = normalize_query(query)
        query_tokens = tokenize(norm_result.expanded_query)
    else:
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


def vector_search(query: str, top_k: int) -> list[tuple[str, float]]:
    # \"\"\"Encode the query and find nearest neighbors by cosine similarity.

    # Expansion here is OPTIONAL and independently toggleable from BM25
    # (config.URDU_NORMALIZATION.enable_for_vector) — the fine-tuned
    # embedding model already saw ~6,100 roman-urdu/misspelling anchors
    # during training (grouped_training_data_v2.json), so appending the
    # canonical term is a smaller, more uncertain lift here than on the
    # BM25 side. Keep this on by default but re-verify with the eval set
    # (Phase 1 of the rollout) before assuming it helps rather than
    # diluting an already-robust embedding.
    # \"\"\"
    res = _get_resources()

    if URDU_NORMALIZATION.enable_for_vector:
        norm_result = normalize_query(query)
        embed_text = norm_result.expanded_query
    else:
        embed_text = query

    q_emb = res["embed_model"].encode([embed_text], normalize_embeddings=True, convert_to_numpy=True)
    scores = (res["embeddings"] @ q_emb.T).squeeze()

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(res["embed_ids"][i], float(scores[i])) for i in top_indices]


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Weighted Reciprocal Rank Fusion (WRRF)
# ═══════════════════════════════════════════════════════════════════════════


def weighted_reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int,
    alpha: float,
) -> list[tuple[str, float]]:
    """
    Merge vector and BM25 results using Weighted Reciprocal Rank Fusion.

    WRRF_score(d) = alpha * (1/(k+rank_vector(d))) + (1-alpha) * (1/(k+rank_bm25(d)))

    Args:
        vector_results: (id, score) tuples from dense search, sorted descending.
        bm25_results: (id, score) tuples from keyword search, sorted descending.
        k: Smoothing constant to penalize low-ranked documents.
        alpha: Weight assigned to vector results; (1 - alpha) goes to BM25.
    """
    wrrf_scores: dict[str, float] = {}

    def _add_to_fusion(ranked_list: list[tuple[str, float]], weight: float):
        for rank, (doc_id, _raw_score) in enumerate(ranked_list, start=1):
            wrrf_scores[doc_id] = wrrf_scores.get(doc_id, 0.0) + weight * (1.0 / (k + rank))

    _add_to_fusion(vector_results, weight=alpha)
    _add_to_fusion(bm25_results, weight=(1.0 - alpha))

    return sorted(wrrf_scores.items(), key=lambda x: x[1], reverse=True)


def _filter_candidates_by_confidence(
    fused: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    threshold: float,
    exempt_ids: frozenset[str] | set[str] = frozenset(),
) -> list[tuple[str, float]]:
    """
    Confidence filter applied to EVERY candidate coming out of WRRF fusion —
    not just the #1 result. Each candidate's confidence is its vector
    cosine similarity to the query (0.0 if BM25-only, i.e. no semantic
    support at all). Anything below `threshold` is dropped before it ever
    reaches metadata filtering, MMR, or reranking.

    `exempt_ids` (typically products with positive feedback for this query)
    bypass the check entirely — a human already verified relevance, so a
    middling vector score shouldn't override that signal.

    This is the primary relevance mechanism in the current configuration
    (BM25 + Vector -> WRRF -> confidence threshold on all candidates),
    since MMR/reranker are currently disabled by default in config.RETRIEVAL.
    """
    vector_score_map = dict(vector_results)
    return [
        (doc_id, score) for doc_id, score in fused
        if doc_id in exempt_ids or vector_score_map.get(doc_id, 0.0) >= threshold
    ]


def _check_stage1_confidence(
    fused_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    k: int,
    threshold: float,
) -> tuple[bool, str]:
    """
    Evaluates whether Stage 1 retrieval produced a high-confidence match,
    so the (expensive) cross-encoder can be safely skipped. Only relevant
    when config.RETRIEVAL.use_reranker=True and conditional_rerank=True.

    Returns:
        (is_high_confidence, reason)
    """
    if not fused_results:
        return False, "no_candidates"

    # Signal 1: Top-1 agreement between BM25 and vector search.
    top_bm25_id = bm25_results[0][0] if bm25_results else None
    top_vector_id = vector_results[0][0] if vector_results else None
    if top_bm25_id and top_vector_id and top_bm25_id == top_vector_id:
        return True, f"top1_agreement (SKU: {top_bm25_id})"

    # Signal 2: proximity to the maximum theoretical WRRF score.
    max_possible_score = 1.0 / (k + 1)
    top_fused_score = fused_results[0][1]
    score_ratio = top_fused_score / max_possible_score

    if score_ratio >= threshold:
        return True, f"high_score_ratio ({score_ratio:.2%} >= {threshold:.0%})"

    return False, f"low_confidence (ratio={score_ratio:.2%})"


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2a: Maximal Marginal Relevance (MMR) — currently disabled by default
# (config.RETRIEVAL.mmr_lambda = 1.0). Fully implemented below so it can be
# turned back on any time by lowering mmr_lambda in config.py.
# ═══════════════════════════════════════════════════════════════════════════


def mmr_rerank(
    query: str,
    candidate_ids: list[str],
    top_k: int,
    lambda_param: float,
) -> list[str]:
    """
    Apply Maximal Marginal Relevance to reduce redundancy.

    MMR = lambda * Sim(d, query) - (1-lambda) * max{ Sim(d, d') for d' in selected }

    Args:
        query: The user's search query.
        candidate_ids: Chunk IDs to consider (from RRF output).
        top_k: Number of diverse documents to select.
        lambda_param: Balance parameter. 1.0 = pure relevance, 0.0 = pure diversity.

    Returns:
        List of selected chunk IDs in MMR order.
    """
    res = _get_resources()
    id_to_idx = res["embed_id_to_idx"]
    embeddings = res["embeddings"]

    q_emb = res["embed_model"].encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).squeeze()

    valid_ids = [cid for cid in candidate_ids if cid in id_to_idx]
    if not valid_ids:
        return []

    candidate_indices = [id_to_idx[cid] for cid in valid_ids]
    candidate_embs = embeddings[candidate_indices]

    relevance_scores = candidate_embs @ q_emb

    selected_indices: list[int] = []
    remaining = set(range(len(valid_ids)))
    top_k = min(top_k, len(valid_ids))

    for _ in range(top_k):
        best_idx = -1
        best_mmr = -float("inf")

        for idx in remaining:
            rel = relevance_scores[idx]

            if selected_indices:
                selected_embs = candidate_embs[selected_indices]
                sims_to_selected = selected_embs @ candidate_embs[idx]
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
# Stage 2b: Cross-Encoder Reranking — currently disabled by default
# (config.RETRIEVAL.use_reranker = False). Fully implemented below so it
# can be turned back on any time via config.py.
# ═══════════════════════════════════════════════════════════════════════════


def cross_encoder_rerank(query: str, candidate_ids: list[str], top_k: int) -> list[tuple[str, float]]:
    """
    Re-score candidates using the cross-encoder (BAAI/bge-reranker-v2-m3).

    Unlike bi-encoders, the cross-encoder jointly processes the (query, doc)
    pair, giving much more accurate relevance scores but at higher latency
    (~50-100ms/pair on CPU) — hence only run on a small candidate set.

    Raw BGE-reranker scores are unbounded logits, so we pass them through a
    sigmoid to get an interpretable relevance probability in [0, 1]. This is
    a monotonic transform (doesn't change ranking) but makes the resulting
    score directly comparable to config.RETRIEVAL.min_relevance_score.
    """
    res = _get_resources()
    reranker = _get_reranker()
    chunks_by_id = res["chunks_by_id"]

    pairs = []
    valid_ids = []
    for cid in candidate_ids:
        chunk = chunks_by_id.get(cid)
        if chunk:
            pairs.append((query, chunk["text"]))
            valid_ids.append(cid)

    if not pairs:
        return []

    raw_scores = reranker.predict(pairs)
    relevance_scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]

    scored = list(zip(valid_ids, relevance_scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _top_match_confidence(
    final_ids_scores: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    reranker_was_run: bool,
) -> float:
    """
    Confidence score for the single best surviving match — a final sanity
    check after Stage 2 (if it ran). Note: the primary relevance filtering
    now happens earlier in search(), via _filter_candidates_by_confidence()
    applied to the FULL WRRF candidate pool. This check just re-confirms
    the top result using the reranker's score when Stage 2 actually ran
    (a more precise signal than vector similarity alone).

    - If the cross-encoder ran, its top score is already a sigmoid-normalized
      relevance probability in [0, 1] — use it directly.
    - Otherwise (reranker disabled, or conditionally skipped), fall back to
      the top vector cosine similarity — which, in the current default
      configuration, was already checked once for every candidate upstream,
      so this is a no-op re-confirmation.
    """
    if not final_ids_scores:
        return 0.0
    if reranker_was_run:
        return final_ids_scores[0][1]
    return vector_results[0][1] if vector_results else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Main search orchestrator (with caching)
# ═══════════════════════════════════════════════════════════════════════════


def search(request: SearchRequest) -> SearchResponse:
    """
    Execute the full hybrid search pipeline with caching.

    Pipeline (current default configuration — MMR/reranker OFF):
        1. Resolve hyperparameters (request overrides > config.RETRIEVAL)
        2. Check cache
        3. On miss: BM25 + Vector -> WRRF fusion -> confidence filter
           (applied to every fused candidate) -> metadata filter
           -> [MMR -> cross-encoder rerank, if turned on in config.RETRIEVAL]
        4. Cache and return

    MMR and the cross-encoder reranker are fully implemented and wired in
    (Stage 2 below) but are no-ops in the current default config
    (mmr_lambda=1.0, use_reranker=False) — flip those in config.py to turn
    them back on without touching this function.
    """
    params = _resolve_params(request)
    cache_key = _generate_cache_key(request, params)

    if cache_key in query_cache:
        _cache_stats["hits"] += 1
        _cache_stats["total"] += 1
        cached_response = query_cache[cache_key]
        cached_response.from_cache = True
        return cached_response

    _cache_stats["misses"] += 1
    _cache_stats["total"] += 1

    res = _get_resources()
    chunks_by_id = res["chunks_by_id"]
    timings: dict[str, float] = {}

    # ── Stage 1a: BM25 ──
    t0 = time.perf_counter()
    bm25_results = bm25_search(request.query, top_k=params.bm25_candidates)
    timings["bm25"] = time.perf_counter() - t0

    norm_debug = normalize_query(request.query)
    timings_extra_urdu_hits = [h.matched_text + "->" + h.canonical for h in norm_debug.hits]
    # ── Stage 1b: Vector ──
    t0 = time.perf_counter()
    vector_results = vector_search(request.query, top_k=params.vector_candidates)
    timings["vector"] = time.perf_counter() - t0

    # ── Stage 1c: WRRF fusion ──
    t0 = time.perf_counter()
    fused = weighted_reciprocal_rank_fusion(
        vector_results, bm25_results, k=params.rrf_k, alpha=params.rrf_alpha
    )
    timings["rrf"] = time.perf_counter() - t0

    fused = fused[:RETRIEVAL.fusion_top_n]

    # ── Feedback boost ──
    # Every click/add-to-cart/relevant/irrelevant event for this (or a
    # similar) query adjusts the product's fused score — positive events
    # push it up, "irrelevant" pushes it down. A product explicitly marked
    # "irrelevant" for this query is HARD-EXCLUDED outright, not just
    # demoted — giving negative feedback should reliably stop a wrong
    # product from showing up, not just nudge its rank. A product with
    # strong positive feedback that WRRF/BM25/vector missed entirely is
    # injected as a new candidate, so add-to-cart/relevant reliably
    # resurfaces the right product on a similar future query.
    t0 = time.perf_counter()
    boosts: dict[str, float] = {}
    if FEEDBACK.enabled:
        boosts = feedback.get_boosts(request.query, embed_fn=_embed_texts)
    positive_boosted_ids = {pid for pid, score in boosts.items() if score > 0}
    excluded_ids = {pid for pid, score in boosts.items() if score < 0}

    if boosts:
        fused_map = dict(fused)
        for product_id, boost_score in boosts.items():
            if product_id not in chunks_by_id:
                continue  # stale feedback for a product no longer in the catalogue
            fused_map[product_id] = fused_map.get(product_id, 0.0) + boost_score * FEEDBACK.boost_weight
        fused = [(doc_id, score) for doc_id, score in fused_map.items() if doc_id not in excluded_ids]
        fused.sort(key=lambda x: x[1], reverse=True)
    timings["feedback_boost"] = time.perf_counter() - t0

    # ── Confidence filter: applied to EVERY WRRF candidate ──
    # This is the current pipeline's primary relevance mechanism:
    # BM25 + Vector -> WRRF -> confidence threshold on ALL candidates.
    # Every fused candidate is checked here — not just whichever one ends
    # up ranked #1 — so a query with a couple of strong matches and a pile
    # of weak, semantically-unrelated stragglers doesn't return the
    # stragglers just because top_k padding needs something to show.
    # Products with POSITIVE feedback are exempt: a human already verified
    # relevance, so a middling vector score shouldn't override that.
    t0 = time.perf_counter()
    fused = _filter_candidates_by_confidence(
        fused, vector_results, params.min_relevance_score, exempt_ids=positive_boosted_ids
    )
    timings["confidence_filter"] = time.perf_counter() - t0

    fused_score_map = dict(fused)
    total_candidates = len(fused)

    # ── Metadata filtering ──
    t0 = time.perf_counter()
    filtered_ids = [
        doc_id
        for doc_id, _score in fused
        if doc_id in chunks_by_id and _passes_filters(chunks_by_id[doc_id], request)
    ]
    timings["filtering"] = time.perf_counter() - t0

    # ── Stage 2a: MMR diversity (no-op while mmr_lambda >= 1.0) ──
    t0 = time.perf_counter()
    if params.mmr_lambda >= 1.0:
        # Pure relevance -> MMR would be a no-op, skip the O(n^2) computation.
        mmr_ids = filtered_ids[: params.mmr_candidates]
    else:
        mmr_ids = mmr_rerank(
            query=request.query,
            candidate_ids=filtered_ids,
            top_k=params.mmr_candidates,
            lambda_param=params.mmr_lambda,
        )
    timings["mmr"] = time.perf_counter() - t0

    # ── Stage 2b: Cross-encoder reranking (no-op while use_reranker=False) ──
    t0 = time.perf_counter()
    skipped_reranker = False
    skip_reason = ""
    reranker_was_run = False

    if params.use_reranker and mmr_ids:
        if params.conditional_rerank:
            is_confident, skip_reason = _check_stage1_confidence(
                fused_results=fused,
                bm25_results=bm25_results,
                vector_results=vector_results,
                k=params.rrf_k,
                threshold=params.confidence_threshold,
            )
            skipped_reranker = is_confident

        if skipped_reranker:
            final_ids_scores = [(cid, fused_score_map.get(cid, 0.0)) for cid in mmr_ids[: params.top_k]]
            print(f"[rerank] SKIPPED cross-encoder ({skip_reason})")
        else:
            reranked = cross_encoder_rerank(
                query=request.query,
                candidate_ids=mmr_ids,
                top_k=params.top_k,
            )
            final_ids_scores = reranked
            reranker_was_run = True
            print(f"[rerank] EXECUTED cross-encoder ({skip_reason or 'reranker enabled'})")
    else:
        final_ids_scores = [(cid, fused_score_map.get(cid, 0.0)) for cid in mmr_ids[: params.top_k]]

    timings["reranker"] = time.perf_counter() - t0

    # ── Final confidence re-check on the #1 result ──
    # In the current default config (reranker off), everything reaching
    # this point already passed the full-pool confidence filter above, so
    # this is a no-op re-confirmation. It becomes meaningful again the
    # moment use_reranker=True, since it then checks the reranker's more
    # precise score instead. Positively-boosted products remain exempt,
    # consistent with the earlier filter.
    t0 = time.perf_counter()
    top_id = final_ids_scores[0][0] if final_ids_scores else None
    top_confidence = _top_match_confidence(final_ids_scores, vector_results, reranker_was_run)
    top_passes = top_id is not None and (top_id in positive_boosted_ids or top_confidence >= params.min_relevance_score)
    if not top_passes:
        print(
            f"[relevance-gate] top confidence {top_confidence:.3f} < "
            f"min_relevance_score {params.min_relevance_score} -> no results"
        )
        final_ids_scores = []
    timings["relevance_gate"] = time.perf_counter() - t0

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

    response = SearchResponse(
        query=request.query,
        results=results,
        total_candidates=total_candidates,
        timings={k: round(v, 4) for k, v in timings.items()},
        from_cache=False,
    )

    query_cache[cache_key] = response
    return response


def merge_search_responses(responses: list[SearchResponse], combined_query: str) -> SearchResponse:
    """Combine multiple per-ingredient SearchResponses into one, re-ranked by rank position."""
    all_results = []
    next_rank = 1
    for resp in responses:
        for r in resp.results:
            all_results.append(r.model_copy(update={"rank": next_rank}))
            next_rank += 1

    merged_timings: dict[str, float] = {}
    for resp in responses:
        for stage, t in resp.timings.items():
            merged_timings[stage] = merged_timings.get(stage, 0.0) + t

    return SearchResponse(
        query=combined_query,
        results=all_results,
        total_candidates=sum(r.total_candidates for r in responses),
        timings={k: round(v, 4) for k, v in merged_timings.items()},
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retrieval search for Naheed product catalogue")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=None, help="Number of results")
    parser.add_argument("--in-stock", action="store_true", help="Only in-stock products")
    parser.add_argument("--category", type=str, default=None, help="Category filter")
    parser.add_argument("--brand", type=str, default=None, help="Brand filter")
    parser.add_argument("--min-price", type=float, default=None, help="Minimum price")
    parser.add_argument("--max-price", type=float, default=None, help="Maximum price")
    parser.add_argument("--min-relevance", type=float, default=None, help="Confidence cutoff applied to all WRRF candidates")
    parser.add_argument("--mmr-lambda", type=float, default=None, help="MMR lambda (0=diversity, 1=relevance)")
    parser.add_argument("--use-rerank", action="store_true", help="Turn ON cross-encoder reranking (off by default)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the query cache before running")
    parser.add_argument("--show-cache", action="store_true", help="Show cache statistics")
    args = parser.parse_args()

    if args.clear_cache:
        clear_cache()
        return

    if args.show_cache:
        info = get_cache_info()
        print("Cache Information:")
        print(f"  Size: {info['size']}/{info['maxsize']}")
        print(f"  TTL: {info['ttl']} seconds")
        print(f"  Hits: {info['stats']['hits']}")
        print(f"  Misses: {info['stats']['misses']}")
        print(f"  Hit Rate: {info['hit_rate']}")
        return

    request = SearchRequest(
        query=args.query,
        top_k=args.top_k,
        in_stock=True if args.in_stock else None,
        category=args.category,
        brand=args.brand,
        min_price=args.min_price,
        max_price=args.max_price,
        min_relevance_score=args.min_relevance,
        mmr_lambda=args.mmr_lambda,
        use_reranker=(True if args.use_rerank else None),
    )

    print(f"Searching for: \"{request.query}\"")
    response = search(request)

    print(f"Found {response.total_candidates} candidates after WRRF fusion + confidence filter")
    print(f"Returning top {len(response.results)} results  (from_cache={response.from_cache})\n")

    for r in response.results:
        stock_str = "in stock" if r.in_stock else "out of stock"
        price_str = f"Rs.{r.price:,.0f}" if r.price else "N/A"
        print(f"  {r.rank}. [{r.id}] {r.name}")
        print(f"     Brand: {r.brand} | Category: {r.category} | {price_str} | {stock_str}")
        print(f"     Score: {r.score:.4f}\n")

    print("Timings:")
    for stage, elapsed in response.timings.items():
        print(f"  {stage:>18s}: {elapsed:.4f}s")
    print(f"  {'TOTAL':>18s}: {sum(response.timings.values()):.4f}s")

    info = get_cache_info()
    print(f"\nCache Stats: {info['size']}/{info['maxsize']} items, Hit Rate: {info['hit_rate']}")


if __name__ == "__main__":
    main()