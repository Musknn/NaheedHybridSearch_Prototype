"""
Search History Logger
-----------------------
Logs every search: the query, every product returned, and each product's
final score — regardless of whether the user ever clicked or added
anything to cart. This is what lets you go back and answer "for this
query, what did we actually return, and was it right?"

This is deliberately separate from feedback.py's log:
  - `history_logger.py` (this file) — an unconditional record of what the
    pipeline returned. Written automatically on every search.
  - `feedback.py` — a record of user/reviewer *judgments* about those
    results (click, add-to-cart, or an explicit relevant/irrelevant label
    given while reviewing this history). Written only when feedback happens.

Typical review workflow:
    1. `GET /api/history` to see what a query returned and each item's score.
    2. Decide which of those items were actually correct/incorrect.
    3. `POST /api/feedback` with event_type="relevant" or "irrelevant" for
       the specific (query, product_id) pairs you just judged — see
       feedback.py. That feedback then boosts/suppresses those products the
       next time the same or a similar query comes in.
"""
from __future__ import annotations

import json
import os
import time

from pydantic import BaseModel

from config import SEARCH_HISTORY_PATH


class HistoryResultItem(BaseModel):
    """One product as it appeared in a logged search result set."""

    id: str
    rank: int
    score: float
    name: str
    brand: str
    category: str
    price: float | None
    in_stock: bool


class SearchHistoryEntry(BaseModel):
    """A single logged search: the query and everything returned for it."""

    query: str
    timestamp: float
    total_candidates: int
    results: list[HistoryResultItem]
    timings: dict[str, float]
    from_cache: bool = False


def log_search(
    query: str,
    results: list,
    total_candidates: int,
    timings: dict[str, float],
    from_cache: bool = False,
) -> SearchHistoryEntry:
    """
    Append one search event to the history log. `results` accepts
    retrieval.SearchResult objects (or anything with the same fields) —
    duck-typed via .id/.rank/.score/etc. rather than importing
    retrieval.py, to keep this module dependency-free and reusable from
    api.py or a CLI/batch script equally.
    """
    entry = SearchHistoryEntry(
        query=query,
        timestamp=time.time(),
        total_candidates=total_candidates,
        results=[
            HistoryResultItem(
                id=r.id,
                rank=r.rank,
                score=r.score,
                name=r.name,
                brand=r.brand,
                category=r.category,
                price=r.price,
                in_stock=r.in_stock,
            )
            for r in results
        ],
        timings=timings,
        from_cache=from_cache,
    )
    with open(SEARCH_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")
    return entry


def load_search_history() -> list[SearchHistoryEntry]:
    """Load the full search history log."""
    if not os.path.exists(SEARCH_HISTORY_PATH):
        return []
    entries = []
    with open(SEARCH_HISTORY_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(SearchHistoryEntry(**json.loads(line)))
    return entries