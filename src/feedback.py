"""
Feedback — click/add-to-cart/relevant/irrelevant signals as relevance labels
---------------------------------------------------------------------------
Two kinds of feedback feed the same mechanism:
  - IMPLICIT: "click" / "add_to_cart" — inferred from user behavior.
  - EXPLICIT: "relevant" / "irrelevant" — a manual label given while
    reviewing search history (see history_logger.py / GET /api/history).

Every event is a relevance signal for a (query, product) pair — via
embedding similarity, it also applies to SIMILAR future queries, not just
an exact repeat of the same query.

This module has two responsibilities, kept separate on purpose:

  1. `log_event()` / `load_all_events()` — an append-only JSONL history of
     every event, for manual review ("what did we get right vs. wrong")
     and as the single source of truth everything else is rebuilt from.

  2. `get_boosts()` — computes a {product_id: boost_score} map for a given
     query by combining exact-query matches with similarity-weighted
     matches against past feedback queries. Positive scores (click,
     add_to_cart, relevant) boost a product in retrieval.py's WRRF fused
     scores; negative scores (irrelevant) suppress it and, in retrieval.py,
     trigger a hard exclusion for the exact query rather than just a
     rank demotion.

All feedback hyperparameters (event weights, similarity threshold, boost
strength) live in config.FEEDBACK — nothing hardcoded here.

Design note: this rebuilds its aggregate from the on-disk log on every
call rather than maintaining a live cache. At prototype scale (a few
thousand feedback events over 2130 products) that's cheap and means a
click/add-to-cart/label is reflected in the very next search with zero
staleness or cache-invalidation logic. If the log grows large, swap this
for an incrementally-updated cache — the public functions below wouldn't
need to change.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable

import numpy as np
from pydantic import BaseModel

from config import FEEDBACK, FEEDBACK_LOG_PATH

VALID_EVENT_TYPES = frozenset(FEEDBACK.event_weights.keys())

EmbedFn = Callable[[list[str]], np.ndarray]


class FeedbackEvent(BaseModel):
    """A single feedback event (click/add_to_cart/relevant/irrelevant), as
    recorded to the history log."""

    query: str
    normalized_query: str
    product_id: str
    event_type: str
    weight: float
    timestamp: float


class InvalidEventTypeError(ValueError):
    pass


def _normalize(query: str) -> str:
    """Lowercase + collapse whitespace, so 'Pampers  Diapers' and 'pampers diapers'
    aggregate as the same feedback query."""
    return " ".join(query.strip().lower().split())


# ═══════════════════════════════════════════════════════════════════════════
# 1. Append-only history log
# ═══════════════════════════════════════════════════════════════════════════


def log_event(query: str, product_id: str, event_type: str) -> FeedbackEvent:
    """Record a feedback event (click/add_to_cart/relevant/irrelevant) to
    the history log. Raises InvalidEventTypeError for an unrecognized
    event_type rather than silently accepting typos that would never
    contribute a boost."""
    if event_type not in VALID_EVENT_TYPES:
        raise InvalidEventTypeError(
            f"event_type must be one of {sorted(VALID_EVENT_TYPES)}, got {event_type!r}"
        )

    event = FeedbackEvent(
        query=query,
        normalized_query=_normalize(query),
        product_id=product_id,
        event_type=event_type,
        weight=FEEDBACK.event_weights[event_type],
        timestamp=time.time(),
    )
    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(event.model_dump_json() + "\n")
    return event


def load_all_events() -> list[FeedbackEvent]:
    """Load the full feedback history — used both for boost computation
    and for manual review/analysis (e.g. via /api/feedback/history)."""
    if not os.path.exists(FEEDBACK_LOG_PATH):
        return []
    events = []
    with open(FEEDBACK_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(FeedbackEvent(**json.loads(line)))
    return events


# ═══════════════════════════════════════════════════════════════════════════
# 2. Boost computation
# ═══════════════════════════════════════════════════════════════════════════


def _aggregate_by_query(events: list[FeedbackEvent]) -> dict[str, dict[str, float]]:
    """normalized_query -> {product_id: summed_weight}."""
    agg: dict[str, dict[str, float]] = {}
    for e in events:
        agg.setdefault(e.normalized_query, {})
        agg[e.normalized_query][e.product_id] = agg[e.normalized_query].get(e.product_id, 0.0) + e.weight
    return agg


def get_boosts(query: str, embed_fn: EmbedFn) -> dict[str, float]:
    """
    Returns {product_id: boost_score} for `query`, combining:
      - exact normalized-query matches (full weight), and
      - similar historical feedback queries, weighted by cosine similarity
        (only counted above config.FEEDBACK.similarity_threshold).

    `embed_fn` must take a list of strings and return an (N, dim) array of
    normalized embeddings — pass retrieval.py's shared embedding model in,
    so query and feedback-query embeddings live in the same fine-tuned
    space as product embeddings.
    """
    if not FEEDBACK.enabled:
        return {}

    events = load_all_events()
    if not events:
        return {}

    agg = _aggregate_by_query(events)
    normalized = _normalize(query)
    boosts: dict[str, float] = {}

    # Exact match — always included, no embedding call needed.
    if normalized in agg:
        for pid, score in agg[normalized].items():
            boosts[pid] = boosts.get(pid, 0.0) + score

    historical_queries = list(agg.keys())
    if not historical_queries:
        return boosts

    hist_embeddings = embed_fn(historical_queries)
    q_emb = embed_fn([query]).squeeze()
    similarities = hist_embeddings @ q_emb

    for idx, sim in enumerate(similarities):
        if sim < FEEDBACK.similarity_threshold:
            continue
        hist_query = historical_queries[idx]
        if hist_query == normalized:
            continue  # already counted at full weight above
        for pid, score in agg[hist_query].items():
            boosts[pid] = boosts.get(pid, 0.0) + score * float(sim)

    return boosts