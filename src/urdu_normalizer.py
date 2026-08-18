"""
Roman-Urdu / Misspelling Query Normalizer
-------------------------------------------
Query-time preprocessing layer that sits BEFORE both BM25 and vector
search. It does not touch the index or the corpus — it only expands the
incoming query string by appending the canonical English/roman-urdu term
whenever it recognizes a roman-urdu word or a known misspelling of one.

Why append instead of replace:
  If the dictionary/fuzzy match is ever wrong, the user's original tokens
  are still in the query, so both BM25 and the embedding model still have
  the raw text to work with. Expansion can only add recall, never remove it.

Two matching passes:
  1. EXACT   — dictionary lookup built from urdu_lookup_layer.csv (see
               build_lookup.py / build_lookup_round2.py). Multi-word
               phrases ("kabli chana", "gol gappay") are matched before
               single words so they aren't double-counted.
  2. FUZZY   — rapidfuzz fallback for tokens NOT covered by the exact
               dictionary (e.g. a novel misspelling like "nehariii").
               Only fires on individual tokens (not phrases) above a
               minimum length, to keep the false-positive rate low and
               the cost bounded (~400 dictionary keys, this is cheap).

This module is also imported by query_log_miner.py (offline job) to
re-check historical queries against the CURRENT dictionary, so "did this
query have a dictionary hit" is always evaluated the same way live and
offline.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz, process

from config import ROMAN_URDU_LOOKUP_PATH, URDU_NORMALIZATION

# ═══════════════════════════════════════════════════════════════════════════
# Dictionary loading
# ═══════════════════════════════════════════════════════════════════════════

_WORD_RE = re.compile(r"[^\w\s]")


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = _WORD_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text)


@dataclass(frozen=True)
class UrduDictionary:
    term_to_canonical: dict[str, str]
    # keys sorted by word-count descending, then by length descending —
    # so multi-word phrases are tried before their sub-words, and longer
    # single words before shorter ones (avoids "tez" matching inside a
    # longer unrelated token boundary edge case).
    sorted_keys: tuple[str, ...]
    single_word_keys: tuple[str, ...]  # subset used for fuzzy matching


@lru_cache(maxsize=1)
def load_dictionary(path: str = ROMAN_URDU_LOOKUP_PATH) -> UrduDictionary:
    """
    Load urdu_lookup_layer.csv into a matchable dictionary. Cached — the
    file is read once per process; call load_dictionary.cache_clear() if
    you hot-reload the CSV after appending new terms from the review
    queue (see query_log_miner.py) without restarting the service.
    """
    term_to_canonical: dict[str, str] = {}
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Roman-Urdu lookup CSV not found at {csv_path}. "
            f"Set config.ROMAN_URDU_LOOKUP_PATH or place the file there."
        )

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            term = row["roman_urdu_term"].strip().lower()
            canonical = row["canonical_meaning"].strip()
            if term and canonical:
                term_to_canonical[term] = canonical

    sorted_keys = tuple(
        sorted(term_to_canonical.keys(), key=lambda k: (-len(k.split()), -len(k)))
    )
    single_word_keys = tuple(k for k in sorted_keys if " " not in k)

    return UrduDictionary(
        term_to_canonical=term_to_canonical,
        sorted_keys=sorted_keys,
        single_word_keys=single_word_keys,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Match result — returned so callers (retrieval.py, query_log_miner.py,
# eval scripts) can log/inspect what fired, not just get the final string.
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MatchHit:
    matched_text: str       # the substring/token that matched
    canonical: str           # what it expanded to
    method: str               # "exact" or "fuzzy"
    fuzzy_score: Optional[float] = None  # only set for method="fuzzy"


@dataclass
class NormalizationResult:
    original_query: str
    expanded_query: str
    hits: list[MatchHit]

    @property
    def had_any_hit(self) -> bool:
        return len(self.hits) > 0

    @property
    def had_exact_hit(self) -> bool:
        return any(h.method == "exact" for h in self.hits)


# ═══════════════════════════════════════════════════════════════════════════
# Exact pass
# ═══════════════════════════════════════════════════════════════════════════


def _exact_match(normalized_query: str, dictionary: UrduDictionary) -> tuple[list[MatchHit], set[tuple[int, int]]]:
    """
    Scan normalized_query for dictionary keys, longest phrases first.
    Returns hits + the character spans consumed (so the fuzzy pass can
    skip tokens that already matched exactly).
    """
    hits: list[MatchHit] = []
    consumed_spans: list[tuple[int, int]] = []

    def _overlaps(span: tuple[int, int]) -> bool:
        return any(not (span[1] <= s or span[0] >= e) for s, e in consumed_spans)

    for key in dictionary.sorted_keys:
        pattern = r"(?<!\w)" + re.escape(key) + r"(?!\w)"
        for m in re.finditer(pattern, normalized_query):
            span = m.span()
            if _overlaps(span):
                continue
            consumed_spans.append(span)
            hits.append(
                MatchHit(
                    matched_text=key,
                    canonical=dictionary.term_to_canonical[key],
                    method="exact",
                )
            )

    return hits, set(consumed_spans)


# ═══════════════════════════════════════════════════════════════════════════
# Fuzzy pass (rapidfuzz) — only for tokens the exact pass didn't cover
# ═══════════════════════════════════════════════════════════════════════════


def _fuzzy_match(
    normalized_query: str,
    dictionary: UrduDictionary,
    consumed_spans: set[tuple[int, int]],
) -> list[MatchHit]:
    hits: list[MatchHit] = []
    cfg = URDU_NORMALIZATION

    if not cfg.enable_fuzzy_fallback or not dictionary.single_word_keys:
        return hits

    for m in re.finditer(r"\w+", normalized_query):
        span = m.span()
        token = m.group()

        if len(token) < cfg.fuzzy_min_token_length:
            continue
        if any(not (span[1] <= s or span[0] >= e) for s, e in consumed_spans):
            continue  # already exact-matched
        if token.isdigit():
            continue

        best = process.extractOne(
            token,
            dictionary.single_word_keys,
            scorer=fuzz.ratio,
            score_cutoff=cfg.fuzzy_score_cutoff,
        )
        if best is not None:
            matched_key, score, _ = best
            hits.append(
                MatchHit(
                    matched_text=token,
                    canonical=dictionary.term_to_canonical[matched_key],
                    method="fuzzy",
                    fuzzy_score=score,
                )
            )

    return hits


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════


def normalize_query(query: str) -> NormalizationResult:
    """
    Expand `query` with canonical terms for every roman-urdu word or
    known misspelling found (exact dictionary hit, then fuzzy fallback
    for anything left over). Safe to call on every search request —
    dictionary is cached in memory after first load, fuzzy pass is
    bounded by query length (typically 1-6 tokens for product search).
    """
    dictionary = load_dictionary()
    normalized = _normalize_text(query)

    exact_hits, consumed_spans = _exact_match(normalized, dictionary)
    fuzzy_hits = _fuzzy_match(normalized, dictionary, consumed_spans)

    all_hits = exact_hits + fuzzy_hits
    canonicals = list(dict.fromkeys(h.canonical for h in all_hits))  # dedupe, keep order

    expanded = query if not canonicals else f"{query} {' '.join(canonicals)}"

    return NormalizationResult(
        original_query=query,
        expanded_query=expanded,
        hits=all_hits,
    )


if __name__ == "__main__":
    import sys

    test_queries = sys.argv[1:] or [
        "kheema masala",
        "nehariii mix",
        "dahi bara mix",
        "kabli chana",
        "regular english query with no urdu",
    ]
    for q in test_queries:
        result = normalize_query(q)
        print(f"\nQuery: {q!r}")
        print(f"  Expanded: {result.expanded_query!r}")
        for h in result.hits:
            extra = f" (fuzzy score={h.fuzzy_score:.0f})" if h.method == "fuzzy" else ""
            print(f"  [{h.method}] {h.matched_text!r} -> {h.canonical!r}{extra}")
        if not result.hits:
            print("  (no dictionary hits)")
