"""
Query Log Miner
------------------
Offline job (run on a schedule — cron/Airflow/whatever you already use;
no code dependency on a specific scheduler here) that:

  1. Reads every entry via history_logger.load_search_history() (written
     by history_logger.log_search() on every /api/search call).
  2. Flags queries that were a "miss": zero results returned, OR the
     top result's score fell below
     config.URDU_NORMALIZATION.mining_low_confidence_threshold.
  3. For each miss, re-runs it through the CURRENT urdu_normalizer
     dictionary. If normalize_query() found NO hits at all (neither
     exact nor fuzzy), every token in that query is a candidate — it
     might be a roman-urdu word/misspelling we don't know about yet,
     or it might just be genuinely out-of-catalogue. We can't tell
     which automatically, hence step 4.
  4. Aggregates candidate tokens by frequency across all miss queries,
     filters to >= mining_min_frequency occurrences (skips one-off
     typos not worth permanently adding), and writes a review CSV —
     NOT an automatic merge into urdu_lookup_layer.csv. A human looks
     at each candidate + its sample queries + sample zero-hit context
     before deciding whether it's a real roman-urdu term to add, an
     out-of-catalogue product to ignore, or noise.

Usage:
    python query_log_miner.py                  # full run
    python query_log_miner.py --days 7          # last 7 days only
    python query_log_miner.py --dry-run         # print, don't write CSV

Output:
    data/review/urdu_candidate_terms.csv — columns:
        candidate_term, frequency, sample_queries, sample_miss_reason

Review workflow:
    1. Open the CSV, skim candidate_term + sample_queries.
    2. For genuine roman-urdu terms/misspellings: add a row to
       urdu_lookup_layer.csv (roman_urdu_term, canonical_meaning, source
       = "round3_query_mining"), then call
       urdu_normalizer.load_dictionary.cache_clear() (or restart the
       service) to pick it up.
    3. For noise / genuinely out-of-catalogue queries: delete the row,
       no action needed — it'll resurface if it keeps recurring, but
       won't spam the queue every single run (see --seen-log below).
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from config import URDU_CANDIDATE_TERMS_PATH, URDU_NORMALIZATION
from history_logger import SearchHistoryEntry, load_search_history
from urdu_normalizer import normalize_query, load_dictionary

# OFFLINE / BUILD-TIME SCRIPT: run manually or on a schedule (see module
# docstring above). Not imported by api.py — it's a maintenance job that
# reads search history and writes a review CSV of candidate roman-urdu
# terms for a human to merge into urdu_lookup_layer.csv.

# Tokens that are almost never a roman-urdu culinary term worth adding —
# filtered out before frequency counting so the review queue isn't
# swamped with English stopwords / obvious brand fragments / units.
_IGNORE_TOKENS = frozenset({
    "the", "and", "for", "with", "of", "in", "on", "a", "an", "gm", "g",
    "kg", "ml", "l", "oz", "pack", "packet", "box", "jar", "bottle",
    "pouch", "tin", "size", "large", "small", "medium", "new", "best",
    "top", "cheap", "price", "buy", "online", "near", "me", "shop",
    # Generic words that are already literal tokens in most product names
    # (masala, mix, powder, sauce...) — they never need dictionary
    # expansion themselves, and would otherwise flood the review queue
    # on every single miss query in this catalogue.
    "masala", "mix", "powder", "sauce", "recipe", "spice", "spices",
    "paste", "seasoning", "flavour", "flavor",
})

# A row already reviewed (approved into the CSV, or explicitly rejected)
# is tracked here so re-runs don't keep re-surfacing it. Simple flat file
# of one term per line; delete a line to force it to resurface.
_SEEN_LOG_PATH = Path(URDU_CANDIDATE_TERMS_PATH).parent / "urdu_candidate_terms.seen.txt"


def _load_seen() -> set[str]:
    if not _SEEN_LOG_PATH.exists():
        return set()
    return {line.strip().lower() for line in _SEEN_LOG_PATH.read_text().splitlines() if line.strip()}


def _load_history(days: int | None) -> list[SearchHistoryEntry]:
    """
    Load search history via history_logger.load_search_history() — reuses
    the real SearchHistoryEntry model rather than re-parsing the JSONL by
    hand, so this stays correct automatically if that schema ever changes.
    `timestamp` is a float epoch (time.time()) per history_logger.py.
    """
    entries = load_search_history()
    if not entries:
        print(f"[miner] no history entries yet — nothing to mine.")
        return []

    if days is not None:
        cutoff = (datetime.now() - timedelta(days=days)).timestamp()
        entries = [e for e in entries if e.timestamp >= cutoff]

    return entries


def _entry_is_miss(entry: SearchHistoryEntry, threshold: float) -> tuple[bool, str]:
    """
    Decide whether a logged search entry counts as a "miss" worth mining:
      - entry.results == []                -> zero-result miss
      - entry.results[0].score < threshold  -> low-confidence miss
    `score` is a required float on HistoryResultItem, never None, so no
    defensive None-check needed here.
    """
    if len(entry.results) == 0:
        return True, "zero_results"

    top_score = entry.results[0].score
    if top_score < threshold:
        return True, f"low_confidence(score={top_score:.3f})"

    return False, ""


def mine(days: int | None = None) -> list[dict]:
    cfg = URDU_NORMALIZATION
    entries = _load_history(days)
    print(f"[miner] loaded {len(entries)} history entries")

    # Force a fresh dictionary read in case new terms were merged since
    # this process started.
    load_dictionary.cache_clear()

    seen = _load_seen()

    candidate_queries: dict[str, list[tuple[str, str]]] = defaultdict(list)
    # candidate_token -> list of (original_query, miss_reason)

    miss_count = 0
    for entry in entries:
        query = entry.query
        if not query:
            continue

        is_miss, reason = _entry_is_miss(entry, cfg.mining_low_confidence_threshold)
        if not is_miss:
            continue
        miss_count += 1

        norm_result = normalize_query(query)
        # Only skip the SPECIFIC tokens the normalizer already matched —
        # not the whole query. Otherwise a single spurious fuzzy hit on
        # one token (e.g. correctly-spelled "powder" fuzzy-matching our
        # own "poder"->"powder" misspelling entry at ratio 91) silently
        # hides every other, genuinely-unknown token in the same query
        # from ever reaching the review queue.
        matched_tokens: set[str] = set()
        for h in norm_result.hits:
            matched_tokens.update(h.matched_text.lower().split())

        for token in re.findall(r"[a-zA-Z]+", query.lower()):
            if len(token) < 3 or token in _IGNORE_TOKENS or token in seen:
                continue
            if token in matched_tokens:
                continue
            candidate_queries[token].append((query, reason))

    print(f"[miner] {miss_count} miss entries found; "
          f"{len(candidate_queries)} candidate tokens with no dictionary hit")

    rows = []
    for token, occurrences in candidate_queries.items():
        if len(occurrences) < cfg.mining_min_frequency:
            continue
        sample_queries = "; ".join(q for q, _ in occurrences[:5])
        sample_reasons = "; ".join(dict.fromkeys(r for _, r in occurrences[:5]))
        rows.append({
            "candidate_term": token,
            "frequency": len(occurrences),
            "sample_queries": sample_queries,
            "sample_miss_reason": sample_reasons,
        })

    rows.sort(key=lambda r: -r["frequency"])
    return rows


def write_review_csv(rows: list[dict], path: str = URDU_CANDIDATE_TERMS_PATH) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_term", "frequency", "sample_queries", "sample_miss_reason"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[miner] wrote {len(rows)} candidate terms -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Mine search history for unrecognized roman-urdu terms")
    parser.add_argument("--days", type=int, default=None, help="Only consider the last N days of history")
    parser.add_argument("--dry-run", action="store_true", help="Print results, don't write the CSV")
    args = parser.parse_args()

    rows = mine(days=args.days)

    if not rows:
        print("[miner] no new candidate terms above the frequency threshold — nothing to review.")
        return

    print("\nTop candidates:")
    for r in rows[:20]:
        print(f"  {r['frequency']:>3}x  {r['candidate_term']:<20s}  e.g. {r['sample_queries'][:80]}")

    if not args.dry_run:
        write_review_csv(rows)
    else:
        print("\n[miner] --dry-run set, CSV not written.")


if __name__ == "__main__":
    main()