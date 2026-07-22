"""
STEP 1b: BM25 Index Generation
-------------------------------
Reads the chunked product catalogue (chunks.jsonl) produced by chunking.py
and builds a BM25Okapi index that is serialized to disk as a pickle file.

Why BM25Okapi?
  - It's the strongest of the three BM25 variants in rank_bm25 for
    variable-length documents (our chunks range from ~10 tokens for
    description-less products to ~100+ tokens for products with full
    descriptions). Okapi's length-normalization parameter (b) means short
    chunks aren't unfairly penalized.

Tokenization choices:
  - Lowercase + regex word splitting (keeps alphanumerics, hyphens for
    product codes like "IC-1143194", and dots for sizes like "0.5").
  - A small, curated English stop-word list is removed. We intentionally
    do NOT strip Roman-Urdu stop words (ka, ke, ki, ko, se, etc.) here.
    Those are common in USER queries but almost never appear in the
    English-only product text, so removing them from the corpus would
    have no effect. They need to be handled at query-time in retrieval.py.
  - No stemming — brand names, product codes, and category terms are
    already in their canonical form; stemming would mangle them
    (e.g. "Pampers" -> "pamper", "Diapers" -> "diaper") and could hurt
    exact-brand matching.

Output:
  indexes/bm25_index.pkl   – pickled dict containing:
      "index"   : the BM25Okapi object
      "ids"     : list[str] aligned with the BM25 corpus (chunk ids)
      "corpus"  : list[list[str]] the tokenized corpus (useful for debugging)

Usage:
  python bm25_index.py              # builds from the full chunks.jsonl
  python bm25_index.py --sample 500 # quick smoke test with 500 chunks
"""

import argparse
import json
import pickle
import re
import time
from typing import Optional

from rank_bm25 import BM25Okapi

from config import BM25_INDEX_PATH, CHUNKS_PATH

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Matches sequences of word chars, hyphens (for SKUs like IC-1143194),
# and dots (for sizes like 0.5kg). Single standalone dots/hyphens that
# were used as separators in the chunk text (" . ") are naturally excluded
# because they don't match the \w requirement on either side.
_TOKEN_PATTERN = re.compile(r"[\w][\w.\-]*[\w]|[\w]", re.UNICODE)

# Minimal English stop words. Deliberately small — we'd rather have a few
# extra "the"s in the index than accidentally strip a meaningful term.
# Not including "s" because it's commonly a size indicator ("S", "M", "L").
# Minimal English and Roman Urdu stop words. Deliberately small — we'd rather 
# have a few extra "the"s in the index than accidentally strip a meaningful term.
# Not including "s" because it's commonly a size indicator ("S", "M", "L").
_STOP_WORDS: frozenset[str] = frozenset({
    # English
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "this", "that", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "not", "no", "nor", "so", "if", "then", "than",
    "too", "very", "just", "about", "above", "after", "before", "between",
    "into", "through", "during", "out", "off", "over", "under", "again",
    "further", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "also",
    
    # Roman Urdu - Prepositions, Conjunctions & Particles
    "ka", "ke", "ki", "ko", "se", "mein", "pe", "par", "tak", "andar", 
    "bahar", "paas", "sath", "saath", "aur", "ya", "magar", "lekin", 
    "kyunke", "kyunki", "balkay", "warna", "taake", "k", "liye", "keliye", 
    "wala", "wali", "wale", "bhi", "hi", "bas", "sirf",
    
    # Roman Urdu - Pronouns
    "main", "mera", "meri", "mere", "mujhey", "mujhe", 
    "tum", "tumhara", "tumhari", "tumhare", "tujhey", "tujhe",
    "aap", "aapka", "aapki", "aapke", 
    "hum", "hamara", "hamari", "hamare", "humein",
    "wo", "woh", "uska", "uski", "uske", "usay", "usey",
    "un", "unka", "unki", "unke", "unhain", "unhein",
    "yeh", "ye", "is", "iska", "iski", "iske", "isey",
    "in", "inka", "inki", "inke", "inhain", "inhein",
    
    # Roman Urdu - Verbs (Being/Doing)
    "hai", "hain", "ho", "hoon", "tha", "thi", "thay", "the", 
    "hoga", "hogi", "honge", "karna", "karo", "karen", "kiye",
    
    # Roman Urdu - Question Words & Quantifiers
    "kya", "kab", "kahan", "kaise", "kesa", "kesi", "kese", 
    "kyun", "kyu", "kiyu", "kitna", "kitni", "kitne", "kaun", "kon",
    "bohat", "thora", "thori", "kuch", "sab", "her", "har", "koi"
})


def tokenize(text: str) -> list[str]:
    """
    Tokenize a chunk's text field into a list of lowercase terms,
    with stop words removed.

    >>> tokenize("Pampers Baby Dry Size 4 . Mother & Baby")
    ['pampers', 'baby', 'dry', 'size', '4', 'mother', 'baby']
    """
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def load_chunks(path: str = CHUNKS_PATH) -> list[dict]:
    """Load chunks from a JSONL file. Mirrors chunking.load_chunks."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_bm25_index(
    chunks: list[dict],
) -> tuple[BM25Okapi, list[str], list[list[str]]]:
    """
    Build a BM25Okapi index from a list of chunk dicts.

    Returns:
        index   - the BM25Okapi object, ready for .get_scores()
        ids     - list of chunk IDs, positionally aligned with the index
        corpus  - the tokenized corpus (list of token lists)
    """
    ids: list[str] = []
    corpus: list[list[str]] = []

    for chunk in chunks:
        ids.append(chunk["id"])
        corpus.append(tokenize(chunk["text"]))

    # BM25Okapi defaults: k1=1.5, b=0.75, epsilon=0.25
    # These are the classic IR defaults and work well for product search.
    # k1 controls term-frequency saturation (1.5 is standard).
    # b controls length normalization (0.75 is standard — important for us
    #   because chunks vary a lot in length depending on whether the product
    #   had a description or not).
    index = BM25Okapi(corpus)

    return index, ids, corpus


def save_index(
    index: BM25Okapi,
    ids: list[str],
    corpus: list[list[str]],
    path: str = BM25_INDEX_PATH,
) -> None:
    """Persist the BM25 index, ids, and corpus to a pickle file."""
    payload = {
        "index": index,
        "ids": ids,
        "corpus": corpus,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path: str = BM25_INDEX_PATH) -> tuple[BM25Okapi, list[str], list[list[str]]]:
    """
    Load a previously saved BM25 index from disk.

    Returns:
        index, ids, corpus — same as build_bm25_index().
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["index"], payload["ids"], payload["corpus"]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(sample: Optional[int] = None) -> None:
    print(f"Loading chunks from {CHUNKS_PATH} ...")
    t0 = time.perf_counter()
    chunks = load_chunks()
    print(f"  Loaded {len(chunks):,} chunks in {time.perf_counter() - t0:.1f}s")

    if sample is not None and sample < len(chunks):
        chunks = chunks[:sample]
        print(f"  (using first {sample:,} chunks for smoke test)")

    print("Tokenizing and building BM25Okapi index ...")
    t0 = time.perf_counter()
    index, ids, corpus = build_bm25_index(chunks)
    elapsed = time.perf_counter() - t0
    print(f"  Built index over {len(ids):,} documents in {elapsed:.1f}s")

    # Quick corpus stats for sanity checking
    token_counts = [len(doc) for doc in corpus]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    print(f"  Token stats — min: {min_tokens}, avg: {avg_tokens:.0f}, max: {max_tokens}")

    print(f"Saving index to {BM25_INDEX_PATH} ...")
    t0 = time.perf_counter()
    save_index(index, ids, corpus)
    elapsed = time.perf_counter() - t0
    import os
    size_mb = os.path.getsize(BM25_INDEX_PATH) / (1024 * 1024)
    print(f"  Saved ({size_mb:.1f} MB) in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Quick smoke test: run a couple of queries to prove the index works
    # ------------------------------------------------------------------
    test_queries = [
        "pampers baby diapers",
        "shampoo hair care",
        "face cream moisturizer",
    ]
    print("\n--- Smoke Test ---")
    for query in test_queries:
        query_tokens = tokenize(query)
        scores = index.get_scores(query_tokens)

        # Get top-5 results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        print(f"\nQuery: '{query}'  (tokens: {query_tokens})")
        for rank, idx in enumerate(top_indices, 1):
            chunk = chunks[idx]
            name = chunk["metadata"]["name"]
            print(f"  {rank}. [{ids[idx]}] {name[:60]:<60s}  score={scores[idx]:.2f}")

    print("\n✓ BM25 index built and verified successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a BM25 index from chunks.jsonl"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use only the first N chunks (for quick testing)",
    )
    args = parser.parse_args()
    main(sample=args.sample)
