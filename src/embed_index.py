"""
STEP 1c: Dense Embedding Index Generation
-------------------------------------------
Reads the chunked product catalogue (chunks.jsonl) and encodes every chunk's
`text` field into a dense vector using BAAI/bge-m3.

Why BGE-M3?
  - Multilingual: 100+ languages including Urdu — handles mixed-script
    Roman Urdu + English queries in a single embedding space without
    needing translation.
  - M3 = Multi-lingual, Multi-Functionality, Multi-Granularity.
  - Produces 1024-dim embeddings.

Performance note:
  On a CPU-only machine with 155k chunks, expect ~30-60 minutes for the
  full corpus. Use --sample N for quick smoke tests.

Output:
  indexes/embeddings.npy     – float32 array of shape (N, 1024)
  indexes/embedding_ids.json – list[str] of chunk IDs aligned with rows

Usage:
  python embed_index.py                # full corpus
  python embed_index.py --sample 500   # quick smoke test
  python embed_index.py --batch-size 32 # smaller batches if RAM-constrained
"""

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    EMBED_IDS_PATH,
    EMBEDDING_MODEL_NAME,
)

# ---------------------------------------------------------------------------
# Chunk loading (duplicated from chunking.py to avoid circular deps)
# ---------------------------------------------------------------------------

def load_chunks(path: str = CHUNKS_PATH) -> list[dict]:
    """Load chunks from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def build_embeddings(
    chunks: list[dict],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = 64,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Encode all chunk texts into dense vectors.

    Returns:
        embeddings – np.ndarray of shape (N, dim), dtype float32
        ids        – list of chunk IDs, positionally aligned with rows
    """
    print(f"Loading embedding model: {model_name} ...")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    ids: list[str] = [chunk["id"] for chunk in chunks]
    texts: list[str] = [chunk["text"] for chunk in chunks]

    print(f"Encoding {len(texts):,} texts (batch_size={batch_size}) ...")
    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # L2-normalize so dot product = cosine sim
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Encoded in {elapsed:.1f}s ({len(texts) / elapsed:.0f} texts/sec)")

    return embeddings.astype(np.float32), ids


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_embeddings(
    embeddings: np.ndarray,
    ids: list[str],
    embeddings_path: str = EMBEDDINGS_PATH,
    ids_path: str = EMBED_IDS_PATH,
) -> None:
    """Save embeddings array and aligned IDs to disk."""
    np.save(embeddings_path, embeddings)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)


def load_embeddings(
    embeddings_path: str = EMBEDDINGS_PATH,
    ids_path: str = EMBED_IDS_PATH,
) -> tuple[np.ndarray, list[str]]:
    """
    Load previously saved embeddings from disk.

    Returns:
        embeddings – np.ndarray of shape (N, dim)
        ids        – list of chunk IDs aligned with rows
    """
    embeddings = np.load(embeddings_path)
    with open(ids_path, encoding="utf-8") as f:
        ids = json.load(f)
    return embeddings, ids


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(
    sample: Optional[int] = None,
    batch_size: int = 64,
) -> None:
    print(f"Loading chunks from {CHUNKS_PATH} ...")
    t0 = time.perf_counter()
    chunks = load_chunks()
    print(f"  Loaded {len(chunks):,} chunks in {time.perf_counter() - t0:.1f}s")

    if sample is not None and sample < len(chunks):
        chunks = chunks[:sample]
        print(f"  (using first {sample:,} chunks for smoke test)")

    embeddings, ids = build_embeddings(chunks, batch_size=batch_size)
    print(f"  Embedding shape: {embeddings.shape}  dtype: {embeddings.dtype}")

    print(f"Saving to {EMBEDDINGS_PATH} and {EMBED_IDS_PATH} ...")
    save_embeddings(embeddings, ids)
    emb_mb = os.path.getsize(EMBEDDINGS_PATH) / (1024 * 1024)
    print(f"  Saved embeddings ({emb_mb:.1f} MB)")

    # ------------------------------------------------------------------
    # Quick smoke test: nearest-neighbor search for a sample query
    # ------------------------------------------------------------------
    print("\n--- Smoke Test ---")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    test_queries = [
        "pampers baby diapers",
        "shampoo for hair",
        "face cream moisturizer",
    ]
    for query in test_queries:
        q_emb = model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        # Cosine similarity (embeddings are already L2-normalized → dot product)
        scores = embeddings @ q_emb.T
        scores = scores.squeeze()
        top_indices = np.argsort(scores)[::-1][:5]

        print(f"\nQuery: '{query}'")
        for rank, idx in enumerate(top_indices, 1):
            chunk = chunks[idx]
            name = chunk["metadata"]["name"]
            print(
                f"  {rank}. [{ids[idx]}] {name[:60]:<60s}  "
                f"cosine={scores[idx]:.4f}"
            )

    print("\n✓ Embedding index built and verified successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build dense vector embeddings from chunks.jsonl"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use only the first N chunks (for quick testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Encoding batch size (reduce if RAM-constrained)",
    )
    args = parser.parse_args()
    main(sample=args.sample, batch_size=args.batch_size)