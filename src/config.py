"""
Central configuration for the Naheed RAG search pipeline.
-----------------------------------------------------------
SINGLE SOURCE OF TRUTH for:
  1) Paths (catalogue, indexes, models)
  2) Model names (embedding, reranker, generation, router, evaluation)
  3) EVERY retrieval/generation hyperparameter (alpha, mmr, reranker,
     confidence threshold, cache size, etc.)

If you want to tune the pipeline, this is the ONLY file you should need
to edit. Nothing downstream (retrieval.py, generation.py, router.py,
api.py) should hardcode a hyperparameter default of its own — they all
read from RETRIEVAL / GENERATION / ROUTING below.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# 1. PATHS
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
EVAL_DIR = os.path.join(DATA_DIR, "evaluation")
HELD_OUT_EVAL_PATH = os.path.join(EVAL_DIR, "held_out_eval_v2.csv")

# Full ~109k-row product dump (source of truth for image URLs, among other
# fields not present in the trimmed prototype catalogue).
PRODUCTS_FULL_PATH = os.path.join(DATA_DIR, "products_full.jsonl")

MODELS_DIR = os.path.join(BASE_DIR, "models")

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
IMAGE_MAPPING_PATH = os.path.join(FRONTEND_DIR, "image_mapping.json")

# ── Catalogue switch ────────────────────────────────────────────────────────
# "prototype" = the 2130-product subset the embedding model was fine-tuned
#               on (final_products.csv). Use this for the demo/prototype.
# "full"      = the full ~155k-row catalogue (AllProducts.csv).
# Controlled by env var so you never have to touch code to flip it:
#   CATALOGUE_MODE=full python api.py
CATALOGUE_MODE = os.getenv("CATALOGUE_MODE", "prototype").lower()

if CATALOGUE_MODE not in {"prototype", "full"}:
    raise ValueError(f"CATALOGUE_MODE must be 'prototype' or 'full', got {CATALOGUE_MODE!r}")

if CATALOGUE_MODE == "prototype":
    CATALOGUE_PATH = os.path.join(RAW_DIR, "final_products.csv")
    INDEX_DIR = os.path.join(BASE_DIR, "indexes", "prototype")
else:
    CATALOGUE_PATH = os.path.join(RAW_DIR, "AllProducts.csv")
    INDEX_DIR = os.path.join(BASE_DIR, "indexes", "full")

# Small hand-made catalogue kept around for fast smoke tests / demos.
SAMPLE_CATALOGUE_PATH = os.path.join(SAMPLES_DIR, "sample_catalogue.csv")

CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25_index.pkl")
EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
EMBED_IDS_PATH = os.path.join(INDEX_DIR, "embedding_ids.json")

os.makedirs(INDEX_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 2. MODEL NAMES
# ═══════════════════════════════════════════════════════════════════════════

# Embedding model — BGE-M3, fine-tuned (LoRA, merged) on the 2130-product
# / 6,133-anchor Roman-Urdu+English pair dataset ("model_run_6").
EMBEDDING_MODEL_NAME = os.path.join(MODELS_DIR, "model_run_6")

# Cross-encoder / reranker — multilingual BGE family, no separate
# translation step needed for Roman Urdu + English joint scoring.
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Generation LLM ("LLM1" — smaller/faster, customer-facing answer).
GENERATION_MODEL_NAME = "llama-3.1-8b-instant"

# Router / intent-classification + structured-extraction models.
# Kept separate from GENERATION_MODEL_NAME since classification and
# extraction can use a different (and here, larger) model than the
# one that writes the final customer-facing answer.
ROUTER_INTENT_MODEL_NAME = "llama-3.1-8b-instant"
ROUTER_EXTRACTION_MODEL_NAME = "llama-3.3-70b-versatile"

# Evaluation LLM ("LLM2" — a judge model shouldn't be weaker than what
# it's judging, so this defaults larger than GENERATION_MODEL_NAME).
EVALUATION_MODEL_NAME = "llama-3.3-70b-versatile"

# ═══════════════════════════════════════════════════════════════════════════
# 3. API KEYS
# ═══════════════════════════════════════════════════════════════════════════

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# NOTE: we deliberately do NOT hard-fail on a missing OPENAI_API_KEY here.
# Every model currently configured above (GENERATION/ROUTER/EVALUATION)
# is a Groq-hosted "llama-*" model, so requiring an OpenAI key at import
# time would block the whole app from starting for no functional reason.
# If you point one of the *_MODEL_NAME settings at a "gpt-*"/"o1-*" model,
# llm_client.call_llm() will raise a clear error at call time if
# GROQ/OPENAI/GEMINI_API_KEY is missing for the provider actually used.


# ═══════════════════════════════════════════════════════════════════════════
# 4. RETRIEVAL HYPERPARAMETERS  (the single dial-board for tuning search)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RetrievalConfig:
    """
    All retrieval pipeline hyperparameters live here. `retrieval.py` reads
    these as defaults; a caller can still override any of them per-request
    via `SearchRequest`, but if a request field is left as `None`, this is
    what it falls back to.
    """

    # --- Stage 1: candidate generation ---
    bm25_candidates: int = 100
    vector_candidates: int = 100

    # --- Stage 1: Weighted Reciprocal Rank Fusion ---
    rrf_k: int = 60
    rrf_alpha: float = 0.4          # weight given to vector results; (1-alpha) to BM25
    fusion_top_n: int = 100         # truncate fused candidates to this many before filtering

    # --- Stage 2a: MMR diversity ---
    # mmr_lambda = 1.0 -> pure relevance (MMR effectively skipped)
    # mmr_lambda = 0.0 -> pure diversity
    mmr_lambda: float = 1.0
    mmr_candidates: int = 30

    # --- Stage 2b: Cross-encoder reranking ---
    use_reranker: bool = False
    conditional_rerank: bool = False     # skip reranker when Stage 1 is already confident
    confidence_threshold: float = 0.7   # fraction of max theoretical WRRF score

    # --- Output ---
    default_top_k: int = 12

    # --- Relevance gate ---
    # If the best match's confidence falls below this, search() returns an
    # EMPTY result set rather than forcing a top-k match on an irrelevant
    # query (e.g. a product that doesn't exist in the catalogue at all).
    # The confidence signal used is:
    #   - the cross-encoder's relevance score (sigmoid-normalized to 0-1),
    #     when the reranker actually ran, OR
    #   - the top vector cosine similarity, when the reranker was skipped
    #     (disabled via use_reranker=False, or conditionally skipped
    #     because Stage 1 already agreed strongly).
    # Raise this to be stricter about rejecting irrelevant queries; lower
    # it to be more permissive.
    min_relevance_score: float = 0.30

    # --- Cache ---
    cache_maxsize: int = 1000
    cache_ttl_seconds: int = 3600


RETRIEVAL = RetrievalConfig()


# ═══════════════════════════════════════════════════════════════════════════
# 5. GENERATION HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.3


GENERATION = GenerationConfig()


# ═══════════════════════════════════════════════════════════════════════════
# 6. ROUTER HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RoutingConfig:
    intent_max_new_tokens: int = 50
    intent_temperature: float = 0.0
    extraction_max_new_tokens: int = 300
    extraction_temperature: float = 0.0


ROUTING = RoutingConfig()