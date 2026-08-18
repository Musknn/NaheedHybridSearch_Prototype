"""
Central configuration for the Naheed hybrid search pipeline.
-----------------------------------------------------------
SINGLE SOURCE OF TRUTH for:
  1) Paths (catalogue, indexes, models)
  2) Model names (embedding, reranker)
  3) EVERY retrieval hyperparameter (alpha, mmr, reranker, confidence
     threshold, cache size, etc.)

If you want to tune the pipeline, this is the ONLY file you should need
to edit. Nothing downstream (retrieval.py, api.py, etc.) should hardcode
a hyperparameter default of its own — they all read from RETRIEVAL /
URDU_NORMALIZATION / FEEDBACK / CROSS_SELL below.

NOTE: this project originally also had an LLM answer-generation layer
(generation.py/router.py/llm_client.py + their GENERATION/ROUTING config
sections and OPENAI/GEMINI/GROQ API keys). That layer was scoped out —
this submission is the hybrid search + cross-sell pipeline only — so
those sections were removed from here along with the files themselves.
"""
import os
from dataclasses import dataclass, field

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

MODELS_DIR = os.path.join(BASE_DIR, "models")

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
ORDER_HISTORY_PATH = os.path.join(RAW_DIR, "order_history.csv")

# ── Roman-Urdu lookup layer ─────────────────────────────────────────────
ROMAN_URDU_DIR = os.path.join(DATA_DIR, "roman_urdu")
ROMAN_URDU_LOOKUP_PATH = os.path.join(ROMAN_URDU_DIR, "urdu_lookup_layer.csv")
os.makedirs(ROMAN_URDU_DIR, exist_ok=True)

# Query-log mining output: candidate new terms surfaced from real zero-hit
# / low-confidence queries, for human review before being merged into
# urdu_lookup_layer.csv above. See query_log_miner.py.
REVIEW_DIR = os.path.join(DATA_DIR, "review")
URDU_CANDIDATE_TERMS_PATH = os.path.join(REVIEW_DIR, "urdu_candidate_terms.csv")
os.makedirs(REVIEW_DIR, exist_ok=True)

CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25_index.pkl")
EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
EMBED_IDS_PATH = os.path.join(INDEX_DIR, "embedding_ids.json")

# ── Feedback (clicks / add-to-cart) ─────────────────────────────────────────
# Append-only history log: every click/add-to-cart event, for manual review
# ("what did we get right/wrong") and as the source of truth the boost index
# is rebuilt from.
FEEDBACK_DIR = os.path.join(DATA_DIR, "feedback")
FEEDBACK_LOG_PATH = os.path.join(FEEDBACK_DIR, "feedback_log.jsonl")
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# ── Search history (query -> every returned product + its final score) ─────
# Separate from FEEDBACK_LOG_PATH above: this logs EVERY search result set
# regardless of whether the user ever clicked/added anything, so you can
# review what the pipeline actually returned for any query, and manually
# label results as correct/incorrect after the fact.
HISTORY_DIR = os.path.join(DATA_DIR, "history")
SEARCH_HISTORY_PATH = os.path.join(HISTORY_DIR, "search_history.jsonl")
os.makedirs(HISTORY_DIR, exist_ok=True)

os.makedirs(INDEX_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 2. MODEL NAMES
# ═══════════════════════════════════════════════════════════════════════════

# Embedding model — BGE-M3, fine-tuned (LoRA, merged) on the 2130-product
# / 6,133-anchor Roman-Urdu+English pair dataset ("model_run_6").
#
# Hosted on the Hugging Face Hub at https://huggingface.co/muskannnnn/Prototype
# — SentenceTransformer(...) below loads directly from the Hub by repo id,
# so no local model folder is required. This is the SAME repo id used both
# here (for query-time encoding) and in the Kaggle notebook that generates
# the corpus embeddings offline — see "Generating embeddings" in README.md.
# Override with the EMBEDDING_MODEL_NAME env var to point at a local path
# (e.g. MODELS_DIR/model_run_6) instead, if you'd rather not depend on the Hub.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "muskannnnn/Prototype")

# Cross-encoder / reranker — multilingual BGE family, no separate
# translation step needed for Roman Urdu + English joint scoring.
# Fully wired into retrieval.py (see cross_encoder_rerank()) but OFF by
# default — see RETRIEVAL.use_reranker below.
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# ═══════════════════════════════════════════════════════════════════════════
# 3. RETRIEVAL HYPERPARAMETERS  (the single dial-board for tuning search)
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
    bm25_candidates: int = 150
    vector_candidates: int = 150

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
    conditional_rerank: bool = True     # skip reranker when Stage 1 is already confident
    confidence_threshold: float = 0.7   # fraction of max theoretical WRRF score

    # --- Output ---
    default_top_k: int = 5

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

@dataclass(frozen=True)
class UrduNormalizationConfig:
    # \"\"\"
    # Controls urdu_normalizer.py — the query-time expansion layer that
    # sits in front of both BM25 and vector search (see retrieval.py's
    # bm25_search()/vector_search() call sites).
    # \"\"\"

    # Master toggles — each retrieval path can be independently switched
    # off (e.g. if eval later shows expansion hurts the embedding side
    # for some query classes, flip enable_for_vector=False without
    # touching bm25's behavior).
    enable_for_bm25: bool = True
    enable_for_vector: bool = True

    # --- Fuzzy fallback (rapidfuzz) ---
    enable_fuzzy_fallback: bool = True
    # rapidfuzz.fuzz.ratio score (0-100). 85 is intentionally strict —
    # false positives here silently corrupt the query, so tune upward
    # first if you see bad matches, downward only after checking logs.
    fuzzy_score_cutoff: float = 85.0
    # Don't fuzzy-match short tokens ("tez", "dal") — too easy to false-
    # positive against an unrelated 3-4 letter dictionary key.
    fuzzy_min_token_length: int = 5

    # --- Query log mining (query_log_miner.py) ---
    # A query counts as a "miss" worth reviewing if the top result's
    # confidence fell below this, OR it returned zero results outright.
    # Deliberately reuses the same scale as RETRIEVAL.min_relevance_score
    # but is independently tunable.
    mining_low_confidence_threshold: float = 0.30
    # Only surface a candidate term for human review once it's appeared
    # in at least this many distinct low-confidence/zero-result queries
    # — filters out one-off typos that aren't worth adding permanently.
    mining_min_frequency: int = 2


URDU_NORMALIZATION = UrduNormalizationConfig()

# ═══════════════════════════════════════════════════════════════════════════
# 4. FEEDBACK (click / add-to-cart) HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class FeedbackConfig:

    enabled: bool = True

    event_weights: dict[str, float] = field(
        default_factory=lambda: {
            "click": 1.0,
            "add_to_cart": 3.0,
            "relevant": 2.0,
            "irrelevant": -5.0,
        }
    )
    similarity_threshold: float = 0.75

    boost_weight: float = 0.02


FEEDBACK = FeedbackConfig()

# ═══════════════════════════════════════════════════════════════════════════
# 5. CROSS-SELL / UPSELL HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class CrossSellConfig:

    default_top_n: int = 5

    min_bought_percent: float = 5.0
    min_orders: int = 2
    in_stock_only: bool = True


CROSS_SELL = CrossSellConfig()