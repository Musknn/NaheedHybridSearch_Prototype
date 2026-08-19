# Naheed Product Search

A hybrid search engine and cross-sell/upsell recommender for the Naheed
Super Market product catalogue, with a web frontend on top. Search
combines BM25 keyword matching and dense vector retrieval, fused with
Weighted Reciprocal Rank Fusion and filtered by a confidence threshold,
with Roman-Urdu/misspelling query normalization and a click/add-to-cart
feedback loop layered on top.

## Project structure

```
NaheedChatbot/
├── data/
│   ├── raw/
│   │   ├── final_products.csv       # the 2,130-product catalogue actually in use
│   │   ├── npk_prods_dmp (1).csv    # full catalogue dump — not currently used
│   │   └── order_history.csv        # order co-purchase data, feeds cross_sell.py
│   ├── roman_urdu/
│   │   └── urdu_lookup_layer.csv    # roman-urdu/misspelling → canonical term dictionary
│   ├── feedback/feedback_log.jsonl  # every click/add_to_cart/relevant/irrelevant event
│   ├── history/search_history.jsonl # every search request + what it returned
│   └── review/                      # query_log_miner.py's output, for human review
├── frontend/
│   ├── index.html                   # search UI (API_URL hardcoded to localhost:8000/api)
│   ├── image_mapping.json           # SKU → image URL, fetched directly by index.html
│   ├── images/
│   └── unnamed.png                  # header logo
├── indexes/
│   └── prototype/
│       ├── chunks.jsonl             # one row per product, built by chunking.py
│       ├── bm25_index.pkl           # built by bm25_index.py
│       ├── embeddings.npy           # built on Kaggle GPU, see below
│       └── embedding_ids.json       # positionally aligned with embeddings.npy
├── finetuning/
│   ├── Notebooks(Cleaned)/
│   │   ├── Finetuning(cleaned).ipynb          # annotated fine-tuning code, explained step by step
│   │   └── EmbeddingGeneration(cleaned).ipynb # annotated embedding-generation code
│   └── Notebooks(Output)/
│       ├── FineTuning.ipynb                   # original fine-tuning run, with its training output
│       └── EmbeddingGeneration.ipynb          # original embedding-generation run, with its output
├── logs/
│   └── pipeline_history.jsonl
├── models/
│   └── model_run_6/                 # local copy of the fine-tuned embedding model
├── src/
│   ├── api.py                       # FastAPI app — entry point
│   ├── retrieval.py                 # core hybrid search pipeline
│   ├── bm25_index.py                # BM25 index builder + tokenize() used at query time
│   ├── chunking.py                  # catalogue CSV → chunks.jsonl
│   ├── urdu_normalizer.py           # roman-urdu/misspelling query expansion
│   ├── feedback.py                  # click/cart feedback → ranking boosts
│   ├── history_logger.py            # search history logging
│   ├── cross_sell.py                # "bought together" recommendations
│   ├── query_log_miner.py           # mines search history to grow the roman-urdu dictionary
│   └── config.py                    # single source of truth for paths/models/hyperparameters
├── testqueries.md                   # example queries used to sanity-check search
├── requirements.txt
├── .env
└── .gitignore
```

The catalogue actually served is `final_products.csv` (2,130 products) —
`npk_prods_dmp (1).csv` is a full-catalogue dump kept in `data/raw/` but
not wired into `config.py` or read by anything in `src/`.

## Architecture

```
                    ┌─────────────────────────────────────────┐
 Offline / build ──▶│ chunking.py  →  bm25_index.py            │  (local CPU)
 time (run once,    │                                          │
 rerun when the     │ Kaggle notebook, GPU  →  embeddings.npy  │  (see below)
 catalogue changes) │                          + embedding_ids.json
                     └─────────────────────────────────────────┘
                                        │
                                        ▼
                          indexes/prototype/{chunks.jsonl,
                          bm25_index.pkl, embeddings.npy,
                          embedding_ids.json}
                                        │
                                        ▼
Runtime (every  ┌───────────────────────────────────────────────────┐
search request) │  api.py                                            │
                 │    └─▶ retrieval.py : search()                    │
                 │          ├─ urdu_normalizer.py  (query expansion) │
                 │          ├─ bm25_search()   (uses bm25_index.py's │
                 │          │                   tokenize())          │
                 │          ├─ vector_search()  (encodes the query   │
                 │          │   with muskannnnn/Prototype)            │
                 │          ├─ weighted_reciprocal_rank_fusion()     │
                 │          ├─ feedback.py     (click/cart boosts)   │
                 │          ├─ confidence filter (relevance gate)    │
                 │          └─ [MMR + cross-encoder rerank — wired   │
                 │               in but OFF by default, see below]  │
                 │    └─▶ cross_sell.py  (on add-to-cart / top result)│
                 │    └─▶ history_logger.py (logs every search)      │
                 └───────────────────────────────────────────────────┘
```

### Pipeline stages

1. **Chunking** (`chunking.py`) — one product row → one chunk (`{id, text,
   metadata}`), built from `final_products.csv`. Handles real-world
   messiness in the source data (HTML in descriptions, truncated text,
   missing fields).
2. **BM25 index** (`bm25_index.py`) — builds the keyword index from the
   chunks. CPU-only, offline, run once per catalogue change.
3. **Embedding index** — every chunk encoded with the fine-tuned
   `muskannnnn/Prototype` model into `embeddings.npy` +
   `embedding_ids.json`. This runs on a Kaggle GPU notebook rather than
   locally — see [Generating embeddings](#generating-embeddings-kaggle-gpu).
4. **Query normalization** (`urdu_normalizer.py`) — expands roman-urdu
   words / known misspellings in the query to their canonical English
   term (exact dictionary match, then a fuzzy fallback) before BM25/
   vector search run, so a query like `"kheema masala"` also matches
   products indexed under `"qeema"`. See `testqueries.md` for the kind
   of queries this is built to handle.
5. **Hybrid retrieval** (`retrieval.py`) — the core pipeline:
   - Stage 1: BM25 + vector search run independently, then combined with
     **Weighted Reciprocal Rank Fusion (WRRF)**.
   - A **confidence filter** (vector cosine similarity threshold) is
     applied to every fused candidate, not just the top result — this is
     the primary relevance mechanism in the current default config.
   - Stage 2 (**MMR diversity** + **cross-encoder reranking**) is fully
     implemented and wired in, but toggled OFF by default
     (`config.RETRIEVAL.mmr_lambda=1.0`, `use_reranker=False`) — flip
     those in `config.py` to turn them back on without touching the
     retrieval code.
   - Metadata filters (in_stock, category, brand, price range) and a
     query-result cache (TTL-based) are also handled here.
6. **Feedback loop** (`feedback.py`) — clicks, add-to-carts, and manual
   relevant/irrelevant labels (given while reviewing `/api/history`) boost
   or hard-exclude products for a query and similar future queries
   (via embedding similarity).
7. **Cross-sell/upsell** (`cross_sell.py`) — "customers who bought X also
   bought Y", built from `order_history.csv`
   (`bought_percent = orders / frequency`), restricted to the active
   product catalogue.
8. **API** (`api.py`) — FastAPI app exposing search, autocomplete,
   cross-sell, feedback, and history-review endpoints.
9. **Frontend** (`frontend/index.html`) — a single-page search UI: search
   bar with autocomplete, in-stock/category/brand/price filters, a
   product grid, and both cross-sell surfaces (related to the top search
   result, and triggered by "Add to Cart").

## How to run this project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the BM25 index (local, CPU — fast)

```bash
python src/chunking.py
python src/bm25_index.py
```

This (re)generates `indexes/prototype/chunks.jsonl` and
`indexes/prototype/bm25_index.pkl` from `final_products.csv`. Skip this
step if those files are already current for the catalogue you're using —
`indexes/prototype/` already has both checked in.

### 3. Embeddings

`indexes/prototype/embeddings.npy` and `embedding_ids.json` are already
built and checked in. Only regenerate them (see
[Generating embeddings](#generating-embeddings-kaggle-gpu) below) if the
catalogue or the fine-tuned model changes.

### 4. Run the API

```bash
python src/api.py
# or: uvicorn api:app --reload --app-dir src
```

The API starts on `http://localhost:8000`. First request to `/api/search`
takes a few seconds — that's `retrieval.py` lazy-loading the BM25 index,
the embeddings, and the `muskannnnn/Prototype` model (downloaded from the
Hugging Face Hub once, then cached locally).

### 5. Serve the frontend

`frontend/index.html` has `API_URL` hardcoded to
`http://localhost:8000/api` — make sure the API is running first. It also
fetches `image_mapping.json` with a relative path, so serve `frontend/`
itself as the web root:

```bash
cd frontend
python -m http.server 5500
# then open http://localhost:5500/index.html
```

## Fine-tuning

The embedding model (`muskannnnn/Prototype`, a fine-tuned `BAAI/bge-m3`) was
arrived at in two stages:

1. **Pilot, to check fine-tuning was worth doing at all.** Once the
   prototype catalogue was narrowed down to 2,130 products, a small
   500-product / 2,500-row subset was used first — 5 roman-urdu/misspelling
   query variations per product — and 5 different LoRA hyperparameter
   configurations (rank, learning rate, batch size, epoch count) were
   trained and compared against a held-out eval set. This run showed a
   clear MRR/Hit-rate gain over the base (non-fine-tuned) `bge-m3`, which is
   what justified scaling up rather than shipping the base model as-is.
2. **Full run, at scale.** The best-performing configuration from the pilot
   was carried forward and re-trained on the full 10,640-pair /
   6,245-unique-anchor dataset, covering all 2,130 products in the
   prototype catalogue — this is the run that produced the checkpoint
   uploaded to `muskannnnn/Prototype`.

All four notebooks live under `finetuning/`:

- **`Notebooks(Output)/FineTuning.ipynb`** and
  **`Notebooks(Output)/EmbeddingGeneration.ipynb`** — the original notebooks
  as actually run, output included (training logs, MRR/Hit-rate numbers,
  embedding shapes).
- **`Notebooks(Cleaned)/Finetuning(cleaned).ipynb`** — the same fine-tuning
  code, with markdown explaining what each step is doing and why (LoRA
  injection, the manual merge-after-`.fit()` step, the save-verification
  checks, the MRR@20/Hit@20 evaluation methodology), so the training run can
  be understood and replicated without reverse engineering the raw notebook.
- **`Notebooks(Cleaned)/EmbeddingGeneration(cleaned).ipynb`** — a clean
  notebook that (re-)generates `embeddings.npy` + `embedding_ids.json` from
  `muskannnnn/Prototype` on the Hub. Use this whenever the catalogue or the
  fine-tuned model changes — see the next section for the details of what
  it does and where its output goes.

## Generating embeddings (Kaggle GPU)

Encoding the full catalogue is too slow on CPU, so embedding generation
runs as a Kaggle notebook rather than locally —
`finetuning/Notebooks(Cleaned)/EmbeddingGeneration(cleaned).ipynb` is ready
to use as-is; the steps below are what it does:

1. Go to [kaggle.com](https://www.kaggle.com) → **New Notebook**.
2. **Settings → Accelerator → GPU** (T4 x2 or P100).
3. Upload `indexes/prototype/chunks.jsonl` as a Kaggle **Dataset** and
   add it to the notebook's inputs (or copy its rows in directly).
4. Run:

```python
!pip install -q sentence-transformers

import json
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "/kaggle/input/<your-dataset-name>/chunks.jsonl"

def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

chunks = load_chunks(CHUNKS_PATH)
ids = [c["id"] for c in chunks]
texts = [c["text"] for c in chunks]

# Loads the fine-tuned model from the Hugging Face Hub —
# https://huggingface.co/muskannnnn/Prototype
model = SentenceTransformer("muskannnnn/Prototype", device="cuda")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,   # L2-normalize so dot product = cosine similarity
    convert_to_numpy=True,
).astype(np.float32)

np.save("embeddings.npy", embeddings)
with open("embedding_ids.json", "w", encoding="utf-8") as f:
    json.dump(ids, f, ensure_ascii=False)

print(f"Saved {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
```

5. Download `embeddings.npy` and `embedding_ids.json` from the notebook's
   **Output** panel and place them in `indexes/prototype/`, replacing the
   existing files.

`ids` must stay positionally aligned with the rows of `embeddings` —
`retrieval.py` relies on that alignment (`embed_id_to_idx`), which is why
the notebook builds both from the same `chunks` list in one pass.

`config.EMBEDDING_MODEL_NAME` (default `"muskannnnn/Prototype"`, loaded
from the Hugging Face Hub) is the single place this repo id is set — the
same value is used both by this notebook and by `retrieval.py` for
query-time encoding, so they stay in the same embedding space. A local
copy of the same model also lives in `models/model_run_6/`; override
`EMBEDDING_MODEL_NAME` via env var if you'd rather point at that instead
of the Hub.

## API endpoints

- `GET /api/search?q=...` — hybrid search with optional filters
  (`in_stock`, `category`, `brand`, `min_price`, `max_price`)
- `GET /api/autocomplete?q=...` — fast, low-latency suggestions (reranker
  always skipped)
- `GET /api/cross-sell?sku=...` — "customers who bought this also bought"
- `POST /api/feedback` — log a click/add_to_cart/relevant/irrelevant event
- `GET /api/history` — review what every past query actually returned
- `GET /api/feedback/history` — review logged feedback events
- `GET /api/health` — health check

All retrieval hyperparameters (WRRF alpha, MMR lambda, reranker toggle,
confidence thresholds, cache size, etc.) live in `config.RETRIEVAL` — the
API never accepts them as client-tunable knobs.

## Example queries

`testqueries.md` is a running list of queries used to sanity-check
search, mostly Roman-Urdu spellings, spelling variants of the same word,
and a couple of full-sentence queries — the kind of input
`urdu_normalizer.py` and the confidence filter are built to handle
gracefully rather than returning nothing:

```
lal lobia / laal lobiya      — same word, two spellings
dar cheeni                   — cinnamon
zafran / safran               — saffron, two spellings
nariyal dudh                  — coconut milk
teekhi hari chutney            — spicy green chutney
kache aam ka achaar            — raw mango pickle
mujhe nhari masla chahiye      — full-sentence query ("I need nihari masala")
pyaaz pouder                   — "onion powder", misspelled
```