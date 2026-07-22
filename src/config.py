"""
Shared paths and constants for the retrieval pipeline.
Keeping these in one place so chunking / bm25 / embeddings / retrieval
all agree on where things live.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")

# The real, full product dump from the mentor team.
CATALOGUE_PATH = os.path.join(RAW_DIR, "AllProducts.csv")

# Small hand-made catalogue kept around for fast smoke tests / demos
# that don't need the full 155k-row file.
SAMPLE_CATALOGUE_PATH = os.path.join(SAMPLES_DIR, "sample_catalogue.csv")

CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25_index.pkl")
EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
EMBED_IDS_PATH = os.path.join(INDEX_DIR, "embedding_ids.json")

# Embedding model. BGE-M3 is multilingual (100+ languages incl. Urdu) and
# handles mixed-script text in a single embedding space, which is why it's
# a good fit for Urdu-English mixed queries.
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMBEDDING_MODEL_NAME = os.path.join(MODELS_DIR, "model_run_6")
# Cross-encoder / reranker model. BGE-reranker-v2-m3 is the multilingual
# reranker from the same BGE family — handles Roman Urdu + English joint
# scoring without needing a separate translation step.
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Generation LLM ("LLM1" in Unit 10's pipeline diagram — smaller/faster
# model that generates the actual customer-facing answer).
GENERATION_MODEL_NAME = "llama-3.1-8b-instant"
EVALUATION_MODEL_NAME = "llama-3.3-70b-versatile"

import os
from dotenv import load_dotenv

# Find the .env file in the root directory and load its contents
# By default, it looks in the same directory or traverses up
load_dotenv() 

# Assign the keys to Python variables
# config.py
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Optional: Add a quick sanity check so the script fails early if the key is missing
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

os.makedirs(INDEX_DIR, exist_ok=True)
