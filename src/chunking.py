"""
STEP 1: Chunking (real catalogue version)
-------------------------------------------
Same principle as before: one product row = one chunk, no text splitting
needed. This version is adapted to the ACTUAL columns in npk_prods_dmp.csv
and handles the real-world messiness that a hand-made sample never has:

1. HTML in `description` / `short_description` - stripped with a regex.
2. `description` is truncated at exactly 255 characters at the data
   source (confirmed: 75% of non-null descriptions are exactly 255 chars
   long, often mid-sentence). We can't recover the missing text, so we
   just use what's there - it's still a useful signal, just incomplete.
3. ~103k of 155k rows (two-thirds) have NO description or short_description
   at all. For those, the chunk text falls back to name + brand +
   category hierarchy only. This is expected, not a bug - don't be
   surprised when many chunks are short.
4. `category_tag` is populated for only ~1,500 rows (a legacy/partial
   field) - we use `parent_category` + `category_hierarchy` instead,
   which are populated for all but ~2,300 rows.
5. `category_hierarchy` looks like "Health & Beauty > Hair Care >
   Shampoo & Conditioner" - the ">" separators are replaced with spaces
   so each level becomes a plain searchable term.
6. Stock: `quantity` is the reliable overall stock figure (never
   negative). The four per-warehouse columns (kokon_pharmacy_qty,
   bahadurabad_qty, malir_qty, korangi_qty) sometimes contain negative
   values (a source data-quality quirk, not something to silently
   "correct" - we keep the raw values in metadata for debugging rather
   than hiding the issue).
7. `sku` (e.g. "IC-1143194") is used as the chunk id - it's unique across
   all 155k rows and human-readable, unlike `id` or `product_id`.

IMPORTANT GAP TO FLAG: this real dump has NO Urdu-alias column like our
sample catalogue did. That means the "sons ka pampers" style matching
we tested earlier will NOT work as well out of the box here - there's
no "bache ka pamper" text anywhere in this data for BM25 to match
against. This is a real problem to solve in Step 2 (retrieval), most
likely by normalizing/translating the query with an LLM before search,
rather than something Step 1 alone can fix. Flagging so it doesn't
get lost.
"""
import json
import re

import pandas as pd

from config import CATALOGUE_PATH, CHUNKS_PATH

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def _safe_str(value) -> str:
    """pd.NA/NaN floats are truthy in Python, so `value or ''` doesn't
    catch them - need an explicit isna check."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def strip_html(value) -> str:
    if not isinstance(value, str):
        return ""
    text = _HTML_TAG.sub(" ", value)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def load_catalogue(path: str = CATALOGUE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df


def build_chunk_text(row: pd.Series) -> str:
    name = _safe_str(row.get("name"))
    brand = _safe_str(row.get("brand"))
    parent_category = _safe_str(row.get("parent_category"))
    category_hierarchy = _safe_str(row.get("category_hierarchy")).replace(">", " ")
    short_desc = strip_html(row.get("short_description"))
    desc = strip_html(row.get("description"))

    parts = [
        name, name,                             # weighted x2
        brand,
        parent_category, parent_category,       # weighted x2
        category_hierarchy,
        short_desc,
        desc,
    ]
    return " . ".join(p for p in parts if p)


def _safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_chunks(df: pd.DataFrame) -> list[dict]:
    chunks = []
    for _, row in df.iterrows():
        quantity = _safe_int(row.get("quantity"), 0)
        warehouse_stock = {
            "kokon_pharmacy": _safe_int(row.get("kokon_pharmacy_qty")),
            "bahadurabad": _safe_int(row.get("bahadurabad_qty")),
            "malir": _safe_int(row.get("malir_qty")),
            "korangi": _safe_int(row.get("korangi_qty")),
        }
        chunk = {
            "id": _safe_str(row.get("sku")),
            "text": build_chunk_text(row),
            "metadata": {
                "name": _safe_str(row.get("name")),
                "brand": _safe_str(row.get("brand")),
                "category": _safe_str(row.get("parent_category")),
                "category_path": [p.strip() for p in _safe_str(row.get("category_hierarchy")).split(">") if p.strip()],
                "price": _safe_float(row.get("price")),
                "quantity": quantity,
                "in_stock": quantity > 0,
                "warehouse_stock": warehouse_stock,
                "picking_mode": _safe_str(row.get("picking_mode")),
                "url_key": _safe_str(row.get("url_key")),
            },
        }
        chunks.append(chunk)
    return chunks


def save_chunks(chunks: list[dict], path: str = CHUNKS_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def load_chunks(path: str = CHUNKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

if __name__ == "__main__":
    df = load_catalogue()
    print(f"Loaded {len(df)} rows from {CATALOGUE_PATH}")

    chunks = build_chunks(df)
    save_chunks(chunks)
    print(f"Built {len(chunks)} chunks -> {CHUNKS_PATH}")

    no_desc = df["short_description"].isna().sum()
    no_desc_and_short = (df["short_description"].isna() & df["description"].isna()).sum()
    print(f"Rows missing short_description: {no_desc} ({no_desc/len(df):.1%})")
    print(f"Rows missing BOTH descriptions (chunk = name/brand/category only): "
          f"{no_desc_and_short} ({no_desc_and_short/len(df):.1%})")

    print("\nExample chunk (has description):")
    example = next(c for c in chunks if len(c["text"]) > 100)
    print(json.dumps(example, indent=2, ensure_ascii=False)[:800])