"""
Build frontend/image_mapping.json — SKU -> real product image URL.

The prototype catalogue (indexes/prototype/chunks.jsonl, 2130 products) has
no image data of its own. The old mapping guessed a local file under
frontend/images/ from a numeric id, which is wrong for a lot of products.

This script instead looks up each of the 2130 SKUs in data/products_full.jsonl
(the full catalogue dump, which carries a correct `image_url` per product)
and writes a direct SKU -> CDN image URL mapping. The frontend fetches this
file and uses the URL as-is, no local image files involved.

Run whenever the active catalogue (CATALOGUE_MODE) or products_full.jsonl
changes:
    python src/build_image_mapping.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import CHUNKS_PATH, PRODUCTS_FULL_PATH, IMAGE_MAPPING_PATH


def load_catalogue_skus(chunks_path: str) -> set[str]:
    skus = set()
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            skus.add(json.loads(line)["id"])
    return skus


def load_image_urls(products_full_path: str, wanted_skus: set[str]) -> dict[str, str]:
    sku_to_url = {}
    with open(products_full_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sku = obj.get("sku")
            image_url = obj.get("image_url")
            if sku in wanted_skus and image_url:
                sku_to_url[sku] = image_url
    return sku_to_url


def main():
    catalogue_skus = load_catalogue_skus(CHUNKS_PATH)
    print(f"Catalogue ({CHUNKS_PATH}): {len(catalogue_skus)} products")

    sku_to_url = load_image_urls(PRODUCTS_FULL_PATH, catalogue_skus)
    print(f"Matched image URLs: {len(sku_to_url)}")

    missing = catalogue_skus - sku_to_url.keys()
    if missing:
        print(f"No image found for {len(missing)} SKUs (frontend will fall back to placeholder):")
        for sku in sorted(missing)[:20]:
            print(f"  - {sku}")

    Path(IMAGE_MAPPING_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(IMAGE_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(sku_to_url, f, indent=2, sort_keys=True)

    print(f"Wrote {len(sku_to_url)} entries to {IMAGE_MAPPING_PATH}")


if __name__ == "__main__":
    main()
