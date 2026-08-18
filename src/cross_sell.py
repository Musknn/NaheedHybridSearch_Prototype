"""
Cross-Sell / Upsell Recommendations
--------------------------------------
"When product X is added to cart, what should we suggest alongside it?"

Built from the SAME order_history co-purchase table as popularity.py, but
this is a much better fit for that table's actual shape: `bought_percent`
is literally defined as "of the orders containing product X, what % also
contained Y" -- that IS the definition of cross-sell likelihood, not a
proxy for it.

    bought_percent = orders / frequency   (verified on the real sample:
        row1: 2/3=66.67%, row2: 2/14=14.29% -- exact match)

Symmetry check (verified across all 40 pairs in the sample data, not
just spot-checked ones): bought_percent is IDENTICAL regardless of which
side of the pair is treated as the anchor.

CATALOG RESTRICTION: since the prototype only serves 2130 products
(final_products.csv / chunks.jsonl), any co-purchase pair involving a
SKU outside that set is dropped automatically via the catalog_skus filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from chunking import load_chunks
from config import CHUNKS_PATH, CROSS_SELL, ORDER_HISTORY_PATH


@dataclass(frozen=True)
class CrossSellSuggestion:
    sku: str
    name: str
    brand: str
    price: float | None
    in_stock: bool
    bought_percent: float
    orders: int


@dataclass(frozen=True)
class CrossSellIndex:
    adjacency: dict[str, list[tuple[str, float, int]]]
    catalog_skus: frozenset[str]


def _dedupe_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["sku", "related_product_sku"]).copy()
    df["sku"] = df["sku"].astype(str).str.strip()
    df["related_product_sku"] = df["related_product_sku"].astype(str).str.strip()
    df["_pair_key"] = df.apply(
        lambda r: tuple(sorted([r["sku"], r["related_product_sku"]])), axis=1
    )
    return df.drop_duplicates(subset=["_pair_key"], keep="first")


def _load_catalog_skus() -> frozenset[str]:
    chunks = load_chunks(CHUNKS_PATH)
    return frozenset(c["id"] for c in chunks)


@lru_cache(maxsize=1)
def _get_chunks_by_id() -> dict[str, dict]:
    chunks = load_chunks(CHUNKS_PATH)
    return {c["id"]: c for c in chunks}


@lru_cache(maxsize=1)
def build_cross_sell_index(order_history_path: str = ORDER_HISTORY_PATH) -> CrossSellIndex:
    catalog_skus = _load_catalog_skus()
    df = pd.read_csv(order_history_path)
    deduped = _dedupe_pairs(df)

    adjacency: dict[str, list[tuple[str, float, int]]] = {}
    for _, row in deduped.iterrows():
        a, b = row["sku"], row["related_product_sku"]
        if a not in catalog_skus or b not in catalog_skus:
            continue

        pct = float(row["bought_percent"])
        orders = int(row["orders"])
        if pct < CROSS_SELL.min_bought_percent or orders < CROSS_SELL.min_orders:
            continue

        adjacency.setdefault(a, []).append((b, pct, orders))
        adjacency.setdefault(b, []).append((a, pct, orders))

    for sku in adjacency:
        adjacency[sku].sort(key=lambda x: (-x[1], -x[2]))

    return CrossSellIndex(adjacency=adjacency, catalog_skus=catalog_skus)


def get_cross_sell(
    sku: str,
    top_n: int | None = None,
    in_stock_only: bool | None = None,
) -> list[CrossSellSuggestion]:
    top_n = top_n if top_n is not None else CROSS_SELL.default_top_n
    in_stock_only = in_stock_only if in_stock_only is not None else CROSS_SELL.in_stock_only

    index = build_cross_sell_index()
    chunks_by_id = _get_chunks_by_id()

    suggestions: list[CrossSellSuggestion] = []
    for related_sku, pct, orders in index.adjacency.get(sku, []):
        chunk = chunks_by_id.get(related_sku)
        if not chunk:
            continue
        meta = chunk["metadata"]
        if in_stock_only and not meta.get("in_stock", False):
            continue

        suggestions.append(
            CrossSellSuggestion(
                sku=related_sku,
                name=meta.get("name", ""),
                brand=meta.get("brand", ""),
                price=meta.get("price"),
                in_stock=meta.get("in_stock", False),
                bought_percent=pct,
                orders=orders,
            )
        )
        if len(suggestions) >= top_n:
            break

    return suggestions


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cross_sell.py <sku> [top_n]")
        sys.exit(1)

    sku = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    results = get_cross_sell(sku, top_n=top_n)
    if not results:
        print(f"No qualifying cross-sell data for {sku!r}")
    else:
        print(f"Customers who bought {sku!r} also bought:\n")
        for s in results:
            stock = "in stock" if s.in_stock else "OUT OF STOCK"
            price_str = f"Rs.{s.price:,.0f}" if s.price else "N/A"
            print(f"  [{s.sku}] {s.name}  ({s.brand})  {price_str}  {stock}")
            print(f"      bought together in {s.bought_percent:.1f}% of orders (n={s.orders})")
