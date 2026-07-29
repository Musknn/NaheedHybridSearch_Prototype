"""
3-Way Benchmark Comparison Script
----------------------------------
Compares three retrieval architectures:
  1. Pure Reranker ON   (Always run Cross-Encoder)
  2. Pure Reranker OFF  (Pure Stage 1 WRRF)
  3. Conditional Rerank (Run Cross-Encoder ONLY on low-confidence queries)

Outputs timing, relevancy, faithfulness, and conditional skip metrics.
"""

import json
import time
from evaluation import evaluate, preload_evaluation_models
from generation import generate

QUERIES = [
    "sabzi miks ready made",
    "mayonez wala",
    "hosen bamboo shoot",
    "mtchll s",
    "rssmrr mxd",
    "chli kl mrch",
    "frutna frut cktl",
    "vgtble mks",
    "de lish",
    "saas",
]


def run_benchmark():
    print("=" * 80)
    print("INITIALIZING 3-WAY RETRIEVAL BENCHMARK")
    print("Modes: [1] Pure ON  |  [2] Pure OFF  |  [3] Conditional Rerank")
    print("=" * 80)

    # Pre-load evaluation models once to avoid cold-start bias
    preload_evaluation_models()

    summary = []

    for i, q in enumerate(QUERIES, start=1):
        print(f"\n[{i}/{len(QUERIES)}] Query: '{q}'")
        print("-" * 65)

        # ── Mode 1: Pure Reranker ON ──
        print("  [1/3] Running Pure Reranker ON...")
        t0 = time.perf_counter()
        res_on = generate(
            q, top_k=5, use_reranker=True, conditional_rerank=False
        )
        eval_on = evaluate(q, res_on.response, res_on.context)
        time_on = time.perf_counter() - t0
        top_on = (
            res_on.retrieved.results[0].name
            if res_on.retrieved.results
            else "None"
        )

        # ── Mode 2: Pure Reranker OFF ──
        print("  [2/3] Running Pure Reranker OFF...")
        t0 = time.perf_counter()
        res_off = generate(
            q, top_k=5, use_reranker=False, conditional_rerank=False
        )
        eval_off = evaluate(q, res_off.response, res_off.context)
        time_off = time.perf_counter() - t0
        top_off = (
            res_off.retrieved.results[0].name
            if res_off.retrieved.results
            else "None"
        )

        # ── Mode 3: Conditional Reranking ──
        print("  [3/3] Running Conditional Rerank...")
        t0 = time.perf_counter()
        res_cond = generate(
            q, top_k=5, use_reranker=True, conditional_rerank=True
        )
        eval_cond = evaluate(q, res_cond.response, res_cond.context)
        time_cond = time.perf_counter() - t0
        top_cond = (
            res_cond.retrieved.results[0].name
            if res_cond.retrieved.results
            else "None"
        )

        # Verify whether the Cross-Encoder was skipped or executed in Mode 3
        reranker_time_cond = res_cond.retrieved.timings.get("reranker", 0.0)
        was_skipped = reranker_time_cond == 0.0

        # Construct benchmark record
        record = {
            "query": q,
            "reranker_on": {
                "total_time": round(time_on, 2),
                "retrieval_time": round(res_on.timings.get("retrieval", 0), 2),
                "faithfulness": eval_on.faithfulness.score,
                "relevancy": eval_on.relevancy.score,
                "top_product": top_on,
            },
            "reranker_off": {
                "total_time": round(time_off, 2),
                "retrieval_time": round(
                    res_off.timings.get("retrieval", 0), 2
                ),
                "faithfulness": eval_off.faithfulness.score,
                "relevancy": eval_off.relevancy.score,
                "top_product": top_off,
            },
            "conditional": {
                "total_time": round(time_cond, 2),
                "retrieval_time": round(
                    res_cond.timings.get("retrieval", 0), 2
                ),
                "faithfulness": eval_cond.faithfulness.score,
                "relevancy": eval_cond.relevancy.score,
                "top_product": top_cond,
                "skipped_reranker": was_skipped,
            },
        }
        summary.append(record)

        cond_action = "SKIPPED ⚡" if was_skipped else "EXECUTED 🔍"
        print(
            f"  [PURE ON]   Time: {time_on:5.1f}s | Rel: {eval_on.relevancy.score:.2f} | Top: {top_on[:30]}"
        )
        print(
            f"  [PURE OFF]  Time: {time_off:5.1f}s | Rel: {eval_off.relevancy.score:.2f} | Top: {top_off[:30]}"
        )
        print(
            f"  [COND]      Time: {time_cond:5.1f}s | Rel: {eval_cond.relevancy.score:.2f} | [{cond_action}] Top: {top_cond[:30]}"
        )

    # ── SUMMARY REPORT ──
    print("\n" + "=" * 90)
    print("3-WAY RETRIEVAL BENCHMARK SUMMARY")
    print("=" * 90)
    print(
        f"{'Query':<20} | {'ON (Time/Rel)':<15} | {'OFF (Time/Rel)':<15} | {'COND (Time/Rel)':<15} | {'Action'}"
    )
    print("-" * 90)

    for s in summary:
        q_short = s["query"][:18]
        on_str = f"{s['reranker_on']['total_time']}s / {s['reranker_on']['relevancy']:.2f}"
        off_str = f"{s['reranker_off']['total_time']}s / {s['reranker_off']['relevancy']:.2f}"
        cond_str = f"{s['conditional']['total_time']}s / {s['conditional']['relevancy']:.2f}"
        action_str = (
            "SKIPPED ⚡"
            if s["conditional"]["skipped_reranker"]
            else "EXECUTED 🔍"
        )

        print(
            f"{q_str:<20} | {on_str:<15} | {off_str:<15} | {cond_str:<15} | {action_str}"
        )

    print("-" * 90)

    with open("benchmark_3way_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nDetailed benchmark saved to `benchmark_3way_report.json`.")


if __name__ == "__main__":
    run_benchmark()