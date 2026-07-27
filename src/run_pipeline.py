"""
End-to-End Pipeline Batch Runner
--------------------------------
Runs a batch of queries through the entire Naheed chatbot architecture:
Query → BM25 + Vector → RRF → MMR → Cross-Encoder → LLM Generation → LLM Evaluation.
Logs the results of every step.
"""

import time
from generation import generate
from evaluation import evaluate
from evaluation import preload_evaluation_models
from history_logger import log_query_trace

# 50 Test Queries covering English, Roman Urdu, exact brands, and fuzzy needs.
# Expand or adjust this list based on the specific edge cases you want to grade.
QUERIES = [
    "mujhe dee liish ke shami kebab chahiye",
    "yngs myonise",
    "schr fir di sl bscuits",
    "mtchll",
    "rssmrr mxd",
    "nheed dal hari mash",
    "haynz nachral",
    "aaoidized namak",
    "syprsa kuin",
    "lal mrch pouder"
]

def run_batch():
    print("=" * 60)
    preload_evaluation_models()
    print(f"Starting End-to-End Pipeline Batch Test ({len(QUERIES)} queries)")
    print("=" * 60)
    
    total_t0 = time.perf_counter()
    
    for i, query in enumerate(QUERIES, start=1):
        print(f"\n[{i}/{len(QUERIES)}] Processing: '{query}'")
        
        try:
            # Step 1 & 2: Retrieval + Generation
            # (generate() automatically calls retrieval.search() internally)
            print("  -> Retrieving and Generating...")
            
            # --- UPDATED: CONDITIONAL RERANKING ENABLED ---
            gen_result = generate(
                query, 
                top_k=5, 
                use_reranker=True,           # Must be True to allow Stage 2 routing
                conditional_rerank=True,     # Enable the smart skip logic
                confidence_threshold=1.0    # Use the more aggressive 35% skip threshold
            )
            
            # Print immediate feedback on the routing decision
            reranker_time = gen_result.retrieved.timings.get("reranker", 0.0)
            if reranker_time == 0.0:
                print("  ⚡ Routing: SKIPPED Cross-Encoder (High Confidence)")
            else:
                print(f"  🔍 Routing: EXECUTED Cross-Encoder (Low Confidence) in {reranker_time:.2f}s")

            # Step 3: Evaluation
            print("  -> Evaluating Faithfulness & Relevancy...")
            eval_result = evaluate(
                query=query,
                response=gen_result.response,
                context=gen_result.context,
                n_questions=3
            )
            
            # Step 4: Logging
            log_query_trace(query, gen_result, eval_result)
            
            print(f"  ✓ Done. Faithfulness: {eval_result.faithfulness.score:.2f} | Relevancy: {eval_result.relevancy.score:.2f} | Relevancy: {eval_result.retrieval.:.2f}")
            print(f"  ⏱ Pipeline time: {sum(gen_result.timings.values()):.2f}s")
            
        except Exception as e:
            print(f"  ❌ FAILED on query '{query}': {str(e)}")
        
        # Reduced from 10 to 1 to speed up local batch execution
        time.sleep(1)
            
    total_time = time.perf_counter() - total_t0
    print("\n" + "=" * 60)
    print(f"Batch test complete! Total time: {total_time/60:.1f} minutes.")
    print("Check `logs/pipeline_history.jsonl` for full execution traces.")

if __name__ == "__main__":
    run_batch()