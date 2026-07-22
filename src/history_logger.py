"""
Pipeline History Logger
-----------------------
Captures the complete trace of a query as it travels through retrieval,
generation, and evaluation, and appends it to a JSONL log file for 
debugging and reporting.
"""

import os
import json
from datetime import datetime
from config import BASE_DIR

# Ensure a logs directory exists in the project root
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "pipeline_history.jsonl")

def log_query_trace(query: str, gen_result, eval_result) -> dict:
    """
    Extracts the pipeline execution trace and appends it to the log file.
    
    Args:
        query: The original user string.
        gen_result: The GenerationResult object from generation.py
        eval_result: The EvaluationResult object from evaluation.py
    """
    trace = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "execution_timings_seconds": gen_result.timings,
        "retrieval_diagnostics": {
            "total_candidates_after_rrf": gen_result.retrieved.total_candidates,
            "final_results_returned": len(gen_result.retrieved.results),
            "top_products": [
                {
                    "rank": r.rank,
                    "id": r.id,
                    "name": r.name,
                    "brand": r.brand,
                    "reranker_score": r.score,
                    "in_stock": r.in_stock
                } for r in gen_result.retrieved.results
            ]
        },
        "generation_trace": {
            "grounding_context": gen_result.context,
            "llm_response": gen_result.response
        },
        "evaluation_metrics": {
            "faithfulness": eval_result.faithfulness.score if eval_result.faithfulness else None,
            "relevancy": eval_result.relevancy.score if eval_result.relevancy else None
        }
    }
    
    # Append safely to the JSONL file
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        
    return trace