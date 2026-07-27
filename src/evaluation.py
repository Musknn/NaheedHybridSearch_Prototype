"""
STEP 3: RAG Evaluation — Manual Implementation
-------------------------------------------------
Hand-built faithfulness and relevancy metrics following the methodology
from Unit 10 (Sajjad Haider's NLP with Deep Learning, Spring 2026).

This is NOT a black-box RAGAS call. We implement each step explicitly so
you understand the internals:

Faithfulness:
  1. Extract factual claims from the LLM response
  2. Verify each claim against the retrieved context
  3. Score = supported_claims / total_claims

Relevancy:
  1. Generate N questions from the response
  2. Embed them + embed the original query
  3. Score = mean cosine similarity between generated questions and original query

LLM Integration:
  The `call_llm()` function is a stub — plug in your Tiny-Aya endpoint,
  an OpenAI API, or any other LLM. The prompts are written to work with
  small instruction-following models.

Usage:
    from evaluation import evaluate_faithfulness, evaluate_relevancy

    faith = evaluate_faithfulness(
        response="Einstein was born in Germany on 14 March 1879.",
        context="Albert Einstein (born 14 March 1879) was a German-born physicist."
    )
    print(faith.score)  # 1.0

    rel = evaluate_relevancy(
        query="Where was Einstein born?",
        response="Einstein was born in Germany on 14 March 1879."
    )
    print(rel.score)  # ~0.9
"""

from __future__ import annotations

import re
import time
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from retrieval import get_embed_model as _get_shared_embed_model
from config import EMBEDDING_MODEL_NAME, EVALUATION_MODEL_NAME
import llm_client

# ═══════════════════════════════════════════════════════════════════════════
# Pydantic models for evaluation results
# ═══════════════════════════════════════════════════════════════════════════


class ClaimVerification(BaseModel):
    """Result of verifying a single claim against the context."""

    claim: str
    verdict: str = Field(description="SUPPORTED, UNSUPPORTED, or IMPLICIT")
    is_supported: bool = Field(description="True if verdict is SUPPORTED or IMPLICIT")


class FaithfulnessResult(BaseModel):
    """Complete faithfulness evaluation result."""

    score: float = Field(ge=0.0, le=1.0, description="Fraction of supported claims")
    total_claims: int
    supported_claims: int
    claims: list[ClaimVerification]
    response: str
    context: str


class RelevancyResult(BaseModel):
    """Complete relevancy evaluation result."""

    score: float = Field(ge=0.0, le=1.0, description="Mean cosine similarity of generated questions to original query")
    generated_questions: list[str]
    similarities: list[float] = Field(description="Cosine similarity for each generated question")
    query: str
    response: str


class RetrievalMetricsResult(BaseModel):
    """
    Hit@K / MRR@K for a single query, when a known-correct product id is
    available (e.g. from held_out_eval_v2.csv).

    Unlike faithfulness/relevancy, this needs no LLM call and no embedding
    call — it's a pure lookup against retrieved_ids, so it carries none of
    the temperature=0.3 sampling noise that affects the other two metrics.
    Use this whenever ground truth exists; fall back to faithfulness/
    relevancy for live queries where there's no known-correct answer.
    """

    hit: bool = Field(description="True if expected_id appears anywhere in retrieved_ids")
    reciprocal_rank: float = Field(ge=0.0, le=1.0, description="1/rank if found (1-indexed), else 0.0")
    rank: Optional[int] = Field(default=None, description="1-indexed position of expected_id, if found")
    expected_id: str
    retrieved_ids: list[str]
    top_k: int = Field(description="len(retrieved_ids) — i.e. K in Hit@K/MRR@K")


class EvaluationResult(BaseModel):
    """Combined evaluation result."""

    faithfulness: Optional[FaithfulnessResult] = None
    relevancy: Optional[RelevancyResult] = None
    retrieval: Optional[RetrievalMetricsResult] = None


# ═══════════════════════════════════════════════════════════════════════════
# LLM Integration (STUB — replace with your actual LLM)
# ═══════════════════════════════════════════════════════════════════════════


def call_llm(prompt: str) -> str:
    """
    Call the evaluation LLM ("LLM2" in Unit 10's pipeline diagram) and
    return its response.

    Delegates to llm_client.call_llm(), pointed at config.EVALUATION_MODEL_NAME
    rather than config.GENERATION_MODEL_NAME. Per the lecture, the evaluation
    model is meant to be LARGER/more capable than the generation model (a
    model shouldn't be its own judge) — see config.py for how to point this
    at a different model than generation.py uses.

    Uses temperature=0.0 (greedy decoding) rather than generation.py's
    default: claim extraction/verification and question generation need
    consistent, easily-parsed output, not creative variation.
    """
    return llm_client.call_llm(
        prompt,
        model_name=EVALUATION_MODEL_NAME,
        max_new_tokens=200,
        temperature=0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Embedding model (lazy-loaded)
# ═══════════════════════════════════════════════════════════════════════════

# _embed_model: Optional[SentenceTransformer] = None

def _get_embed_model() -> SentenceTransformer:
    """Reuse retrieval.py's already-loaded embedding model rather than
    loading a second ~2.2GB copy for relevancy scoring."""
    return _get_shared_embed_model()

# Add this to evaluation.py

def preload_evaluation_models():
    """
    Force the loading of heavy evaluation models into memory.
    Call this ONCE before starting any batch process.
    """
    print("-> Pre-loading Evaluation Models...")
    # This will populate the _embed_model global cache
    _get_embed_model() 
    print("-> Evaluation models loaded and cached.")
# ═══════════════════════════════════════════════════════════════════════════
# Faithfulness: Claim Extraction
# ═══════════════════════════════════════════════════════════════════════════

# Prompt template for extracting factual claims from a response.
# Designed to work with small instruction-following models.
CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the answer below as a numbered list. \
Include only explicit factual statements that can be verified. \
Do not include opinions, greetings, or meta-comments. \
Output ONLY the numbered list with no other text.

Answer: {response}"""


def extract_claims(response: str) -> list[str]:
    """
    Use the LLM to extract factual claims from a response.

    Args:
        response: The LLM-generated answer to extract claims from.

    Returns:
        List of claim strings.
    """
    prompt = CLAIM_EXTRACTION_PROMPT.format(response=response)
    llm_output = call_llm(prompt)
    return _parse_numbered_list(llm_output)


def _parse_numbered_list(text: str) -> list[str]:
    """
    Parse a numbered list from LLM output.
    Handles formats like:
      1. Claim text
      1) Claim text
      1: Claim text
    """
    lines = text.strip().split("\n")
    items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove numbering prefix: "1. ", "1) ", "1: ", "- ", "* "
        cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", line)
        cleaned = re.sub(r"^[\-\*]\s*", "", cleaned)
        if cleaned:
            items.append(cleaned)
    return items


# ═══════════════════════════════════════════════════════════════════════════
# Faithfulness: Claim Verification
# ═══════════════════════════════════════════════════════════════════════════

CLAIM_VERIFICATION_PROMPT = """Given the following context and claim, determine whether the claim is supported by the context.

Context: {context}

Claim: {claim}

Answer with exactly one of: SUPPORTED, UNSUPPORTED, or IMPLICIT
- SUPPORTED: The claim is directly stated in or can be logically inferred from the context.
- UNSUPPORTED: The claim contradicts the context or introduces information not present in it.
- IMPLICIT: The claim is partially supported but requires minor inference.

Your answer (one word only):"""


def verify_claim(claim: str, context: str) -> ClaimVerification:
    """
    Verify a single claim against the retrieved context using the LLM.

    Args:
        claim: A factual claim extracted from the response.
        context: The retrieved context (concatenated chunks) to verify against.

    Returns:
        ClaimVerification with the verdict.
    """
    prompt = CLAIM_VERIFICATION_PROMPT.format(context=context, claim=claim)
    llm_output = call_llm(prompt).strip().upper()

    # Parse the verdict (LLMs sometimes add extra text)
    if "SUPPORTED" in llm_output and "UNSUPPORTED" not in llm_output:
        verdict = "SUPPORTED"
    elif "UNSUPPORTED" in llm_output:
        verdict = "UNSUPPORTED"
    elif "IMPLICIT" in llm_output:
        verdict = "IMPLICIT"
    else:
        # Default to UNSUPPORTED if the LLM response is unclear
        verdict = "UNSUPPORTED"

    is_supported = verdict in ("SUPPORTED", "IMPLICIT")

    return ClaimVerification(
        claim=claim,
        verdict=verdict,
        is_supported=is_supported,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Faithfulness: Full Evaluation
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_faithfulness(
    response: str,
    context: str,
) -> FaithfulnessResult:
    """
    Evaluate the faithfulness of a response against the retrieved context.

    Faithfulness = supported_claims / total_claims

    Steps:
      1. Extract factual claims from the response (LLM call)
      2. Verify each claim against the context (LLM call per claim)
      3. Compute the score

    Args:
        response: The LLM-generated answer.
        context: The retrieved context (concatenated chunks or gold answer).

    Returns:
        FaithfulnessResult with score, claims, and verification details.
    """
    # Step 1: Extract claims
    claim_texts = extract_claims(response)

    if not claim_texts:
        return FaithfulnessResult(
            score=1.0,  # No claims = vacuously faithful
            total_claims=0,
            supported_claims=0,
            claims=[],
            response=response,
            context=context,
        )

    # Step 2: Verify each claim
    verifications = [verify_claim(claim, context) for claim in claim_texts]

    # Step 3: Compute score
    supported = sum(1 for v in verifications if v.is_supported)
    score = supported / len(verifications)

    return FaithfulnessResult(
        score=round(score, 4),
        total_claims=len(verifications),
        supported_claims=supported,
        claims=verifications,
        response=response,
        context=context,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Relevancy: Question Generation
# ═══════════════════════════════════════════════════════════════════════════

QUESTION_GENERATION_PROMPT = """Generate exactly {n} distinct questions that can be answered using ONLY the information in the answer below. \
Each question should reflect a different aspect of the answer's content. \
Output ONLY a numbered list with no other text.

Answer: {response}"""


def generate_questions(response: str, n: int = 3) -> list[str]:
    """
    Use the LLM to generate N questions that the response can answer.

    This is the core of the relevancy metric: if the response is relevant
    to the original query, then questions generated from the response should
    be semantically similar to the original query.

    Args:
        response: The LLM-generated answer.
        n: Number of questions to generate (default 3, per RAGAS).

    Returns:
        List of generated question strings.
    """
    prompt = QUESTION_GENERATION_PROMPT.format(n=n, response=response)
    llm_output = call_llm(prompt)
    questions = _parse_numbered_list(llm_output)
    return questions[:n]  # Ensure we don't return more than requested


# ═══════════════════════════════════════════════════════════════════════════
# Relevancy: Full Evaluation
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_relevancy(
    query: str,
    response: str,
    n_questions: int = 3,
) -> RelevancyResult:
    """
    Evaluate the relevancy of a response to the original query.

    Relevancy = mean cosine_similarity(embedding(generated_q_i), embedding(original_query))

    Steps:
      1. Generate N questions from the response (LLM call)
      2. Embed the generated questions and the original query
      3. Compute cosine similarity between each generated question and the query
      4. Score = mean similarity

    Args:
        query: The original user query.
        response: The LLM-generated answer.
        n_questions: Number of questions to generate (default 3).

    Returns:
        RelevancyResult with score, generated questions, and similarities.
    """
    # Step 1: Generate questions from the response
    gen_questions = generate_questions(response, n=n_questions)

    if not gen_questions:
        return RelevancyResult(
            score=0.0,
            generated_questions=[],
            similarities=[],
            query=query,
            response=response,
        )

    # Step 2: Embed everything
    model = _get_embed_model()
    all_texts = [query] + gen_questions
    embeddings = model.encode(
        all_texts, normalize_embeddings=True, convert_to_numpy=True
    )

    query_emb = embeddings[0]       # shape: (dim,)
    question_embs = embeddings[1:]  # shape: (N, dim)

    # Step 3: Cosine similarity (embeddings are L2-normalized → dot product)
    similarities = [
        float(np.dot(q_emb, query_emb))
        for q_emb in question_embs
    ]

    # Step 4: Mean similarity
    score = sum(similarities) / len(similarities)

    return RelevancyResult(
        score=round(score, 4),
        generated_questions=gen_questions,
        similarities=[round(s, 4) for s in similarities],
        query=query,
        response=response,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval metrics: Hit@K / MRR@K (noise-free, needs ground truth)
# ═══════════════════════════════════════════════════════════════════════════
# Not from the Unit 10 lecture — these are standard IR metrics, included
# here as the noise-free complement to faithfulness/relevancy. Use them
# when you have a known-correct product id for a query (e.g. iterating
# over held_out_eval_v2.csv); use faithfulness/relevancy for live queries
# where no ground truth exists.


def evaluate_retrieval_hit_mrr(
    expected_id: str,
    retrieved_ids: list[str],
) -> RetrievalMetricsResult:
    """
    Compute Hit@K and reciprocal rank for a single query against a known-
    correct product id, where K = len(retrieved_ids) (i.e. whatever top_k
    you already searched with).

    Args:
        expected_id: The ground-truth correct product id for this query.
        retrieved_ids: The ids actually returned by retrieval, in rank order
                       (e.g. [r.id for r in search(request).results]).

    Returns:
        RetrievalMetricsResult with hit, reciprocal_rank, and rank (if found).
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id == expected_id:
            return RetrievalMetricsResult(
                hit=True,
                reciprocal_rank=round(1.0 / rank, 4),
                rank=rank,
                expected_id=expected_id,
                retrieved_ids=retrieved_ids,
                top_k=len(retrieved_ids),
            )
    return RetrievalMetricsResult(
        hit=False,
        reciprocal_rank=0.0,
        rank=None,
        expected_id=expected_id,
        retrieved_ids=retrieved_ids,
        top_k=len(retrieved_ids),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Combined evaluation
# ═══════════════════════════════════════════════════════════════════════════


def evaluate(
    query: str,
    response: str,
    context: str,
    n_questions: int = 3,
    retrieved_ids: Optional[list[str]] = None,
    expected_id: Optional[str] = None,
) -> EvaluationResult:
    """
    Run faithfulness and relevancy, and optionally Hit@K/MRR@K.

    Args:
        query: The original user query.
        response: The LLM-generated answer.
        context: The retrieved context (concatenated chunks).
        n_questions: Number of questions for relevancy (default 3).
        retrieved_ids: The retrieved product ids in rank order, e.g.
                       [r.id for r in search(request).results]. Optional —
                       pass this together with expected_id to also get a
                       noise-free retrieval metric (e.g. when running
                       against held_out_eval_v2.csv, which has known-
                       correct answers). Leave as None for live queries.
        expected_id: The ground-truth correct product id for this query,
                     if known. Must be passed together with retrieved_ids.

    Returns:
        EvaluationResult with faithfulness and relevancy always populated,
        and retrieval populated only when both retrieved_ids and
        expected_id were provided.
    """
    faith = evaluate_faithfulness(response=response, context=context)
    rel = evaluate_relevancy(query=query, response=response, n_questions=n_questions)

    retrieval_result = None
    if retrieved_ids is not None and expected_id is not None:
        retrieval_result = evaluate_retrieval_hit_mrr(
            expected_id=expected_id, retrieved_ids=retrieved_ids
        )

    return EvaluationResult(faithfulness=faith, relevancy=rel, retrieval=retrieval_result)


# ═══════════════════════════════════════════════════════════════════════════
# CLI demo
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    print("=" * 70)
    print("RAG Evaluation Module — Manual Implementation")
    print("=" * 70)
    print()
    print("This module requires an LLM to be connected via call_llm().")
    print("See the docstring in call_llm() for integration instructions.")
    print()
    print("Example usage (after connecting your LLM):")
    print()
    print("  from evaluation import evaluate_faithfulness, evaluate_relevancy")
    print()
    print("  # Faithfulness")
    print('  faith = evaluate_faithfulness(')
    print('      response="Einstein was born in Germany on 14 March 1879.",')
    print('      context="Albert Einstein (born 14 March 1879) was a German-born physicist."')
    print("  )")
    print("  print(faith.score)  # → 1.0")
    print()
    print("  # Relevancy")
    print('  rel = evaluate_relevancy(')
    print('      query="Where was Einstein born?",')
    print('      response="Einstein was born in Germany on 14 March 1879."')
    print("  )")
    print("  print(rel.score)  # → ~0.9")
    print()
    print("  # Combined")
    print('  result = evaluate(query=..., response=..., context=...)')
    print("  print(result.faithfulness.score, result.relevancy.score)")