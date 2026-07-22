"""
Shared LLM Client
--------------------
Used by both generation.py (STEP 2b — grounded answer generation) and
evaluation.py (STEP 3 — faithfulness/relevancy scoring) so that a model
loaded for one is reused for the other whenever they point at the same
model name, instead of each module loading its own copy independently.

Model: Tiny Aya (Cohere Labs) — per Units 8-10 of the course. Tiny Aya is a
massively multilingual (70+ languages, including Urdu), ~3.35B-parameter,
instruction-tuned model designed for local deployment under realistic
compute constraints. That combination (multilingual + runs locally without
an API key) is exactly what a Roman-Urdu/English product chatbot needs.

LLM1 vs LLM2 (per Unit 10's "Complete RAG Pipeline with Evaluation" slide):
    The lecture's pipeline diagram explicitly labels the generation model
    "LLM1 (Smaller Size)" and the evaluation/claim-extraction model
    "LLM2 (Size > LLM1)" — a judge model shouldn't be weaker than the model
    it's judging. This module supports that split via two separate model
    names (config.GENERATION_MODEL_NAME / config.EVALUATION_MODEL_NAME).
    They default to the same Tiny Aya model for simplicity and so this runs
    without extra downloads out of the box — point EVALUATION_MODEL_NAME at
    something bigger (a larger Aya variant, an OpenAI/Claude model via a
    different call_llm backend, etc.) once you have the compute/API budget.

Performance note:
    A ~3.35B-parameter model in float32 on CPU is slow — expect anywhere
    from several seconds to over a minute per call depending on your
    machine and max_new_tokens. If you have a CUDA GPU available (e.g.
    running this on Kaggle/Colab like the embedding fine-tuning step was),
    it loads in float16 automatically and is far faster. For local CPU-only
    testing, keep max_new_tokens small and test on a handful of queries at
    a time rather than running a full evaluation batch.

Swapping backends:
    Every caller in this project goes through call_llm() in this one file.
    To swap in OpenAI, Claude, or an HTTP endpoint instead of a locally-run
    Tiny Aya, you only need to change the body of call_llm() below — nothing
    in generation.py or evaluation.py needs to change.

Requirements:
    pip install transformers torch accelerate
"""
import os
from openai import OpenAI
from google import genai
from google.genai import types as genai_types
from groq import Groq  # <--- Make sure this line is added!
from config import GENERATION_MODEL_NAME

def call_llm(prompt: str, model_name: str = GENERATION_MODEL_NAME, max_new_tokens: int = 256, temperature: float = 0.3) -> str:
    # 1. Route to OpenAI (for evaluation.py judge)
    if model_name.startswith("gpt-"):
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    # 2. Route to Groq (for Llama / Mixtral / Gemma models)
    elif "llama" in model_name.lower() or "gemma" in model_name.lower() or "mixtral" in model_name.lower():
        client = Groq() # Auto-reads GROQ_API_KEY from environment
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    # 3. Route to Google Gemini (Fallback if needed later)
    elif "gemini" in model_name.lower():
        client = genai.Client()
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=max_new_tokens,
                temperature=temperature,
            ),
        )
        return resp.text.strip()


if __name__ == "__main__":
    # Quick manual smoke test: `python llm_client.py`
    print("Loading model and running a smoke test prompt ...\n")
    reply = call_llm("In one short sentence, what is BM25 used for?")
    print("Response:", reply)
