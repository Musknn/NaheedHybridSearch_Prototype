"""
Shared LLM Client
--------------------
Single interface to multiple LLM providers (Groq, OpenAI, Gemini), used by
generation.py, router.py, and evaluation.py. Every caller in this project
goes through call_llm() — to swap providers or add a new one, only this
file needs to change.

Model routing is by name prefix/substring:
    "gpt-*", "o1*", "o3*"                       -> OpenAI
    "llama*", "gemma*", "mixtral*", "qwen*",
        "gpt-oss*"                              -> Groq
    "gemini*"                                   -> Gemini

Default model name and default hyperparameters (max_new_tokens,
temperature) come from config.GENERATION — never hardcoded here.
"""
from openai import OpenAI
from google import genai
from google.genai import types as genai_types
from groq import Groq

from config import GENERATION, GENERATION_MODEL_NAME


class UnsupportedModelError(ValueError):
    """Raised when model_name doesn't match any known provider prefix."""


def call_llm(
    prompt: str,
    model_name: str = GENERATION_MODEL_NAME,
    max_new_tokens: int = GENERATION.max_new_tokens,
    temperature: float = GENERATION.temperature,
) -> str:
    """Route `prompt` to the right provider based on `model_name` and return the text response."""
    name = model_name.lower()

    if name.startswith("gpt-") or name.startswith("o1") or name.startswith("o3"):
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    if any(p in name for p in ("llama", "gemma", "mixtral", "qwen", "gpt-oss")):
        client = Groq()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    if "gemini" in name:
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

    raise UnsupportedModelError(
        f"No provider route for model_name={model_name!r}. "
        f"Add a branch in llm_client.call_llm() for this model family."
    )


if __name__ == "__main__":
    print("Running a smoke test prompt ...\n")
    reply = call_llm("In one short sentence, what is BM25 used for?")
    print("Response:", reply)
    
