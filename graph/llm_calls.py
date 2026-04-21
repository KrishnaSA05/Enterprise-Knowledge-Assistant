"""
graph/llm_calls.py
------------------
Lightweight Groq helpers used by the grader and router nodes.

Why a separate file?
  The main Generator class in generation/llm.py is designed for full
  RAG answer generation with context building, long outputs, and
  temperature=0.2. The grader/router nodes need fast, deterministic,
  structured JSON calls with temperature=0 and max_tokens=64.

  Separating them avoids modifying the original Generator class and
  keeps responsibilities clean.

Two helpers:
  call_groq_json  →  for Router, Retrieval Grader, Answer Grader
  call_groq_text  →  for Query Rewriter, Direct Generator
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Shared client — instantiated once, reused across all calls
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"


def call_groq_json(system_prompt: str, user_prompt: str) -> dict:
    """
    Calls Groq and parses the response as JSON.

    Used by: Router, Retrieval Grader, Answer Grader
    Temp=0 for fully deterministic classification output.

    Returns
    -------
    dict
        Parsed JSON. If parsing fails, returns {"error": raw_text}
        so nodes can handle gracefully without crashing the graph.
    """
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=64
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences the LLM sometimes adds
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": raw}


def call_groq_text(system_prompt: str, user_prompt: str) -> str:
    """
    Calls Groq and returns the raw text response.

    Used by: Query Rewriter, Direct Generator
    Slightly higher temp for more natural rewriting.

    Returns
    -------
    str
        Plain text response from the model.
    """
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=256
    )

    return response.choices[0].message.content.strip()
