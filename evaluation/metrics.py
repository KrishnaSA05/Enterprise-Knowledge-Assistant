"""
evaluation/metrics.py
---------------------
All evaluation metrics for the Enterprise Knowledge Assistant.

Retrieval metrics (unchanged from V1):
  precision_at_k   — what fraction of top-K retrieved sections are relevant
  recall_at_k      — what fraction of relevant sections appear in top-K

Generation metrics (new):
  answer_has_keywords   — does the answer contain expected keywords
  answer_faithfulness   — does the answer avoid claiming the doc has no info
                          when relevant sections were retrieved

Agentic metrics (new):
  router_accuracy       — did the router classify query_type correctly
  rewrite_triggered     — did the query rewriter fire when it should have
"""

from typing import List, Optional


# ── Retrieval Metrics (unchanged) ─────────────────────────────────────────────

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Fraction of top-K retrieved sections that are in the relevant set.
    Range: 0.0 – 1.0. Higher is better.
    """
    if k == 0:
        return 0.0

    retrieved_k  = retrieved[:k]
    relevant_set = set(relevant)
    true_positives = sum(1 for r in retrieved_k if r in relevant_set)
    return true_positives / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Fraction of all relevant sections that appear in top-K retrieved.
    Range: 0.0 – 1.0. Higher is better.
    Returns 1.0 for out-of-scope questions (no relevant sections expected).
    """
    if not relevant:
        return 1.0  # out-of-scope: nothing to recall, not a retrieval failure

    retrieved_k  = retrieved[:k]
    relevant_set = set(relevant)
    true_positives = sum(1 for r in retrieved_k if r in relevant_set)
    return true_positives / len(relevant_set)


# ── Generation Metrics (new) ──────────────────────────────────────────────────

def answer_has_keywords(answer: str, expected_keywords: List[str]) -> float:
    """
    Fraction of expected keywords present in the answer (case-insensitive).
    Range: 0.0 – 1.0. Higher is better.
    Returns 1.0 when no keywords are expected (direct/chitchat queries).

    This is a lightweight proxy for answer correctness — not a replacement
    for semantic evaluation, but meaningful for known-answer questions.
    """
    if not expected_keywords:
        return 1.0

    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def answer_is_grounded(
    answer: str,
    relevant_sections: List[str],
    retrieved_sections: List[str]
) -> bool:
    """
    Checks whether the answer is consistent with the retrieval outcome.

    Grounded = True when:
      - Relevant sections were retrieved AND answer does NOT say "I don't know"
      - No relevant sections exist (out-of-scope) AND answer DOES say "I don't know"

    Grounded = False when:
      - Relevant sections were retrieved BUT answer says "I don't know"
        (retrieval succeeded but generation failed)
      - No relevant sections but answer makes confident claims
        (hallucination on out-of-scope query)
    """
    answer_lower   = answer.lower()
    says_dont_know = "don't know" in answer_lower or "not found" in answer_lower

    relevant_set   = set(relevant_sections)
    retrieved_set  = set(retrieved_sections)
    found_relevant = bool(relevant_set & retrieved_set)  # intersection

    if relevant_sections:
        # We expect an answer — "I don't know" when relevant chunks were
        # retrieved means the generation stage failed
        return found_relevant and not says_dont_know
    else:
        # Out-of-scope — we expect "I don't know"
        return says_dont_know


# ── Agentic Metrics (new) ─────────────────────────────────────────────────────

def router_correct(actual_query_type: str, expected_query_type: str) -> bool:
    """
    Whether the router classified the query type correctly.
    True/False — aggregated as accuracy across the eval set.
    """
    return actual_query_type == expected_query_type


def rewrite_was_appropriate(
    retry_count: int,
    relevant_sections: List[str],
    retrieved_sections: List[str]
) -> Optional[bool]:
    """
    Checks whether the query rewriter fired appropriately.

    Returns:
      True   — rewrite fired AND relevant sections were not in initial retrieval
      False  — rewrite fired BUT relevant sections were already retrieved (unnecessary)
      None   — rewrite did not fire (not applicable)
    """
    if retry_count == 0:
        return None  # rewriter never fired — not applicable

    relevant_set  = set(relevant_sections)
    retrieved_set = set(retrieved_sections)
    was_needed    = not bool(relevant_set & retrieved_set)

    return was_needed  # True = appropriately fired, False = fired unnecessarily
