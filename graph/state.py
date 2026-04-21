"""
graph/state.py
--------------
AgentState — the single typed dict that flows through every LangGraph node.

Phase 1 fields: query, department, top_k, top_n,
                retrieved_chunks, reranked_chunks, answer

Phase 2 additions:
  original_query    — preserves the user's raw query before any rewriting
  query_type        — router decision: "retrieve" | "direct"
  retrieval_grade   — grader decision: "relevant" | "irrelevant"
  answer_grade      — grader decision: "faithful" | "hallucinated"
  retry_count       — tracks how many rewrite+retrieve loops have run
  max_retries       — circuit breaker ceiling (set in workflow.py)
"""

from typing import List, Dict, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):

    # ── Core query fields ────────────────────────────────────────────────────
    query: str
    """Current query — may be rewritten by query_rewriter_node."""

    original_query: str
    """The user's original raw query — never mutated after entry."""

    department: Optional[str]
    """Optional department filter for scoped FAISS retrieval."""

    # ── Retrieval config ─────────────────────────────────────────────────────
    top_k: int
    """Number of chunks to fetch from FAISS before reranking."""

    top_n: int
    """Number of chunks to keep after reranking."""

    # ── Pipeline data ────────────────────────────────────────────────────────
    retrieved_chunks: List[Dict]
    """Raw FAISS results. Keys: score, metadata (section, source_file, text)"""

    reranked_chunks: List[Dict]
    """Reranked subset. Same as retrieved_chunks + rerank_score (float)."""

    answer: str
    """Final generated answer. Empty string until generate_node runs."""

    # ── Phase 2: Agent decisions ─────────────────────────────────────────────
    query_type: str
    """
    Router output.
      "retrieve" → full RAG pipeline
      "direct"   → answer without retrieval (chitchat / general knowledge)
    """

    retrieval_grade: str
    """
    Retrieval grader output.
      "relevant"   → chunks are useful, proceed to reranking
      "irrelevant" → chunks are poor, rewrite query and retry
    """

    answer_grade: str
    """
    Answer grader output.
      "faithful"      → answer is grounded in sources, return to user
      "hallucinated"  → answer not supported by chunks, retry
    """

    retry_count: int
    """Number of rewrite+retrieve loops completed. Checked against max_retries."""

    max_retries: int
    """Maximum allowed retries before the graph forces an answer. Default: 2."""
