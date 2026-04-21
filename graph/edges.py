"""
graph/edges.py
--------------
All conditional edge functions for the LangGraph workflow.

Edge functions receive AgentState and return a string key
that maps to the next node name (defined in workflow.py).

Phase 2 — all three edges are now fully implemented:

  route_query       → "retrieve" | "direct"
  grade_retrieval   → "rerank"   | "rewrite"
  grade_answer      → "end"      | "retry"

Circuit breaker
---------------
retry_count and max_retries are checked in grade_retrieval and grade_answer
to prevent infinite loops. When the limit is hit, the graph forces forward
rather than looping back — ensuring the user always gets a response.
"""

from graph.state import AgentState


def route_query(state: AgentState) -> str:
    """
    Routes to retrieval or direct generation based on router_node output.

    "retrieve" → retrieve_node  (full RAG pipeline)
    "direct"   → generate_direct_node  (no retrieval)
    """
    return state.get("query_type", "retrieve")


def grade_retrieval(state: AgentState) -> str:
    """
    Decides whether to proceed with reranking or rewrite the query.

    Circuit breaker: if retry_count >= max_retries, force forward
    to reranking with whatever chunks we have — better than looping forever.

    "rerank"  → rerank_node  (chunks are good enough)
    "rewrite" → query_rewriter_node  (chunks are irrelevant, try again)
    """
    # Circuit breaker — force forward if max retries exceeded
    if state["retry_count"] >= state["max_retries"]:
        print(f"\n⚠️  [grade_retrieval] Max retries ({state['max_retries']}) reached — forcing forward")
        return "rerank"

    grade = state.get("retrieval_grade", "relevant")
    return "rerank" if grade == "relevant" else "rewrite"


def grade_answer(state: AgentState) -> str:
    """
    Decides whether to return the answer or retry generation.

    Circuit breaker: if retry_count >= max_retries, always return "end"
    to prevent the graph from looping on answer quality indefinitely.

    "end"   → END  (answer is faithful, return to user)
    "retry" → retrieve_node  (hallucinated, try full retrieval again)
    """
    # Circuit breaker — always end if retries are exhausted
    if state["retry_count"] >= state["max_retries"]:
        print(f"\n⚠️  [grade_answer] Max retries reached — returning best available answer")
        return "end"

    grade = state.get("answer_grade", "faithful")
    return "end" if grade == "faithful" else "retry"
