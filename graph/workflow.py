"""
graph/workflow.py
-----------------
Builds and compiles the full Phase 2 LangGraph workflow.

Phase 2 Graph:

  [router] ──────────────────────────────────────► [generate_direct] ──► END
      │ "retrieve"
      ▼
  [retrieve]
      │
      ▼
  [retrieval_grader]
      │ "relevant"                      "rewrite" (+ retry_count++)
      ▼                                     ▲
  [rerank] ◄────────────────────────── [query_rewriter]
      │
      ▼
  [generate]
      │
      ▼
  [answer_grader]
      │ "faithful"          "retry" (loops back to retrieve)
      ▼                          ▲
     END ◄──────────────────────┘

Circuit breakers in edges.py prevent infinite loops.
All nodes are injected with their dependencies via functools.partial.
"""

import functools
from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import (
    retrieve_node,
    rerank_node,
    generate_node,
    router_node,
    retrieval_grader_node,
    query_rewriter_node,
    answer_grader_node,
    generate_direct_node,
)
from graph.edges import route_query, grade_retrieval, grade_answer

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.llm import Generator


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_TOP_K      = 10
DEFAULT_TOP_N      = 4
DEFAULT_MAX_RETRIES = 2


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(vector_store):
    """
    Initializes all components and compiles the full Phase 2 LangGraph workflow.

    Parameters
    ----------
    vector_store : FaissVectorStore
        Pre-loaded FAISS index

    Returns
    -------
    CompiledGraph
        Ready to invoke with .invoke(initial_state)
    """

    # Instantiate components (same as main.py)
    retriever = Retriever(vector_store)
    reranker  = Reranker()
    generator = Generator()

    # Bind class instances into node functions
    # Phase 1 nodes need dependencies; Phase 2 nodes use llm_calls directly
    bound_retrieve = functools.partial(retrieve_node, retriever=retriever)
    bound_rerank   = functools.partial(rerank_node,   reranker=reranker)
    bound_generate = functools.partial(generate_node, generator=generator)

    # ── Build graph ───────────────────────────────────────────────────────
    workflow = StateGraph(AgentState)

    # Register all nodes
    workflow.add_node("router",             router_node)
    workflow.add_node("retrieve",           bound_retrieve)
    workflow.add_node("retrieval_grader",   retrieval_grader_node)
    workflow.add_node("query_rewriter",     query_rewriter_node)
    workflow.add_node("rerank",             bound_rerank)
    workflow.add_node("generate",           bound_generate)
    workflow.add_node("answer_grader",      answer_grader_node)
    workflow.add_node("generate_direct",    generate_direct_node)

    # ── Entry point ───────────────────────────────────────────────────────
    workflow.set_entry_point("router")

    # ── Router → RAG or Direct ────────────────────────────────────────────
    workflow.add_conditional_edges(
        "router",
        route_query,
        {
            "retrieve": "retrieve",
            "direct":   "generate_direct"
        }
    )

    # ── Retrieve → Grade retrieval quality ────────────────────────────────
    workflow.add_edge("retrieve", "retrieval_grader")

    # ── Retrieval Grader → Rerank or Rewrite ─────────────────────────────
    workflow.add_conditional_edges(
        "retrieval_grader",
        grade_retrieval,
        {
            "rerank":  "rerank",
            "rewrite": "query_rewriter"
        }
    )

    # ── Query Rewriter loops back to Retrieve ─────────────────────────────
    workflow.add_edge("query_rewriter", "retrieve")

    # ── Rerank → Generate ─────────────────────────────────────────────────
    workflow.add_edge("rerank", "generate")

    # ── Generate → Grade answer faithfulness ──────────────────────────────
    workflow.add_edge("generate", "answer_grader")

    # ── Answer Grader → End or Retry ─────────────────────────────────────
    workflow.add_conditional_edges(
        "answer_grader",
        grade_answer,
        {
            "end":   END,
            "retry": "retrieve"
        }
    )

    # ── Direct generation always ends ────────────────────────────────────
    workflow.add_edge("generate_direct", END)

    return workflow.compile()


# ── Query Runner ──────────────────────────────────────────────────────────────

def run_query(
    graph,
    query: str,
    department: str = None,
    top_k: int = DEFAULT_TOP_K,
    top_n: int = DEFAULT_TOP_N,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> AgentState:
    """
    Runs a query through the compiled Phase 2 graph.

    Parameters
    ----------
    graph       : compiled LangGraph app
    query       : user question
    department  : optional department filter
    top_k       : FAISS fetch count
    top_n       : post-rerank count
    max_retries : circuit breaker ceiling

    Returns
    -------
    AgentState
        Full final state. Access result via state["answer"].
    """

    initial_state: AgentState = {
        # Core
        "query":            query,
        "original_query":   query,   # preserved; query may be rewritten
        "department":       department,

        # Config
        "top_k":        top_k,
        "top_n":        top_n,
        "max_retries":  max_retries,

        # Pipeline data — empty at start
        "retrieved_chunks": [],
        "reranked_chunks":  [],
        "answer":           "",

        # Agent decisions — empty at start
        "query_type":       "",
        "retrieval_grade":  "",
        "answer_grade":     "",
        "retry_count":      0,
    }

    return graph.invoke(initial_state)
