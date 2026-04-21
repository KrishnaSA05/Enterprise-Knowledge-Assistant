"""
graph/nodes.py
--------------
All LangGraph node functions — Phase 1 + Phase 2.

Every node is a pure function: (AgentState) -> dict (partial state update)

Phase 1 Nodes (unchanged behavior):
  retrieve_node       wraps Retriever.retrieve()
  rerank_node         wraps Reranker.rerank()
  generate_node       wraps Generator.generate_answer()

Phase 2 Nodes (new):
  router_node         classifies query → "retrieve" | "direct"
  retrieval_grader_node   grades chunks → "relevant" | "irrelevant"
  query_rewriter_node     rewrites query when retrieval fails
  answer_grader_node      grades answer → "faithful" | "hallucinated"
  generate_direct_node    answers simple queries without any retrieval
"""

import functools
from graph.state import AgentState
from graph.llm_calls import call_groq_json, call_groq_text

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.llm import Generator

from generation.prompts import (
    ROUTER_SYSTEM_PROMPT,        ROUTER_USER_TEMPLATE,
    RETRIEVAL_GRADER_SYSTEM_PROMPT, RETRIEVAL_GRADER_USER_TEMPLATE,
    QUERY_REWRITER_SYSTEM_PROMPT,   QUERY_REWRITER_USER_TEMPLATE,
    ANSWER_GRADER_SYSTEM_PROMPT,    ANSWER_GRADER_USER_TEMPLATE,
    DIRECT_SYSTEM_PROMPT
)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 NODES
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_node(state: AgentState, retriever: Retriever) -> dict:
    """
    Fetches top-K candidate chunks from FAISS.

    Reads : query, top_k, department
    Writes: retrieved_chunks
    """
    print(f"\n📥 [retrieve_node] Querying FAISS | retry={state['retry_count']}")
    print(f"   Query: '{state['query']}'")

    chunks = retriever.retrieve(
        query=state["query"],
        top_k=state["top_k"],
        department=state.get("department")
    )

    print(f"   → Retrieved {len(chunks)} chunks")
    return {"retrieved_chunks": chunks}


def rerank_node(state: AgentState, reranker: Reranker) -> dict:
    """
    Reranks retrieved chunks with a cross-encoder and keeps top-N.

    Reads : query, retrieved_chunks, top_n
    Writes: reranked_chunks
    """
    print(f"\n🏆 [rerank_node] Reranking {len(state['retrieved_chunks'])} → top {state['top_n']}")

    reranked = reranker.rerank(
        query=state["query"],
        retrieved_chunks=state["retrieved_chunks"],
        top_n=state["top_n"]
    )

    for i, chunk in enumerate(reranked):
        section = chunk["metadata"].get("section", "?")
        score   = chunk.get("rerank_score", 0.0)
        print(f"   [{i+1}] score={score:.4f} | section='{section}'")

    return {"reranked_chunks": reranked}


def generate_node(state: AgentState, generator: Generator) -> dict:
    """
    Calls Groq LLM to generate a grounded answer from reranked chunks.

    Reads : query, reranked_chunks
    Writes: answer
    """
    print(f"\n🤖 [generate_node] Generating answer from {len(state['reranked_chunks'])} chunks")

    answer = generator.generate_answer(
        question=state["query"],
        retrieved_chunks=state["reranked_chunks"]
    )

    print(f"   → Answer generated ({len(answer)} chars)")
    return {"answer": answer}


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 NODES
# ─────────────────────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """
    Classifies the query to decide the pipeline path.

    Reads : query
    Writes: query_type → "retrieve" | "direct"

    Falls back to "retrieve" if LLM returns unexpected output,
    ensuring the graph never gets stuck at the entry point.
    """
    print(f"\n🔀 [router_node] Classifying query...")

    result = call_groq_json(
        ROUTER_SYSTEM_PROMPT,
        ROUTER_USER_TEMPLATE.format(query=state["query"])
    )

    # Defensive: fall back to retrieve if classification fails
    query_type = result.get("query_type", "retrieve")
    if query_type not in ("retrieve", "direct"):
        query_type = "retrieve"

    print(f"   → Query type: '{query_type}'")
    return {"query_type": query_type}


def retrieval_grader_node(state: AgentState) -> dict:
    """
    Grades whether the retrieved chunks are relevant to the query.

    Strategy: sample the top 3 chunks and grade each individually.
    If at least 1 chunk is relevant → grade as "relevant".
    This is permissive by design — we only rewrite if ALL top chunks fail.

    Reads : query, retrieved_chunks
    Writes: retrieval_grade → "relevant" | "irrelevant"
    """
    print(f"\n📊 [retrieval_grader_node] Grading retrieval quality...")

    # Sample top 3 only — grading all 10 is expensive and rarely necessary
    chunks_to_grade = state["retrieved_chunks"][:3]

    if not chunks_to_grade:
        print("   → No chunks retrieved, marking as irrelevant")
        return {"retrieval_grade": "irrelevant"}

    relevant_count = 0
    for chunk in chunks_to_grade:
        # Truncate chunk text to 400 chars — enough for grading, cheap on tokens
        chunk_text = chunk["metadata"].get("text", "")[:400]

        result = call_groq_json(
            RETRIEVAL_GRADER_SYSTEM_PROMPT,
            RETRIEVAL_GRADER_USER_TEMPLATE.format(
                query=state["query"],
                chunk_text=chunk_text
            )
        )

        if result.get("grade") == "relevant":
            relevant_count += 1

    # At least 1 relevant chunk → proceed to reranking
    grade = "relevant" if relevant_count >= 1 else "irrelevant"

    print(f"   → {relevant_count}/{len(chunks_to_grade)} chunks relevant → grade='{grade}'")
    return {"retrieval_grade": grade}


def query_rewriter_node(state: AgentState) -> dict:
    """
    Rewrites the query to improve retrieval on the next attempt.

    Called when retrieval_grader grades chunks as "irrelevant".
    Increments retry_count to eventually trigger the circuit breaker.
    Clears stale chunk data so the next retrieve_node starts fresh.

    Reads : query, retry_count
    Writes: query (rewritten), retry_count (+1), retrieved_chunks ([]), reranked_chunks ([])
    """
    print(f"\n✏️  [query_rewriter_node] Rewriting query (attempt {state['retry_count'] + 1})...")
    print(f"   Original: '{state['query']}'")

    rewritten = call_groq_text(
        QUERY_REWRITER_SYSTEM_PROMPT,
        QUERY_REWRITER_USER_TEMPLATE.format(query=state["query"])
    )

    # Strip quotes the LLM sometimes wraps the rewrite in
    rewritten = rewritten.strip('"\'').strip()

    print(f"   Rewritten: '{rewritten}'")

    return {
        "query":            rewritten,
        "retry_count":      state["retry_count"] + 1,
        "retrieved_chunks": [],   # clear stale results
        "reranked_chunks":  []
    }


def answer_grader_node(state: AgentState) -> dict:
    """
    Grades whether the generated answer is faithful to the source chunks.

    Catches hallucinations before the answer reaches the user.
    "I don't know" style answers are always graded as faithful.

    Reads : query (original_query), answer, reranked_chunks
    Writes: answer_grade → "faithful" | "hallucinated"
    """
    print(f"\n✅ [answer_grader_node] Grading answer faithfulness...")

    # Build context string inline (avoids re-instantiating Generator)
    context_blocks = []
    for chunk in state["reranked_chunks"]:
        section = chunk["metadata"].get("section", "Unknown")
        source  = chunk["metadata"].get("source_file", "Unknown")
        text    = chunk["metadata"].get("text", "")
        context_blocks.append(f"[Source: {source} | Section: {section}]\n{text}")
    context = "\n\n".join(context_blocks)

    result = call_groq_json(
        ANSWER_GRADER_SYSTEM_PROMPT,
        ANSWER_GRADER_USER_TEMPLATE.format(
            question=state["original_query"],  # grade against what user actually asked
            context=context,
            answer=state["answer"]
        )
    )

    grade = result.get("grade", "faithful")
    if grade not in ("faithful", "hallucinated"):
        grade = "faithful"  # defensive fallback

    print(f"   → Answer grade: '{grade}'")
    return {"answer_grade": grade}


def generate_direct_node(state: AgentState) -> dict:
    """
    Answers simple queries directly without any document retrieval.

    Called when router classifies query_type as "direct".
    Uses a lighter system prompt — no source-grounding rules.

    Reads : query
    Writes: answer, reranked_chunks ([])
    """
    print(f"\n💬 [generate_direct_node] Answering directly (no retrieval)")

    answer = call_groq_text(
        DIRECT_SYSTEM_PROMPT,
        state["query"]
    )

    print(f"   → Direct answer generated ({len(answer)} chars)")

    # No chunks used — set empty so downstream display logic doesn't break
    return {
        "answer":          answer,
        "reranked_chunks": []
    }
