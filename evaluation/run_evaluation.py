"""
evaluation/run_evaluation.py
-----------------------------
Full evaluation suite for the Enterprise Knowledge Assistant.

Supports three pipeline modes via --pipeline flag:

  --pipeline linear   Evaluates V1 (Retriever + Reranker + Generator)
  --pipeline graph    Evaluates V2 (LangGraph agentic pipeline)
  --pipeline both     Runs both and prints a side-by-side comparison table

Usage:
  # Graph only (two equivalent ways)
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --pipeline graph --k 10
  python evaluation/run_evaluation.py --pipeline both --k 10
  python evaluation/run_evaluation.py --pipeline linear --k 5
"""

import os
import sys
import time
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from evaluation.eval_data import EVAL_QUESTIONS
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    answer_has_keywords,
    answer_is_grounded,
    router_correct,
    rewrite_was_appropriate,
)

from retrieval.vector_store import FaissVectorStore
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.llm import Generator
from graph.workflow import build_graph, run_query

INDEX_PATH = os.path.join(ROOT_DIR, "artifacts", "faiss")


# ── Pipeline Runners ──────────────────────────────────────────────────────────

def run_linear_query(retriever, reranker, generator, question, k, top_n):
    """
    Runs a single query through the original V1 linear pipeline.
    Returns a result dict in the same shape as run_graph_query()
    so both pipelines feed into identical metric computation.
    """
    retrieved  = retriever.retrieve(question, top_k=k)
    reranked   = reranker.rerank(question, retrieved, top_n=top_n)
    answer     = generator.generate_answer(question, reranked)

    return {
        "answer":          answer,
        "reranked_chunks": reranked,
        "query_type":      "retrieve",   # linear always retrieves
        "retry_count":     0,            # no rewriting in linear
        "original_query":  question,
        "query":           question,
    }


def run_graph_query(graph, question, k, top_n, max_retries):
    """
    Runs a single query through the V2 LangGraph agentic pipeline.
    Returns the full AgentState dict.
    """
    return run_query(
        graph       = graph,
        query       = question,
        top_k       = k,
        top_n       = top_n,
        max_retries = max_retries
    )


# ── Core Evaluator ────────────────────────────────────────────────────────────

def evaluate_pipeline(runner_fn, pipeline_name: str, k: int) -> dict:
    """
    Runs the full eval dataset through a given pipeline runner function.

    Parameters
    ----------
    runner_fn     : callable(question) -> result dict
    pipeline_name : label for display ("Linear V1" or "LangGraph V2")
    k             : top-K value (for metric labels)

    Returns
    -------
    dict with aggregate scores and per-question results
    """
    print(f"\n{'=' * 70}")
    print(f"  Evaluating: {pipeline_name}")
    print(f"{'=' * 70}")

    results = []

    for i, item in enumerate(EVAL_QUESTIONS):
        question       = item["question"]
        relevant_secs  = item["relevant_sections"]
        expected_kws   = item["expected_keywords"]
        expected_qtype = item["query_type"]

        state          = runner_fn(question)
        time.sleep(1.5)   # avoid Groq 503 rate limit on back-to-back calls
        answer         = state["answer"]
        retrieved_secs = [c["metadata"].get("section") for c in state["reranked_chunks"]]
        actual_qtype   = state["query_type"]
        retry_count    = state["retry_count"]

        p_k    = precision_at_k(retrieved_secs, relevant_secs, k)
        r_k    = recall_at_k(retrieved_secs, relevant_secs, k)
        kw     = answer_has_keywords(answer, expected_kws)
        grnd   = answer_is_grounded(answer, relevant_secs, retrieved_secs)
        router = router_correct(actual_qtype, expected_qtype)
        rewr   = rewrite_was_appropriate(retry_count, relevant_secs, retrieved_secs)

        result = {
            "question":       question,
            "precision":      p_k,
            "recall":         r_k,
            "keyword_score":  kw,
            "grounded":       grnd,
            "router_correct": router,
            "rewrite":        rewr,
            "retry_count":    retry_count,
            "actual_qtype":   actual_qtype,
            "expected_qtype": expected_qtype,
            "answer":         answer,
        }
        results.append(result)

        status = "✅" if grnd else "❌"
        print(f"  [{i+1:02d}] {status} P@{k}={p_k:.2f} R@{k}={r_k:.2f} "
              f"KW={kw:.2f} Router={'✅' if router else '❌'} "
              f"| {question[:55]}")

    # Aggregate
    rag_items = [r for r in results if r["expected_qtype"] == "retrieve"]
    rewrites  = [r["rewrite"] for r in results if r["rewrite"] is not None]

    scores = {
        "precision":    sum(r["precision"]     for r in rag_items) / len(rag_items),
        "recall":       sum(r["recall"]        for r in rag_items) / len(rag_items),
        "keyword":      sum(r["keyword_score"] for r in results)   / len(results),
        "grounded":     sum(1 for r in results if r["grounded"])   / len(results),
        "router_acc":   sum(1 for r in results if r["router_correct"]) / len(results),
        "rewrite_acc":  sum(rewrites) / len(rewrites) if rewrites else None,
    }

    return {"name": pipeline_name, "scores": scores, "results": results}


# ── Comparison Printer ────────────────────────────────────────────────────────

def print_comparison(linear_eval: dict, graph_eval: dict, k: int):
    """Prints a clean side-by-side comparison table."""

    print(f"\n{'=' * 70}")
    print("  📊 COMPARISON: Linear V1  vs  LangGraph V2")
    print(f"{'=' * 70}")

    metrics = [
        ("Precision@" + str(k),   "precision"),
        ("Recall@"    + str(k),   "recall"),
        ("Keyword Score",          "keyword"),
        ("Answer Grounded",        "grounded"),
        ("Router Accuracy",        "router_acc"),
        ("Rewrite Accuracy",       "rewrite_acc"),
    ]

    ls = linear_eval["scores"]
    gs = graph_eval["scores"]

    print(f"\n  {'Metric':<22} {'Linear V1':>12} {'LangGraph V2':>14} {'Delta':>8}")
    print(f"  {'-'*22} {'-'*12} {'-'*14} {'-'*8}")

    for label, key in metrics:
        lv = ls[key]
        gv = gs[key]

        if lv is None and gv is None:
            print(f"  {label:<22} {'N/A':>12} {'N/A':>14} {'N/A':>8}")
            continue

        lv = lv if lv is not None else 0.0
        gv = gv if gv is not None else 0.0

        delta   = gv - lv
        arrow   = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
        color   = "+" if delta > 0.01 else ("-" if delta < -0.01 else " ")

        print(f"  {label:<22} {lv:>12.2f} {gv:>14.2f} {color}{abs(delta):.2f} {arrow}")

    print(f"\n  {'=' * 70}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(pipeline: str, k: int, top_n: int, max_retries: int):

    # Load shared FAISS index
    print("🔹 Loading FAISS index...")
    store = FaissVectorStore(384)
    store.load(INDEX_PATH)

    linear_eval = None
    graph_eval  = None

    # ── Linear pipeline ───────────────────────────────────────────────────────
    if pipeline in ("linear", "both"):
        retriever = Retriever(store)
        reranker  = Reranker()
        generator = Generator()

        def linear_runner(question):
            return run_linear_query(retriever, reranker, generator, question, k, top_n)

        linear_eval = evaluate_pipeline(linear_runner, "Linear V1", k)

    # ── LangGraph pipeline ────────────────────────────────────────────────────
    if pipeline in ("graph", "both"):
        print("\n🔗 Compiling LangGraph workflow...")
        graph = build_graph(store)

        def graph_runner(question):
            return run_graph_query(graph, question, k, top_n, max_retries)

        graph_eval = evaluate_pipeline(graph_runner, "LangGraph V2", k)

    # ── Summary ───────────────────────────────────────────────────────────────
    if pipeline == "both":
        print_comparison(linear_eval, graph_eval, k)

    elif pipeline == "linear":
        s = linear_eval["scores"]
        print(f"\n📊 Linear V1 — Avg Precision@{k}: {s['precision']:.2f} | "
              f"Recall@{k}: {s['recall']:.2f} | "
              f"Grounded: {s['grounded']:.2f}")

    elif pipeline == "graph":
        s = graph_eval["scores"]
        print(f"\n📊 LangGraph V2 — Avg Precision@{k}: {s['precision']:.2f} | "
              f"Recall@{k}: {s['recall']:.2f} | "
              f"Grounded: {s['grounded']:.2f} | "
              f"Router Acc: {s['router_acc']:.2f}")

    print("\n✅ Evaluation complete\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ENA Evaluation Suite")
    parser.add_argument(
        "--pipeline",
        choices=["linear", "graph", "both"],
        default="graph",
        help="Which pipeline to evaluate (default: graph)"
    )
    parser.add_argument("--k",           type=int, default=5, help="Top-K retrieval count")
    parser.add_argument("--top_n",       type=int, default=4, help="Post-rerank count")
    parser.add_argument("--max_retries", type=int, default=2, help="LangGraph max retries")
    args = parser.parse_args()

    main(
        pipeline    = args.pipeline,
        k           = args.k,
        top_n       = args.top_n,
        max_retries = args.max_retries
    )