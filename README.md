# 📘 Enterprise Knowledge Assistant (Agentic RAG System)

> **Retrieval-Augmented Generation (RAG) system extended with a LangGraph agentic pipeline for self-correcting, grounded question answering over enterprise documentation**

---

## 🔍 Overview

The **Enterprise Knowledge Assistant** is an end-to-end **Retrieval-Augmented Generation (RAG)** system that enables natural language querying over enterprise documents such as employee handbooks, policies, and operational guidelines.

The system combines **semantic retrieval, reranking, and grounded LLM generation** to provide accurate, source-cited answers while minimizing hallucinations.

This project was built in two stages:

- **Version 1** — Built from scratch without LangChain to deeply understand every stage of the RAG pipeline. A LangChain baseline is also provided for comparison.
- **Version 2** — Extended with a **LangGraph agentic workflow** that adds self-correcting retrieval, query rewriting, faithfulness grading, and intelligent routing on top of the same core modules.

---

## 🚀 Key Features

**Core RAG Pipeline**
- 📄 Section-aware ingestion of Markdown enterprise documents
- 🔎 Dense semantic retrieval using Hugging Face embeddings
- 🏆 Cross-encoder reranking for improved precision
- 🤖 Low-latency LLM inference using Groq (LLaMA-3.1-8B)
- 📚 Source-grounded answers with citations
- ⚡ Persistent FAISS vector index for fast startup
- 🌐 FastAPI backend for serving queries
- 🎨 Interactive Streamlit web UI for real-time question answering
- 📊 Quantitative evaluation using Precision@K and Recall@K

**Agentic LangGraph Extension**
- 🔀 Query Router — skips retrieval for direct/conversational queries
- 📊 Retrieval Grader — detects poor retrieval before reranking
- ✏️ Query Rewriter — self-corrects on retrieval failure
- ✅ Answer Grader — catches hallucinations before returning to user
- 🔁 Self-correcting loops with circuit breakers to guarantee termination
- 🏢 Department-scoped retrieval support

---

## 🧠 System Architecture

### Version 1 — Linear RAG Pipeline
```
User Query
   ↓
Query Embedding (MiniLM)
   ↓
FAISS Vector Search (Top-K)
   ↓
Cross-Encoder Reranker (Top-N)
   ↓
LLM (Groq LLaMA-3.1-8B)
   ↓
Answer + Source Citations
```

### Version 2 — LangGraph Agentic Pipeline
```
User Query
   ↓
┌──────────┐
│  Router  │ ──── "direct" ──────────────────────► Direct LLM Answer
└────┬─────┘
     │ "retrieve"
     ▼
┌──────────┐
│ Retrieve │ ◄─────────────────────────────────┐
└────┬─────┘                                   │
     ▼                                         │
┌──────────────────┐   "irrelevant"   ┌────────┴────────┐
│ Retrieval Grader │ ───────────────► │  Query Rewriter │
└────────┬─────────┘                  └─────────────────┘
         │ "relevant"
         ▼
┌──────────────────┐
│      Rerank      │
└────────┬─────────┘
         ▼
┌──────────────────┐
│     Generate     │
└────────┬─────────┘
         ▼
┌──────────────────┐   "hallucinated"
│  Answer Grader   │ ──────────────► Retry Retrieve
└────────┬─────────┘
         │ "faithful"
         ▼
  Answer + Sources + Agent Trace
```

---

## 🗂️ Project Structure
```
Enterprise-Knowledge-Assistant/
│
├── ingestion/
│   ├── loader.py
│   ├── chunker.py
│   └── embedder.py
│
├── retrieval/
│   ├── vector_store.py
│   ├── retriever.py
│   └── reranker.py
│
├── generation/
│   ├── prompts.py          ← Updated: Router, Grader, Rewriter prompts
│   └── llm.py
│
├── graph/                  ← LangGraph agentic pipeline
│   ├── __init__.py
│   ├── state.py            ← AgentState TypedDict
│   ├── nodes.py            ← All node functions (Phase 1 + Phase 2)
│   ├── edges.py            ← Conditional routing logic + circuit breakers
│   ├── llm_calls.py        ← Lightweight Groq helpers for graders/router
│   └── workflow.py         ← Compiled LangGraph app + run_query()
│
├── evaluation/
│   ├── eval_data.py        ← 15 ground-truth QA pairs across 5 categories
│   ├── metrics.py          ← Retrieval + generation + agentic metrics
│   └── run_evaluation.py   ← Supports --pipeline linear | graph | both
│
├── api/
│   └── main.py             ← Updated: serves LangGraph pipeline
│
├── frontend/
│   └── app.py              ← Updated: Agent Trace panel + sources
│
├── artifacts/
│   ├── faiss.index
│   └── faiss.meta
│
├── data/
│   └── *.md
│
├── ENA_with_langchain.py     ← LangChain baseline
├── ENA_with_langchain_Agentic.py ← Agentic baseline (LangGraph)
├── main.py                 ← Version 1 entry point 
├── main_graph.py           ← Version 2 entry point (LangGraph)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📄 Data Source

This project uses **publicly available enterprise documentation** from the
**37signals (Basecamp) Employee Handbook**, stored as Markdown files.

> The dataset simulates an internal enterprise knowledge base for realistic RAG system development and evaluation.

---

## 🧪 Experiments & Design Decisions

### Chunking Strategies Tested

During development, multiple chunking strategies were evaluated to determine the optimal approach for enterprise documentation.

**1. Fixed-size chunking**
Text was split into equal token windows. This approach often broke semantic boundaries and reduced answer quality.

**2. Sliding window chunking**
Overlapping token windows were created. While this prevented boundary information loss, it significantly increased index size with only minor accuracy improvements.

**3. Sentence-aware chunking**
Chunks were created using fixed groups of sentences. This preserved grammatical structure but caused topic drift across chunks.

**4. Semantic chunking**
Chunks were generated based on topic change detection using embeddings. This produced highly coherent chunks but was computationally expensive and complex to maintain.

**5. Section-based chunking ✅ Final Choice**
Documents were split using Markdown section headings. This preserved the natural structure of enterprise documentation and yielded the best retrieval accuracy.

---

### Agentic Design Decisions

**Why LangGraph over a linear pipeline?**
A linear pipeline has no way to recover from poor retrieval or hallucinated answers. LangGraph allows the system to loop, branch, and self-correct — handling ambiguous queries, terminology mismatches, and low-quality retrievals gracefully.

**Why separate `llm_calls.py` from `llm.py`?**
The main `Generator` class uses `temperature=0.2` and `max_tokens=512`, tuned for long answer generation. Graders and the router need `temperature=0` and `max_tokens=64` for fast, deterministic classification. Separating them avoids modifying the original class and keeps responsibilities clean.

**Retrieval grader strategy**
Only the top 3 chunks are graded (not all 10). At least 1 relevant chunk is sufficient to proceed — the cross-encoder reranker is strong enough to surface the best material from a partially relevant set. This keeps grading fast and avoids over-penalising partially relevant retrievals.

**Circuit breakers**
Both the retrieval loop (rewrite → retrieve) and the answer loop (retry → retrieve) check `retry_count >= max_retries` before acting on grades. This guarantees the graph always terminates and the user always receives a response — even on completely out-of-scope queries.

**top_k vs top_n separation**
FAISS and the cross-encoder rank differently. FAISS finds geometrically similar vectors; the cross-encoder reads query+chunk together and reasons about meaning. A wider `top_k` (e.g. 20) gives the cross-encoder more candidates to find semantically relevant but lexically distant sections — this was the fix for the moonlighting retrieval failure discovered during testing.

---

## 📊 Evaluation

### Evaluation Suite

The evaluation covers 15 ground-truth questions across 5 categories, measuring 6 metrics:

| Category | Questions | What it tests |
|---|---|---|
| Standard RAG | 5 | Clean vocabulary match, single-section answers |
| Vocabulary Mismatch | 3 | Query terms differ from document section names |
| Multi-section | 2 | Answer requires pulling from multiple sections |
| Out-of-scope | 3 | No answer in docs — system should say "I don't know" |
| Direct/Chitchat | 2 | Router should bypass retrieval entirely |

**Metrics computed:**

| Metric | What it measures |
|---|---|
| Precision@K | Fraction of retrieved sections that are relevant |
| Recall@K | Fraction of relevant sections successfully retrieved |
| Keyword Score | Fraction of expected answer keywords present |
| Answer Grounded | Consistency between retrieval outcome and generated answer |
| Router Accuracy | Correct classification of query type |
| Rewrite Accuracy | Whether query rewriter fired on genuinely failed retrievals |

### Results (k=10)

```
Metric                   Linear V1    LangGraph V2     Delta
────────────────────── ──────────── ────────────── ─────────
Precision@10                  0.07           0.07    0.00  →
Recall@10                     0.59           0.59    0.00  →
Keyword Score                 0.62           0.60   -0.02  →
Answer Grounded               0.67           0.67    0.00  →
Router Accuracy               0.87           1.00   +0.13  ↑
Rewrite Accuracy              0.00           1.00   +1.00  ↑
```

### Interpreting the Results

**Retrieval metrics are identical across both pipelines — and that is expected.** Both versions use the same FAISS index and cross-encoder reranker. The LangGraph extension adds intelligence *around* retrieval, not a different retrieval mechanism. Identical retrieval scores confirm the refactoring introduced no regression.

**Router Accuracy +13%** reflects that LangGraph correctly classifies 100% of query types (retrieve vs direct), while the linear pipeline has no routing at all and scores based purely on always defaulting to retrieval — failing on the 2 direct/chitchat queries.

**Rewrite Accuracy +100%** is the most meaningful improvement. Linear V1 scores 0.00 because it has no query rewriting capability — not because it failed, but because the feature doesn't exist. LangGraph scored 1.00, meaning every rewrite attempt was triggered on a query where the initial retrieval genuinely failed. The circuit breaker correctly fired on all 3 out-of-scope queries after exhausting retries.

### Known Failure Cases

**Q3 — Career growth:** Reranker returns 4 `"Introduction"` sections with low scores. `"Mastery & Titles"` and `"Pay & Promotions"` exist in the index but don't surface within top-10 FAISS results. Fixable with `top_k=20`.

**Q6 — "Are employees allowed to moonlight?":** The word *moonlight* doesn't map semantically to the `"Not OK"` section in embedding space. Contrast with Q7 (*"Can we work in another company?"*) which retrieves `"Not OK"` at score 6.20. Classic vocabulary mismatch where the retrieval grader passes through because 1/3 sampled chunks appears loosely relevant.

**Q8 — "Does 37signals allow outside employment?":** The phrase *outside employment* has no semantic overlap with any section in the 37signals handbook. Would require a better embedding model or synonym expansion to fix.

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Embeddings | Hugging Face (`all-MiniLM-L6-v2`) |
| Vector Database | FAISS |
| Reranker | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM | Groq – LLaMA-3.1-8B |
| Agentic Orchestration | LangGraph |
| API Framework | FastAPI |
| UI | Streamlit |
| Evaluation | Precision@K, Recall@K, Keyword Score, Groundedness, Router Accuracy |

---

## ⚙️ Getting Started

### Requirements

```bash
git clone https://github.com/KrishnaSA05/Enterprise-Knowledge-Assistant.git
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

---

### Running the Project

#### Option 1 — Original Linear Pipeline (Version 1)
```bash
python main.py
```

#### Option 2 — LangGraph Agentic Pipeline (Version 2)
```bash
python main_graph.py
```

#### Option 3 — FastAPI Backend
```bash
uvicorn api.main:app --reload
```
API docs available at `http://localhost:8000/docs`

#### Option 4 — Streamlit UI
```bash
streamlit run frontend/app.py
```

---

### Running Evaluation

```bash
# Evaluate LangGraph pipeline only (default)
python evaluation/run_evaluation.py

# Evaluate original linear pipeline only
python evaluation/run_evaluation.py --pipeline linear

# Side-by-side comparison of both pipelines
python evaluation/run_evaluation.py --pipeline both --k 10
```

---

### API Usage

**POST** `/query`

```json
{
  "question": "What benefits does the company offer?",
  "department": null,
  "top_k": 10,
  "top_n": 4,
  "max_retries": 2
}
```

**Response**

```json
{
  "question": "What benefits does the company offer?",
  "answer": "The company offers...",
  "query_type": "retrieve",
  "rewritten_query": null,
  "retrieval_grade": "relevant",
  "answer_grade": "faithful",
  "retries": 0,
  "sources": [
    {
      "source_file": "benefits.md",
      "section": "Employee Benefits",
      "rerank_score": 0.9142
    }
  ]
}
```

---

## 🔮 Future Work

- **Hybrid Search** — Combine BM25 sparse retrieval with dense FAISS retrieval using Reciprocal Rank Fusion to address vocabulary mismatch failures
- **HyDE** — Hypothetical Document Embeddings for improved recall on vague or indirect queries
- **RAGAS Evaluation** — Add semantic answer faithfulness and context precision metrics beyond section-matching
- **Qdrant Migration** — Replace FAISS with Qdrant for metadata filtering and production-ready vector search
- **Multi-turn Conversation** — Add conversation memory for follow-up question handling
- **LangSmith Tracing** — Instrument graph execution for full per-query observability
- **Stronger Embedding Model** — Upgrade from `all-MiniLM-L6-v2` to `all-mpnet-base-v2` to improve semantic coverage on vocabulary mismatch cases
