# 📘 Enterprise Knowledge Assistant (RAG System)

> **Retrieval-Augmented Generation (RAG) system for querying enterprise documentation using Large Language Models**

---

## 🔍 Overview

The **Enterprise Knowledge Assistant** is an end-to-end **Retrieval-Augmented Generation (RAG)** system that enables natural language querying over enterprise documents such as employee handbooks, policies, and operational guidelines.

The system combines **semantic retrieval, reranking, and grounded LLM generation** to provide accurate, source-cited answers while minimizing hallucinations.

This project was built from scratch without LangChain to deeply understand every stage of the RAG pipeline. A LangChain version is also provided for comparison.

---

## 🚀 Key Features

- 📄 Section-aware ingestion of Markdown enterprise documents  
- 🔎 Dense semantic retrieval using Hugging Face embeddings  
- 🏆 Cross-encoder reranking for improved precision  
- 🤖 Low-latency LLM inference using Groq (LLaMA-3.1-8B)  
- 📚 Source-grounded answers with citations  
- ⚡ Persistent FAISS vector index for fast startup  
- 🌐 FastAPI backend for serving queries  
- 📊 Quantitative evaluation using Precision@K and Recall@K  

---

## 🧠 System Architecture
```
User Query
   ↓
Query Embedding (MiniLM)
   ↓
FAISS Vector Search (Top-K)
   ↓
Cross-Encoder Reranker (Top-N)
   ↓
Groq LLaMA-3.1-8B
   ↓
Answer + Source Citations
```
---

## 🗂️ Project Structure
```
Enterprise-Knowledge-Assistant/
│
├── ingestion/
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│
├── retrieval/
│   ├── vector_store.py
│   ├── retriever.py
│   ├── reranker.py
│
├── generation/
│   ├── prompts.py
│   ├── llm.py
│
├── evaluation/
│   ├── metrics.py
│
├── api/
│   └── main.py
│
├── frontend/
│   └── app.py
│
├── artifacts/
│   ├── faiss.index
│   └── faiss.meta
│
├── data/
│   └── *.md
│
├── ENA_with_langchain  # LangChain baseline (single-file version)
├── main.py
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
During development, multiple chunking strategies were evaluated to determine the optimal approach for enterprise documentation.

Chunking Strategies Tested

1. Fixed-size chunking
Text was split into equal token windows. This approach often broke semantic boundaries and reduced answer quality.

2. Sliding window chunking
Overlapping token windows were created. While this prevented boundary information loss, it significantly increased index size with only minor accuracy improvements.

3. Sentence-aware chunking
Chunks were created using fixed groups of sentences. This preserved grammatical structure but caused topic drift across chunks.

4. Semantic chunking
Chunks were generated based on topic change detection using embeddings. This produced highly coherent chunks but was computationally expensive and complex to maintain.

5. Section-based chunking (Final Choice)
Documents were split using Markdown section headings. This preserved the natural structure of enterprise documentation and yielded the best retrieval accuracy.

## ⚙️ Tech Stack

| Component | Technology |
|---------|------------|
| Language | Python |
| Embeddings | Hugging Face (`all-MiniLM-L6-v2`) |
| Vector Database | FAISS |
| Reranker | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM | Groq – LLaMA-3.1-8B |
| API Framework | FastAPI |
| Evaluation | Precision@K, Recall@K |

---

## ⚙️ Getting Started

### 2.1 Requirements
```
git clone https://github.com/KrishnaSA05/Enterprise-Knowledge-Assistant.git
pip install -r requirements.txt
```

### 2.2 How to run

#### 2.2.1 Run End-to-End Pipeline
```
python main.py
```

#### 2.2.2 Run Streamlit UI
```
streamlit run frontend/app.py
```
