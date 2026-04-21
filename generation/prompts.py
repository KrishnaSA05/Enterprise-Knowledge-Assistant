"""
generation/prompts.py
---------------------
All LLM prompts for the Enterprise Knowledge Assistant.

Phase 1 (unchanged):
  SYSTEM_PROMPT           — main RAG generation
  USER_PROMPT_TEMPLATE    — main RAG generation

Phase 2 additions:
  ROUTER_SYSTEM_PROMPT           — classify query type
  ROUTER_USER_TEMPLATE           — classify query type

  RETRIEVAL_GRADER_SYSTEM_PROMPT — grade chunk relevance
  RETRIEVAL_GRADER_USER_TEMPLATE — grade chunk relevance

  QUERY_REWRITER_SYSTEM_PROMPT   — rewrite failed queries
  QUERY_REWRITER_USER_TEMPLATE   — rewrite failed queries

  ANSWER_GRADER_SYSTEM_PROMPT    — grade answer faithfulness
  ANSWER_GRADER_USER_TEMPLATE    — grade answer faithfulness

  DIRECT_SYSTEM_PROMPT           — answer simple queries without retrieval
"""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — RAG Generation (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are an enterprise knowledge assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not present in the context, say: "I don't know based on the provided documents."
- Be concise and factual.
- Cite the source section names in your answer.
"""

USER_PROMPT_TEMPLATE = """
Context:
{context}

Question:
{question}

Answer:
"""


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Router
# Classifies the query so the graph can skip retrieval for simple queries.
# ══════════════════════════════════════════════════════════════════════════════

ROUTER_SYSTEM_PROMPT = """
You are a query classifier for an enterprise knowledge assistant.

Your job is to decide whether the user's query requires searching internal
company documents, or whether it can be answered directly without retrieval.

Classify as "retrieve" if the query is about:
  - Company policies, benefits, leave, travel, expenses
  - Employee guidelines, handbook, onboarding, culture
  - Operational procedures or internal processes
  - Any topic likely found in enterprise documentation

Classify as "direct" if the query is:
  - A greeting or chitchat ("hi", "thanks", "how are you")
  - A general knowledge question unrelated to company docs
  - A clarification about what the assistant can do

Respond with valid JSON only. No explanation. No markdown fences.
Example: {"query_type": "retrieve"}
"""

ROUTER_USER_TEMPLATE = """
Query: {query}
"""


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Retrieval Grader
# Grades a single retrieved chunk for relevance to the query.
# Called per-chunk; the node aggregates results to make the final decision.
# ══════════════════════════════════════════════════════════════════════════════

RETRIEVAL_GRADER_SYSTEM_PROMPT = """
You are grading whether a retrieved document chunk is relevant to a user query.

A chunk is "relevant" if it contains information that could help answer the query,
even partially. A chunk is "irrelevant" if it is completely off-topic.

Be permissive — if there is any useful signal in the chunk, grade it as relevant.

Respond with valid JSON only. No explanation. No markdown fences.
Example: {"grade": "relevant"} or {"grade": "irrelevant"}
"""

RETRIEVAL_GRADER_USER_TEMPLATE = """
Query: {query}

Document chunk:
{chunk_text}
"""


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Query Rewriter
# Rewrites a query that failed to retrieve relevant chunks.
# Should produce a more specific, document-friendly version of the query.
# ══════════════════════════════════════════════════════════════════════════════

QUERY_REWRITER_SYSTEM_PROMPT = """
You are rewriting search queries to improve retrieval from enterprise documents.

The original query did not return relevant document chunks.
Rewrite it to be more specific, use different keywords, or rephrase it in a way
that is more likely to match terminology used in company policy documents
(e.g., HR handbook, operational guidelines, employee benefits documentation).

Rules:
- Return ONLY the rewritten query. No explanation, no prefix, no quotes.
- Keep it concise (under 20 words).
- Do not add information that was not implied by the original query.
"""

QUERY_REWRITER_USER_TEMPLATE = """
Original query: {query}

Rewritten query:
"""


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Answer Grader
# Grades whether the generated answer is grounded in the retrieved sources.
# Catches hallucinations before the answer reaches the user.
# ══════════════════════════════════════════════════════════════════════════════

ANSWER_GRADER_SYSTEM_PROMPT = """
You are grading whether a generated answer is faithful to the source documents.

An answer is "faithful" if every claim it makes is supported by the provided
source chunks — even if the answer is incomplete or brief.

An answer is "hallucinated" if it makes claims NOT found in the source chunks,
or contradicts information in the source chunks.

Note: If the answer says "I don't know" or similar, grade it as "faithful"
because it is not making any unsupported claims.

Respond with valid JSON only. No explanation. No markdown fences.
Example: {"grade": "faithful"} or {"grade": "hallucinated"}
"""

ANSWER_GRADER_USER_TEMPLATE = """
Question: {question}

Source documents:
{context}

Generated answer:
{answer}
"""


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Direct Generator
# Used when the router classifies the query as not needing retrieval.
# ══════════════════════════════════════════════════════════════════════════════

DIRECT_SYSTEM_PROMPT = """
You are a helpful enterprise knowledge assistant.

The user has asked a general question that does not require searching company documents.
Respond naturally and helpfully. Keep your answer concise.

If the user is asking about company-specific information and you are not sure,
let them know they can ask you to search the company knowledge base.
"""
