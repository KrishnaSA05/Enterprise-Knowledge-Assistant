import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END

# -----------------------------
# 1. Load Environment
# -----------------------------
load_dotenv()

# -----------------------------
# 2. Load & Index Documents
# -----------------------------
loader = DirectoryLoader("data", glob="**/*.md")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})

# -----------------------------
# 3. LLM Setup
# -----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

# -----------------------------
# 4. Graph State Definition
# -----------------------------
class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    answer: str
    critique: str
    validated: bool
    retries: int
    confidence: float


# -----------------------------
# 5. Retriever Agent
# -----------------------------
def retriever_agent(state: RAGState):
    docs = retriever.invoke(state["question"])
    return {
        "retrieved_docs": docs
    }


# -----------------------------
# 6. Generator Agent
# -----------------------------
generation_prompt = ChatPromptTemplate.from_template("""
You are an enterprise knowledge assistant.

Use ONLY the provided context to answer the question.
Cite clearly from the context.
If unsure, say you don't know.

Context:
{context}

Question:
{question}
""")

def generator_agent(state: RAGState):
    context = "\n\n".join(
        [doc.page_content for doc in state["retrieved_docs"]]
    )

    response = llm.invoke(
        generation_prompt.format(
            context=context,
            question=state["question"]
        )
    )

    return {
        "answer": response.content
    }


# -----------------------------
# 7. Critic Agent (Validator)
# -----------------------------
critic_prompt = ChatPromptTemplate.from_template("""
You are a strict AI auditor.

Check whether the answer:
1. Uses only the provided context
2. Avoids hallucinations
3. Clearly addresses the question

Respond ONLY with:
PASS - if answer is correct
FAIL - if answer is weak or hallucinated

Answer:
{answer}
""")

def critic_agent(state: RAGState):
    critique = llm.invoke(
        critic_prompt.format(
            answer=state["answer"]
        )
    )

    critique_text = critique.content.strip()

    validated = "PASS" in critique_text

    return {
        "critique": critique_text,
        "validated": validated,
        "retries": state.get("retries", 0) + (0 if validated else 1)
    }


# -----------------------------
# 8. Confidence Scoring Node
# -----------------------------
def confidence_node(state: RAGState):
    if state["validated"]:
        confidence = max(0.5, 1 - (state["retries"] * 0.2))
    else:
        confidence = 0.2

    return {
        "confidence": confidence
    }


# -----------------------------
# 9. Build LangGraph Workflow
# -----------------------------
graph = StateGraph(RAGState)

graph.add_node("retrieve", retriever_agent)
graph.add_node("generate", generator_agent)
graph.add_node("critic", critic_agent)
graph.add_node("confidence", confidence_node)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "critic")

# Conditional Routing
def route_after_critic(state: RAGState):
    if state["validated"] or state["retries"] >= 2:
        return "confidence"
    return "generate"

graph.add_conditional_edges(
    "critic",
    route_after_critic
)

graph.add_edge("confidence", END)

app = graph.compile()

# -----------------------------
# 10. Run Query
# -----------------------------
query = "In United States, who is the medical service insurance provider?"

result = app.invoke({
    "question": query,
    "retries": 0
})

print("\nFinal Answer:\n", result["answer"])
print("\nCritique:\n", result["critique"])
print("\nConfidence Score:", result["confidence"])
