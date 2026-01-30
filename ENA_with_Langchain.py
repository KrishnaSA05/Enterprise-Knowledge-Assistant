import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Load docs
loader = DirectoryLoader("data", glob="**/*.md")
docs = loader.load()

# Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})

# LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)


# Prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are an enterprise knowledge assistant.
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {question}
    """
)

# Runnable chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Query
query = "In United states, who is the medical service insurance provider?"
response = chain.invoke(query)
print(response.content)
