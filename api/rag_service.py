"""
Service layer for the RAG pipeline over constitucion.pdf
Builds embeddings, vector store, and agent once at startup.
Exposes `answer_question(question: str) -> dict` for the API layer.
"""
import os
from pathlib import Path
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


# Environment

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDQxZCOVDtlt0srL1xyLg4ToFficIlJnhU"
os.environ.setdefault("USER_AGENT", "EnigmaCodersRAG/0.1")

PDF_PATH = Path("constitucion.pdf")


def _build_pipeline():
    # Embeddings with fallback
    try:
        embeddings_ = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    except Exception:
        embeddings_ = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    client_ = QdrantClient(":memory:")
    vector_size_ = len(embeddings_.embed_query("sample text"))

    collection = "constitucion"
    if not client_.collection_exists(collection):
        client_.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size_, distance=Distance.COSINE),
        )

    vector_store_ = QdrantVectorStore(
        client=client_,
        collection_name=collection,
        embedding=embeddings_,
    )

    # Load PDF and index
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH.resolve()}")
    docs = PyPDFLoader(str(PDF_PATH)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    if splits:
        vector_store_.add_documents(splits)

    llm_ = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    system_prompt = (
        "Eres un abogado constitucionalista experto en la Constitución Política de Colombia. "
        "Responde de forma concisa y apóyate exclusivamente en los fragmentos del documento proporcionados como contexto."
    )

    return vector_store_, llm_, system_prompt


# Build once
VECTOR_STORE, LLM, SYSTEM_PROMPT = _build_pipeline()


def answer_question(question: str) -> Dict[str, Any]:
    if not question or not question.strip():
        return {"error": "Question is required"}

    # Optionally fetch sources separately
    try:
        sources = VECTOR_STORE.similarity_search(question, k=3)
    except Exception:
        sources = []

    # Build context from retrieval
    context_docs = sources
    context_text = "\n\n".join(doc.page_content for doc in context_docs)

    # Compose messages for the chat model
    messages = [
        SystemMessage(content=SYSTEM_PROMPT + "\n\nContexto:\n" + context_text[:15000]),
        HumanMessage(content=question),
    ]

    result = LLM.invoke(messages)
    response = getattr(result, "content", str(result))

    serialized_sources = [
        {
            "metadata": doc.metadata,
            "content": doc.page_content[:5000],
        }
        for doc in sources
    ]

    return {
        "answer": response,
        "sources": serialized_sources,
    }
