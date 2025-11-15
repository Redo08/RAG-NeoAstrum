"""
Motor RAG para consultas sobre la Constitución Política de Colombia
Versión adaptada a LangChain 1.x y Qdrant
"""

import os
from pathlib import Path

from langchain.tools import tool
from langchain.agents import initialize_agent, Tool
from langchain.schema import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# ===============================
# CONFIGURACIÓN
# ===============================
os.environ.setdefault("GOOGLE_API_KEY", "TU_GOOGLE_API_KEY_AQUI")
os.environ.setdefault("USER_AGENT", "EnigmaCodersRAG/0.1")

PDF_PATH = Path("constitucion.pdf")

# ===============================
# EMBEDDINGS
# ===============================
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
except Exception:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ===============================
# VECTORSTORE QDRANT (en memoria)
# ===============================
client = QdrantClient(":memory:")

vector_size = len(embeddings.embed_query("sample text"))

if not client.collection_exists("constitucion"):
    client.create_collection(
        collection_name="constitucion",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="constitucion",
    embedding=embeddings,
)

# ===============================
# CARGAR PDF Y SPLIT
# ===============================
loader = PyPDFLoader(str(PDF_PATH))
docs = loader.load()
print(f"✅ Cargados {len(docs)} documentos desde {PDF_PATH.name}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Indexar documentos en Qdrant
_ = vector_store.add_documents(all_splits)
print(f"✅ Indexados {len(all_splits)} fragmentos en Qdrant")

# ===============================
# DEFINIR HERRAMIENTA PARA CONSULTAS
# ===============================
@tool
def retrieve_context(query: str):
    """Recupera información relevante del vectorstore"""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
    )
    return serialized

tools = [
    Tool(
        name="RetrieveContext",
        func=retrieve_context,
        description="Recupera fragmentos relevantes de la constitución"
    )
]

# ===============================
# AGENTE LLM
# ===============================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  # O gemini-1.5-pro según tu cuota

system_prompt = (
    "Eres un abogado constitucionalista experto en la Constitución Política de Colombia. "
    "Responde únicamente usando los fragmentos proporcionados por la herramienta RetrieveContext."
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",  # Tipo de agente recomendado
    verbose=True,
    system_message=system_prompt
)

# ===============================
# CONSULTA DE PRUEBA
# ===============================
query = "¿Cuáles son los derechos de los indígenas?"

print("\n💬 Consulta:", query)
result = agent.run(HumanMessage(content=query))
print("\n✅ Respuesta generada:\n")
print(result)
