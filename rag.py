from langchain_core.tools import tool  # Decorador correcto; antes se intentaba importar "tool" desde json, causando que fuera un módulo/no existente
import bs4
from langchain.agents import AgentState, create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.messages import MessageLikeRepresentation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os



# Import del módulo local (si lo necesitas) renombrado para evitar colisión con la variable llm
try:
    import model as local_model  # noqa: F401
except ImportError:
    local_model = None



if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDQxZCOVDtlt0srL1xyLg4ToFficIlJnhU"

if not os.environ.get("USER_AGENT"):
    # Valor por defecto para evitar warnings, puedes cambiarlo
    os.environ["USER_AGENT"] = "EnigmaCodersRAG/0.1"

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Modelo de embeddings: "models/text-embedding-004" es el vigente.
# Se agrega fallback a "models/embedding-001" si el nuevo no está disponible en tu cuota/región.
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
except Exception:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
client = QdrantClient(":memory:")

vector_size = len(embeddings.embed_query("sample text"))

if not client.collection_exists("test"):
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings,
)
file_path="constitucion.pdf"
loader = PyPDFLoader(file_path,mode="single")
# Load and chunk contents of the blog
docs = loader.load()
print(f"Loaded {len(docs)} documents from {file_path}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Construct a tool for retrieving context
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  # Puedes usar gemini-1.5-pro si necesitas más capacidad
agent = create_agent(llm, tools, system_prompt=prompt)

query = "Cuales son los derechos de los indigenas"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()