"""
Service layer for the RAG pipeline.
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Union
from io import BytesIO

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Importamos la nueva integración para MongoDB
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from werkzeug.datastructures import FileStorage


# --- CONFIGURACIÓN ---
# La conexión a la DB se debe tomar de tu módulo `.database` o de variables de entorno
# Aquí asumimos que tienes una conexión de PyMongo disponible.
# IMPORTANTE: Reemplaza con tus valores reales de conexión
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("DB_NAME","chatbot_financiero") # Nombre de tu base de datos
MONGODB_COLLECTION = os.environ.get("MONGO_COLLECTIONS","document_vectors") # Colección donde se guardarán los vectores
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.environ.get("ATLAS_VECTOR_SEARCH_INDEX_NAME", "vector_index") # Nombre de tu índice en Atlas

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDYG9sBJ8btds4TaFhd1C0ZkFwA2Xc8224"

# Environment
os.environ.setdefault("USER_AGENT", "EnigmaCodersRAG/0.1")

# --- COMPONENTES GLOBALES DEL PIPELINE ---
VECTOR_STORE = None
LLM = None
SYSTEM_PROMPT = ""


def _build_pipeline():
    """Inicializa embeddings, conexión a MongoDB, y el LLM."""
    global VECTOR_STORE, LLM, SYSTEM_PROMPT

    # 1. Embeddings con fallback
    try:
        embeddings_ = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    except Exception:
        embeddings_ = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 2. Conexión a MongoDB
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[MONGODB_COLLECTION]
    except Exception as e:
        print(f"Error al conectar con MongoDB: {e}")
        # En un entorno real, deberías fallar o usar un fallback
        raise

    # 3. Vector Store de MongoDB Atlas
    # Asegúrate de que el índice 'vector_index' exista en MongoDB Atlas
    VECTOR_STORE = MongoDBAtlasVectorSearch(
        embedding=embeddings_,
        collection=collection,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        relevance_score_fn="cosine", # Función de similitud
    )

    # 4. LLM
    LLM = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # 5. System Prompt
    SYSTEM_PROMPT = (
        "Eres un asistente RAG experto. Responde de forma concisa y apóyate "
        "exclusivamente en los fragmentos del documento proporcionados como contexto."
    )

    # Nota: Ya no cargamos el PDF inicial aquí. Se cargará vía el nuevo endpoint.
    print("✅ Pipeline RAG inicializado con MongoDB Atlas Vector Search.")


def _get_file_loader(file_data: BytesIO, filename: str) -> Union[PyPDFLoader, TextLoader, None]:
    """Retorna el loader adecuado según la extensión del archivo."""
    extension = Path(filename).suffix.lower()
    temp_path = Path(filename)
    
    # Escribir el archivo en un temporal si es necesario (ej: pypdf requiere un path)
    # Sin embargo, para fines de demostración, LangChain a veces puede manejar bytes/archivos.
    # El método más robusto en Flask es guardar temporalmente o usar lectores en memoria.
    
    # Estrategia: Escribir a un archivo temporal para que los loaders de LangChain puedan leerlo.
    file_data.seek(0)
    
    # Crear un archivo temporal con el contenido de BytesIO
    # En un entorno de producción, usa 'tempfile' para mayor seguridad.
    with open(temp_path, "wb") as f:
        f.write(file_data.read())
        
    if extension == ".pdf":
        return PyPDFLoader(str(temp_path))
    elif extension == ".txt":
        return TextLoader(str(temp_path))
    else:
        return None

def _process_and_index_file(file: FileStorage) -> List[Document]:
    """
    Procesa un archivo subido, lo divide en fragmentos y lo indexa en el Vector Store de MongoDB.
    
    Args:
        file: El objeto FileStorage de Flask.
        
    Returns:
        Lista de documentos indexados.
    """
    if not VECTOR_STORE:
        raise RuntimeError("RAG pipeline no inicializado.")

    filename = file.filename
    file_data = BytesIO(file.read())

    # 1. Obtener Loader
    loader = _get_file_loader(file_data, filename)

    if not loader:
        raise ValueError(f"Tipo de archivo no soportado: {filename}")

    # 2. Cargar Documentos
    docs = loader.load()

    # 3. Dividir en Chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # 4. Asignar metadatos útiles
    for split in splits:
        split.metadata["source_filename"] = filename # Agrega el nombre original
        # Eliminar el archivo temporal si se creó, para evitar contaminación
        if Path(filename).exists():
            Path(filename).unlink() 
            
    # 5. Indexar en MongoDB Atlas
    if splits:
        VECTOR_STORE.add_documents(splits)
        print(f"✅ Indexados {len(splits)} chunks de '{filename}' en MongoDB.")
    
    return splits


# Build once
_build_pipeline()


# Función para responder preguntas (sin cambios importantes)
def answer_question(question: str) -> Dict[str, Any]:
    """Busca en el Vector Store y genera una respuesta usando el LLM."""
    if not question or not question.strip():
        return {"error": "Question is required"}

    # 1. Recuperar fuentes desde MongoDB Atlas Vector Search
    try:
        # Usa similarity_search del vector store
        sources = VECTOR_STORE.similarity_search(question, k=3) 
    except Exception as e:
        print(f"Error en la búsqueda de similitud: {e}")
        sources = []

    # 2. Construir contexto
    context_docs = sources
    context_text = "\n\n".join(doc.page_content for doc in context_docs)

    # 3. Componer mensajes y llamar al LLM
    messages = [
        SystemMessage(content=SYSTEM_PROMPT + "\n\nContexto:\n" + context_text[:15000]),
        HumanMessage(content=question),
    ]

    result = LLM.invoke(messages)
    response = getattr(result, "content", str(result))

    # 4. Serializar fuentes
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