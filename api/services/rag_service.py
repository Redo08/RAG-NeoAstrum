"""
Service layer for the RAG pipeline.
"""
import os
import numpy as np
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
import tempfile
import uuid

#comm
# --- CONFIGURACIÓN ---
# La conexión a la DB se debe tomar de tu módulo `.database` o de variables de entorno
# Aquí asumimos que tienes una conexión de PyMongo disponible.
# IMPORTANTE: Reemplaza con tus valores reales de conexión
MONGODB_URI = os.environ.get("MONGO_URI")
DB_NAME = os.environ.get("DB_NAME","chatbot_financiero") # Nombre de tu base de datos
MONGODB_COLLECTION = os.environ.get("MONGO_COLLECTIONS","document_vectors") # Colección donde se guardarán los vectores
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.environ.get("ATLAS_VECTOR_SEARCH_INDEX_NAME", "vector_index") # Nombre de tu índice en Atlas

# Model actualizado — text-embedding-004 y embedding-001 ya fueron dados de baja por Google
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))  # 768/1536/3072 vía MRL; 768 ahorra espacio en Mongo


# Environment
os.environ.setdefault("USER_AGENT", "EnigmaCodersRAG/0.1")

# --- COMPONENTES GLOBALES DEL PIPELINE ---
VECTOR_STORE = None
LLM = None
SYSTEM_PROMPT = ""


def dot_product_similarity(vec1, vec2):
    """Similitud por producto punto (no normalizado)."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2))

def _build_pipeline():
    """Inicializa embeddings, conexión a MongoDB, y el LLM."""
    global VECTOR_STORE, LLM, SYSTEM_PROMPT

    embeddings_ = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=EMBEDDING_DIM,
    )


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
        relevance_score_fn="dotproduct", # Función de similitud
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


def _get_file_loader(file_data: BytesIO, filename: str) -> tuple[Union[PyPDFLoader, TextLoader, None], Path]:
    """Retorna el loader adecuado y la ruta del temporal creado."""
    extension = Path(filename).suffix.lower()

    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / f"{uuid.uuid4().hex}_{filename}"

    file_data.seek(0)
    with open(temp_path, "wb") as f:
        f.write(file_data.read())

    if extension == ".pdf":
        return PyPDFLoader(str(temp_path)), temp_path
    elif extension == ".txt":
        return TextLoader(str(temp_path)), temp_path
    else:
        temp_path.unlink(missing_ok=True)
        return None, temp_path

def _process_and_index_file(file: FileStorage) -> List[Document]:
    if not VECTOR_STORE:
        raise RuntimeError("RAG pipeline no inicializado.")

    filename = file.filename
    file_data = BytesIO(file.read())

    # 1. Obtener Loader
    loader, temp_path = _get_file_loader(file_data, filename)

    if not loader:
        raise ValueError(f"Tipo de archivo no soportado: {filename}")

    try:
        # 2. Cargar Documentos
        docs = loader.load()

        # 3. Dividir en Chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        splits = splitter.split_documents(docs)

        # 4. Asignar metadatos útiles
        for split in splits:
            split.metadata["source_filename"] = filename

        # 5. Indexar en MongoDB Atlas
        if splits:
            VECTOR_STORE.add_documents(splits)
            print(f"✅ Indexados {len(splits)} chunks de '{filename}' en MongoDB.")

        return splits
    finally:
        # Limpieza garantizada, incluso si algo falla arriba
        temp_path.unlink(missing_ok=True)


# Build once
_build_pipeline()


def answer_question(question: str, k: int = 3, min_score: float = 0.5) -> Dict[str, Any]:
    """
    Busca documentos similares y genera una respuesta usando el LLM.
    Unifica búsqueda vectorial + generación de respuesta.
    """
    if not question or not question.strip():
        return {"error": "Question is required"}

    print(f"💬 Respondiendo pregunta: '{question}'")
    
    try:
        # 1. Búsqueda vectorial con tu función corregida
        sources = search_similar_documents(
            query_text=question,
            k=k,
            min_score=min_score
        )
        
        if not sources:
            # Si no hay fuentes, el LLM responde sin contexto
            print("⚠️ No se encontraron documentos relevantes, respondiendo sin contexto")
            context_text = "No se encontró información específica en los documentos indexados."
        else:
            # 2. Construir contexto desde los resultados
            context_parts = []
            for i, doc in enumerate(sources, 1):
                context_parts.append(
                    f"[Documento {i} - Score: {doc['score']}]\n{doc['content']}"
                )
            context_text = "\n\n".join(context_parts)
            print(f"✅ Contexto construido desde {len(sources)} documentos")

        # 3. Componer mensajes para el LLM
        system_prompt = (
            "Eres un asistente financiero experto. Responde de forma clara y concisa "
            "basándote EXCLUSIVAMENTE en el contexto proporcionado. "
            "Si la información no está en el contexto, indícalo claramente."
        )
        
        messages = [
            SystemMessage(content=f"{system_prompt}\n\n### CONTEXTO:\n{context_text[:15000]}"),
            HumanMessage(content=question),
        ]

        # 4. Llamar al LLM
        try: 
            print("🤖 Invocando LLM...")
            result = LLM.invoke(messages)
            response = getattr(result, "content", str(result))
            print(f"✅ Respuesta generada ({len(response)} caracteres)")
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                return {"error": "Estamos recibiendo muchas preguntas en este momento. Intenta de nuevo en un minuto."}
            raise

        # 5. Serializar fuentes para el frontend
        serialized_sources = [
            {
                "content": doc["content"][:500],  # Limitar tamaño
                "metadata": doc["metadata"],
                "score": doc["score"]
            }
            for doc in sources
        ]

        return {
            "answer": response,
            "sources": serialized_sources,
            "total_sources": len(sources)
        }
        
    except Exception as e:
        print(f"❌ Error en answer_question: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error al procesar la pregunta: {str(e)}"}
        
# ============================================
# AÑADE ESTO A TU rag_service.py
# ============================================

def cosine_similarity(vec1, vec2):
    """Calcula similitud de coseno entre dos vectores."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def search_similar_documents(
    query_text: str,
    k: int = 3,
    source_filter: str = None,
    min_score: float = 0.0
) -> List[Dict[str, Any]]:
    """Búsqueda vectorial nativa usando el índice de Atlas Vector Search."""
    if not VECTOR_STORE:
        raise RuntimeError("RAG pipeline no inicializado.")

    pre_filter = {"source_filename": source_filter} if source_filter else None

    try:
        results = VECTOR_STORE.similarity_search_with_score(
            query_text,
            k=k,
            pre_filter=pre_filter,
        )
    except Exception as e:
        print(f"❌ Error en búsqueda vectorial: {e}")
        import traceback
        traceback.print_exc()
        return []

    output = []
    for doc, score in results:
        if score >= min_score:
            output.append({
                "content": doc.page_content[:1000],
                "metadata": {
                    "source": doc.metadata.get("source", ""),
                    "source_filename": doc.metadata.get("source_filename", ""),
                },
                "score": round(float(score), 4),
            })
    return output
def search_by_metadata(
    filters: Dict[str, Any],
    k: int = 10
) -> List[Dict[str, Any]]:
    """
    Búsqueda simple por metadatos (sin vectores).
    Útil para filtrar por categoría, fuente, etc.
    
    Args:
        filters: Diccionario de filtros (ej: {"metadata.source_filename": "doc.pdf"})
        k: Número máximo de resultados
        
    Returns:
        Lista de documentos que coinciden con los filtros
    """
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[MONGODB_COLLECTION]
        
        # Buscar documentos
        docs = list(collection.find(filters).limit(k))
        
        # Formatear resultados
        results = []
        for doc in docs:
            results.append({
                "content": doc.get("text", doc.get("page_content", ""))[:1000],
                "metadata": doc.get("metadata", {}),
                "score": None  # No hay score en búsqueda por metadatos
            })
        
        print(f"✅ Encontrados {len(results)} documentos por metadatos")
        return results
        
    except Exception as e:
        print(f"❌ Error en búsqueda por metadatos: {e}")
        return []


def get_all_sources() -> List[str]:
    """
    Retorna lista de todos los archivos fuente indexados.
    Útil para mostrar opciones de filtrado.
    
    Returns:
        Lista de nombres de archivos únicos
    """
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[MONGODB_COLLECTION]
        
        # Obtener valores únicos de source_filename
        sources = collection.distinct("metadata.source_filename")
        
        print(f"📚 Fuentes disponibles: {len(sources)}")
        return sources
        
    except Exception as e:
        print(f"❌ Error obteniendo fuentes: {e}")
        return []


def debug_collection_structure():
    """
    Función de debug para inspeccionar la estructura de la colección.
    Útil para verificar cómo están guardados los embeddings.
    """
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[MONGODB_COLLECTION]
        
        # Obtener un documento de ejemplo
        sample_doc = collection.find_one()
        
        if sample_doc:
            print("\n=== 🔍 ESTRUCTURA DE DOCUMENTO ===")
            print(f"Campos disponibles: {list(sample_doc.keys())}")
            
            if "embedding" in sample_doc:
                print(f"✅ Campo 'embedding' encontrado (dim: {len(sample_doc['embedding'])})")
            elif "vector" in sample_doc:
                print(f"✅ Campo 'vector' encontrado (dim: {len(sample_doc['vector'])})")
            else:
                print("⚠️ No se encontró campo de embedding")
            
            if "metadata" in sample_doc:
                print(f"Metadatos: {sample_doc['metadata']}")
            
            print("=================================\n")
        else:
            print("⚠️ La colección está vacía")
            
    except Exception as e:
        print(f"❌ Error en debug: {e}")