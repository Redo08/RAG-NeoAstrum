from flask import Flask, request, jsonify
from typing import Any, Dict
import sys
from pathlib import Path
from database import Database  # <--- Importamos nuestra conexión
from werkzeug.datastructures import FileStorage

# Para variables de entorno desde .env (opcional)
from dotenv import load_dotenv
import os

load_dotenv()  # cargar variables

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from api.tree_service import start_conversation, process_message, get_session
    from api.rag_service import _process_and_index_file
except ModuleNotFoundError:
    from tree_service import start_conversation, process_message, get_session
    from rag_service import _process_and_index_file



app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/health", methods=["GET"])
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.route("/chat/start", methods=["POST"])
def chat_start() -> Any:
    """
    (OPCIONAL) Inicia una nueva conversación explícitamente
    Ya no es necesario usar este endpoint, pero se mantiene por compatibilidad
    Body: {
        "user_id": "optional_user_id"
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get("user_id", "anonymous")
        
        result = start_conversation(user_id)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat/upload", methods=["POST"])
def rag_upload_docs() -> Any:
    """
    Sube múltiples archivos (.pdf o .txt) y los indexa en el servicio RAG (MongoDB).
    Los archivos deben enviarse en el cuerpo de la solicitud (multipart/form-data)
    con la clave 'files[]'.
    """
    
    try:
        if "files[]" not in request.files:
            return jsonify({"error": "No se encontraron archivos en 'files[]'"}), 400

        files: list[FileStorage] = request.files.getlist("files[]")
        
        if not files:
            return jsonify({"error": "La lista de archivos está vacía"}), 400
        
        indexed_files_count = 0
        indexed_chunks_count = 0
        
        for file in files:
            if file.filename == "":
                continue # Saltar archivos sin nombre
            
            # Verificar extensiones permitidas
            allowed_extensions = {".pdf", ".txt"}
            file_extension = Path(file.filename).suffix.lower()
            
            if file_extension not in allowed_extensions:
                print(f"🚫 Archivo omitido: {file.filename} (Extensión no soportada)")
                continue

            # Procesar e indexar
            print(f"⏳ Procesando e indexando: {file.filename}")
            splits = _process_and_index_file(file)
            indexed_files_count += 1
            indexed_chunks_count += len(splits)

        if indexed_files_count == 0:
             return jsonify({"error": "No se procesó ningún archivo. Asegúrate de que sean .pdf o .txt"}), 400

        return jsonify({
            "message": "Archivos subidos e indexados correctamente",
            "files_processed": indexed_files_count,
            "chunks_indexed": indexed_chunks_count
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"❌ ERROR CRÍTICO en /rag/upload: {e}")
        return jsonify({"error": "Error interno del servidor al procesar los archivos"}), 500


@app.route("/chat/message", methods=["POST"])
def chat_message() -> Any:
    """
    Procesa un mensaje del usuario
    Body: {
        "session_id": "opcional - se crea automáticamente si no existe",
        "message": "mensaje del usuario",
        "user_id": "opcional - default: anonymous"
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id")  # Puede ser None
        message = data.get("message")
        user_id = data.get("user_id", "anonymous")
        
        if not message or not message.strip():
            return jsonify({"error": "message es requerido"}), 400
        
        result = process_message(session_id, message.strip(), user_id)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat/session/<session_id>", methods=["GET"])
def get_session_info(session_id: str) -> Any:
    """
    Obtiene información de una sesión
    """
    try:
        session = get_session(session_id)
        if not session:
            return jsonify({"error": "Sesión no encontrada"}), 404
        
        # Convertir ObjectId a string
        session["_id"] = str(session["_id"])
        session["created_at"] = session["created_at"].isoformat()
        session["updated_at"] = session["updated_at"].isoformat()
        
        return jsonify(session), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# --- NUEVO ENDPOINT DE ANALÍTICAS CON DEBUG ---
@app.route("/analytics/visited-nodes", methods=["GET"])
def get_visited_nodes():
    """
    Retorna el conteo de nodos visitados formateado para ngx-charts.
    Formato: [{ "name": "inversion", "value": 15 }, ...]
    """
    try:
        print("\n--- 🔍 INICIO DEBUG ANALYTICS ---")
        
        # 1. Verificar conexión y colección
        # IMPORTANTE: ¿Tu colección se llama 'logs'? ¿O 'conversations', 'mensajes'?
        collection_name = "responses" 
        collection = Database.get_collection(collection_name) 
        
        # DEBUG: Contar documentos totales
        total_docs = collection.count_documents({})
        print(f"📊 DEBUG: Total documentos en colección '{collection_name}': {total_docs}")

        if total_docs == 0:
            print(f"⚠ WARNING: La colección '{collection_name}' está vacía o no existe.")
            print("   -> Verifica el nombre en Database.get_collection('NOMBRE_AQUI')")
            return jsonify([]), 200

        # DEBUG: Verificar si el campo existe en algún documento
        query_check = {"question_node_id": {"$exists": True}}
        matched_docs = collection.count_documents(query_check)
        print(f"🎯 DEBUG: Documentos que tienen el campo 'question_node_id': {matched_docs}")

        if matched_docs == 0:
            print("⚠ WARNING: Ningún documento tiene el campo 'question_node_id'.")
            # Imprimimos un ejemplo para ver qué campos TIENE realmente
            sample = collection.find_one()
            print(f"   -> Muestra de un documento real: {sample}")
            return jsonify([]), 200

        # 2. Pipeline de Agregación de MongoDB
        pipeline = [
            # Filtrar (asegurar que no sea nulo)
            { "$match": { "question_node_id": { "$exists": True, "$ne": None } } },
            
            # Agrupar
            { 
                "$group": { 
                    "_id": "$question_node_id", 
                    "count": { "$sum": 1 } 
                } 
            },
            
            # Ordenar
            { "$sort": { "count": -1 } },
            
            # Limitar
            { "$limit": 10 },

            # Proyectar
            { 
                "$project": { 
                    "_id": 0, 
                    "name": "$_id", 
                    "value": "$count" 
                } 
            }
        ]

        print(f"⚙ DEBUG: Ejecutando pipeline...")
        
        # 3. Ejecutar la consulta
        results = list(collection.aggregate(pipeline))
        
        print(f"✅ DEBUG: Resultados finales ({len(results)}): {results}")
        print("--- FIN DEBUG ---\n")
        
        return jsonify(results), 200

    except Exception as e:
        print(f"❌ ERROR CRÍTICO en analytics: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
# ------------------------------------

# Añade este endpoint a tu app.py
@app.route("/search/vector", methods=["POST"])
def vector_search() -> Any:
    """
    Realiza una búsqueda vectorial en documentos indexados (PDF y TXT).
    Compatible con MongoDB local usando similitud de coseno.
    
    Body: {
        "query": "texto de búsqueda",
        "k": 5,  # opcional, número de resultados (default: 5)
        "source_filter": "nombre_archivo.pdf",  # opcional, filtrar por archivo específico
        "min_score": 0.7  # opcional, score mínimo de similitud (default: 0.0)
    }
    
    Response: {
        "results": [
            {
                "content": "fragmento del documento",
                "metadata": {...},
                "score": 0.95
            }
        ],
        "total_results": 5,
        "query": "texto buscado"
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        query_text = data.get("query")
        k = data.get("k", 5)
        source_filter = data.get("source_filter")
        min_score = data.get("min_score", 0.0)
        
        # Validaciones
        if not query_text or not query_text.strip():
            return jsonify({"error": "El campo 'query' es requerido"}), 400
        
        if not isinstance(k, int) or k < 1 or k > 50:
            return jsonify({"error": "El campo 'k' debe ser un entero entre 1 y 50"}), 400
        
        if not isinstance(min_score, (int, float)) or min_score < 0 or min_score > 1:
            return jsonify({"error": "El campo 'min_score' debe estar entre 0 y 1"}), 400
        
        print(f"🔍 Búsqueda vectorial: '{query_text}' (k={k}, min_score={min_score})")
        
        # Realizar búsqueda vectorial usando RAG service
        from rag_service import search_similar_documents
        
        results = search_similar_documents(
            query_text=query_text,
            k=k,
            source_filter=source_filter,
            min_score=min_score
        )
        
        return jsonify({
            "results": results,
            "total_results": len(results),
            "query": query_text,
            "filters": {
                "source": source_filter,
                "min_score": min_score
            }
        }), 200
        
    except Exception as e:
        print(f"❌ ERROR en /search/vector: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)