from flask import Flask, request, jsonify
from typing import Any, Dict
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from api.tree_service import start_conversation, process_message, get_session
except ModuleNotFoundError:
    from tree_service import start_conversation, process_message, get_session


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)