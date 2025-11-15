from flask import Flask, request, jsonify
from typing import Any, Dict

# Support both: `python -m api.app` and `python api/app.py`
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from api.rag_service import answer_question
except ModuleNotFoundError:
    # If still fails, try direct import from same directory
    from rag_service import answer_question


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


@app.route("/ask", methods=["POST"])
def ask() -> Any:
    data = request.get_json(silent=True) or {}
    question = data.get("question")
    result = answer_question(question)
    status = 400 if "error" in result else 200
    return jsonify(result), status


if __name__ == "__main__":
    # Default port 5000
    app.run(host="0.0.0.0", port=5000)
