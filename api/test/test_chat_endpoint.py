import json
from unittest.mock import patch

def test_chat_message_success(client):
    # Mock del resultado del RAG
    mock_result = {
        "session_id": "abc123",
        "response": "Hola! ¿En qué puedo ayudarte?"
    }

    # Hacemos patch de process_message
    # Patch target must use full module path 'api.app.process_message'
    with patch("api.app.process_message", return_value=mock_result) as mock_pm:
        
        payload = {
            "session_id": "abc123",
            "message": "Hola",
            "user_id": "juan"
        }

        response = client.post(
            "/chat/message",
            data=json.dumps(payload),
            content_type="application/json"
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["response"] == mock_result["response"]
        assert data["session_id"] == mock_result["session_id"]

        # Verifica que process_message se llamó con los parámetros correctos
        mock_pm.assert_called_once_with("abc123", "Hola", "juan")

def test_chat_message_missing_message(client):
    response = client.post("/chat/message", json={})

    assert response.status_code == 400
    assert response.get_json()["error"] == "message es requerido"
def test_chat_message_process_error(client):
    with patch("api.app.process_message", return_value={"error": "fallo el RAG"}):

        response = client.post("/chat/message", json={
            "message": "hola"
        })

        assert response.status_code == 400
        assert response.get_json()["error"] == "fallo el RAG"

def test_chat_message_internal_error(client):
    # Simulamos que process_message lanza una excepción
    with patch("api.app.process_message", side_effect=Exception("boom")):
        response = client.post("/chat/message", json={"message": "hola"})
        
        assert response.status_code == 500
        assert "boom" in response.get_json()["error"]

