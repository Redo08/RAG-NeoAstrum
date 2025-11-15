import { useState, useEffect, useRef } from "react";
import api from "./api";
import "./styles.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [serverStatus, setServerStatus] = useState("checking");
  const [examples, setExamples] = useState([]);
  const messagesEndRef = useRef(null);

  // Verificar estado del servidor al cargar
  useEffect(() => {
    checkServerHealth();
    loadExamples();
  }, []);

  // Auto-scroll al último mensaje
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const checkServerHealth = async () => {
    try {
      const health = await api.healthCheck();
      setServerStatus(health.status === "ok" ? "ready" : "loading");
    } catch (err) {
      setServerStatus("offline");
    }
  };

  const loadExamples = async () => {
    try {
      const data = await api.getExamples();
      setExamples(data.examples);
    } catch (err) {
      console.error("Error cargando ejemplos:", err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput("");
    setError(null);

    // Agregar mensaje del usuario
    setMessages((prev) => [...prev, { type: "user", content: userMessage }]);

    // Mostrar indicador de carga
    setLoading(true);

    try {
      const response = await api.query(userMessage);

      // Agregar respuesta del asistente
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          content: response.answer,
          sources: response.sources,
        },
      ]);
    } catch (err) {
      setError(err.message);
      setMessages((prev) => [
        ...prev,
        {
          type: "error",
          content: err.message,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (example) => {
    setInput(example);
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1>🏛️ RAG Jurídico</h1>
          <p>Consulta sobre la Constitución Política de Colombia</p>
          <div className="server-status">
            <span className={`status-indicator ${serverStatus}`}></span>
            <span className="status-text">
              {serverStatus === "ready" && "Conectado"}
              {serverStatus === "loading" && "Cargando..."}
              {serverStatus === "offline" && "Desconectado"}
              {serverStatus === "checking" && "Verificando..."}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Messages */}
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome">
              <h2>👋 ¡Bienvenido!</h2>
              <p>Haz una pregunta sobre la Constitución Política de Colombia</p>

              {examples.length > 0 && (
                <div className="examples">
                  <h3>💡 Ejemplos de consultas:</h3>
                  <div className="examples-grid">
                    {examples.slice(0, 4).map((example, idx) => (
                      <button
                        key={idx}
                        className="example-btn"
                        onClick={() => handleExampleClick(example)}
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.type}`}>
              <div className="message-content">
                {msg.type === "user" && (
                  <div className="message-header">
                    <span className="message-icon">👤</span>
                    <span className="message-label">Tú</span>
                  </div>
                )}
                {msg.type === "assistant" && (
                  <div className="message-header">
                    <span className="message-icon">⚖️</span>
                    <span className="message-label">Asistente Jurídico</span>
                  </div>
                )}
                {msg.type === "error" && (
                  <div className="message-header">
                    <span className="message-icon">⚠️</span>
                    <span className="message-label">Error</span>
                  </div>
                )}

                <div className="message-text">{msg.content}</div>

                {/* Sources */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources">
                    <h4>📚 Fuentes citadas:</h4>
                    {msg.sources.map((source, sidx) => (
                      <div key={sidx} className="source-card">
                        <div className="source-header">
                          <span className="source-page">
                            Página {source.page}
                          </span>
                        </div>
                        <div className="source-content">{source.content}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message assistant">
              <div className="message-content">
                <div className="message-header">
                  <span className="message-icon">⚖️</span>
                  <span className="message-label">Asistente Jurídico</span>
                </div>
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Form */}
        <div className="input-container">
          {messages.length > 0 && (
            <button className="clear-btn" onClick={clearChat}>
              🗑️ Limpiar chat
            </button>
          )}

          <form onSubmit={handleSubmit} className="input-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Escribe tu pregunta aquí..."
              className="input-field"
              disabled={loading || serverStatus !== "ready"}
            />
            <button
              type="submit"
              className="submit-btn"
              disabled={loading || !input.trim() || serverStatus !== "ready"}
            >
              {loading ? "⏳" : "📤"}
            </button>
          </form>

          {serverStatus === "offline" && (
            <div className="error-banner">
              ⚠️ No se puede conectar al servidor. Asegúrate de que el backend
              esté ejecutándose en http://localhost:8000
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
