/**
 * Cliente API para comunicación con el backend RAG
 */

const API_BASE_URL = 'http://localhost:8000';

/**
 * Manejo de errores de API
 */
class APIError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'APIError';
    this.status = status;
  }
}

/**
 * Wrapper para fetch con manejo de errores
 */
async function fetchAPI(endpoint, options = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        errorData.detail || `HTTP error! status: ${response.status}`,
        response.status
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(
      'Error de conexión con el servidor. Verifica que el backend esté ejecutándose.',
      0
    );
  }
}

/**
 * API Client
 */
export const api = {
  /**
   * Verifica el estado del servidor
   */
  async healthCheck() {
    return fetchAPI('/health');
  },

  /**
   * Realiza una consulta al RAG
   * @param {string} question - Pregunta del usuario
   */
  async query(question) {
    return fetchAPI('/query', {
      method: 'POST',
      body: JSON.stringify({ question }),
    });
  },

  /**
   * Obtiene ejemplos de consultas
   */
  async getExamples() {
    return fetchAPI('/examples');
  },

  /**
   * Reconstruye el vectorstore (admin)
   */
  async rebuildVectorstore() {
    return fetchAPI('/rebuild-vectorstore', {
      method: 'POST',
    });
  },
};

export default api;