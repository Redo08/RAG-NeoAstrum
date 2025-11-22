# EnigmaCoders - Sistema RAG para Consultas Jurídicas

Sistema de Recuperación y Generación Aumentada (RAG) desarrollado para proporcionar respuestas precisas basadas en documentos jurídicos indexados, utilizando embeddings vectoriales y búsqueda semántica.

## 🚀 Tecnologías Utilizadas

### Backend

- **Python 3.x** - Lenguaje principal del backend
- **Flask 3.0.0** - Framework web para la API REST
- **Flask-CORS 4.0.0** - Manejo de políticas CORS
- **LangChain 0.3.14+** - Framework para aplicaciones con LLM
- **LangChain Community** - Conectores y utilidades adicionales
- **LangChain MongoDB** - Integración de MongoDB como vector store
- **Google Generative AI (Gemini)** - Modelos de lenguaje y embeddings
  - `gemini-2.5-pro` - Generación de respuestas
  - `gemini-2.5-flash` - Procesamiento rápido de conversaciones
  - `text-embedding-004` - Generación de embeddings vectoriales
- **MongoDB Atlas** - Base de datos NoSQL con capacidades de búsqueda vectorial
- **PyMongo** - Driver oficial de MongoDB para Python
- **PyPDF** - Procesamiento de documentos PDF
- **python-dotenv** - Gestión de variables de entorno
- **Gunicorn 22.0.0** - Servidor WSGI para producción


### Testing

- **pytest** - Framework de pruebas unitarias
- **unittest.mock** - Mocking para pruebas aisladas

## 📊 Arquitectura del Sistema

### Flujo de Datos RAG

```
Usuario → Frontend (React) → API Flask → RAG Service
                                            ↓
                                   Búsqueda Vectorial
                                            ↓
                          MongoDB Atlas Vector Store
                                            ↓
                              Documentos Relevantes
                                            ↓
                                    LLM (Gemini)
                                            ↓
                                Respuesta Generada
                                            ↓
                          Frontend → Usuario
```

## 🧠 Metodología de Embeddings

### Generación de Embeddings

El sistema utiliza el modelo **Google Text Embedding 004** (`models/text-embedding-004`) para convertir texto en representaciones vectoriales de alta dimensionalidad.

#### Proceso de Indexación de Documentos

1. **Carga de Documentos**

   - Soporte para archivos PDF y TXT
   - Utiliza `PyPDFLoader` y `TextLoader` de LangChain
   - Endpoint: `POST /chat/upload`

2. **División en Fragmentos (Chunking)**

   - **Estrategia**: `RecursiveCharacterTextSplitter`
   - **Tamaño de chunk**: 500 caracteres
   - **Overlap**: 50 caracteres (para mantener contexto entre fragmentos)
   - **Separadores jerárquicos**:
     1. Párrafos (`\n\n`)
     2. Líneas (`\n`)
     3. Puntos finales (`.`)
     4. Espacios (` `)
     5. Caracteres individuales (fallback)

3. **Generación de Embeddings**

   - Cada fragmento se convierte en un vector de embeddings
   - Modelo: `models/text-embedding-004` (con fallback a `models/embedding-001`)
   - Los vectores se almacenan junto con metadatos en MongoDB

4. **Almacenamiento en MongoDB Atlas**
   - Colección: `document_vectors` (configurable)
   - Índice vectorial: `vector_index`
   - Metadatos incluidos:
     - `source_filename`: Nombre del archivo original
     - `source`: Ruta del documento
     - `text`: Contenido del fragmento

## 🔍 Método de Búsqueda

### Búsqueda Vectorial con Similitud

El sistema implementa búsqueda vectorial utilizando **similitud por producto punto (dot product)** para encontrar los documentos más relevantes.

#### Proceso de Búsqueda

1. **Generación del Query Embedding**

   - La pregunta del usuario se convierte en un vector usando el mismo modelo de embeddings
   - Garantiza consistencia dimensional con los documentos indexados

2. **Búsqueda Vectorial**

   - Función: `search_similar_documents()`
   - Parámetros configurables:
     - `k`: Número de documentos a recuperar (default: 3)
     - `min_score`: Umbral mínimo de similitud (default: 0.0)
     - `source_filter`: Filtro opcional por archivo fuente

3. **Cálculo de Similitud**

   ```python
   similarity = dot_product(query_embedding, document_embedding)
   ```

   - **Dot Product**: Mide la similitud direccional entre vectores
   - Alternativa implementada: Similitud de Coseno (normalizada)

4. **Ranking y Filtrado**

   - Los documentos se ordenan por score de similitud (descendente)
   - Se aplica el umbral `min_score`
   - Se retornan los top `k` documentos

5. **Construcción del Contexto**
   - Los fragmentos recuperados se concatenan
   - Se limita el contexto a 15,000 caracteres para el LLM
   - Cada fragmento incluye su score de relevancia

### Generación de Respuestas

**Función**: `answer_question()`

1. Realiza búsqueda vectorial
2. Construye contexto desde los documentos relevantes
3. Compone prompts para el LLM:
   - **System Prompt**: Define el rol del asistente y restricciones
   - **Context**: Fragmentos de documentos recuperados
   - **User Query**: Pregunta original
4. Invoca Gemini 2.5 Pro para generar respuesta
5. Retorna respuesta + fuentes utilizadas

## 🧪 Pruebas Unitarias

### Framework de Testing

El proyecto utiliza **pytest** para pruebas unitarias con **mocking** de dependencias externas.

### Estructura de Pruebas

**Archivo**: `api/test/test_chat_endpoint.py`

#### Pruebas Implementadas

1. **`test_chat_message_success`**

   - Verifica el flujo exitoso de envío de mensajes
   - Mock del servicio RAG
   - Valida respuesta y session_id

2. **`test_chat_message_missing_message`**

   - Prueba validación de parámetros requeridos
   - Espera error 400 cuando falta el mensaje

3. **`test_chat_message_process_error`**

   - Manejo de errores del servicio RAG
   - Simula fallos en procesamiento

4. **`test_chat_message_internal_error`**
   - Prueba manejo de excepciones internas

### Configuración de Testing

**Archivo**: `api/test/conftest.py`

- Define fixtures reutilizables
- Configura cliente de prueba de Flask

### Ejecutar Pruebas

```powershell
# Activar entorno virtual
.\venv\Scripts\Activate.ps1

# Ejecutar todas las pruebas
pytest -v

# Ejecutar con cobertura
pytest --cov=api --cov-report=html
```

## 📋 Instalación y Configuración

### Requisitos Previos

- Python 3.8+
- MongoDB Atlas (con índice vectorial configurado)
- API Key de Google Generative AI
- Node.js 18+ (para frontend)

### Backend Setup

```powershell
# Clonar repositorio
git clone https://github.com/Mandara2/EnigmaCoders.git
cd EnigmaCoders

# Crear entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno (.env)
# MONGO_URI=mongodb+srv://...
# DB_NAME=chatbot_financiero
# MONGO_COLLECTIONS=document_vectors
# ATLAS_VECTOR_SEARCH_INDEX_NAME=vector_index
# GOOGLE_API_KEY=your_api_key

# Ejecutar servidor de desarrollo
python -m api.app
```


## 🎯 Endpoints API

### `GET /health`

Verifica el estado del servidor

### `POST /chat/upload`

Sube y procesa documentos PDF/TXT

- **Body**: `multipart/form-data`
- **Campo**: `files[]` (múltiples archivos)

### `POST /chat/start`

Inicia una nueva sesión de conversación (opcional)

- **Body**: `{ "user_id": "optional_user_id" }`

### `POST /chat/message`

Envía un mensaje y obtiene respuesta RAG

- **Body**: `{ "session_id": "...", "message": "...", "user_id": "..." }`

## 📊 Base de Datos

### Colecciones MongoDB

1. **`document_vectors`**

   - Almacena fragmentos de documentos con embeddings
   - Campos: `text`, `embedding`, `source_filename`, `source`

2. **`sessions`**

   - Gestiona sesiones de usuario
   - Campos: `user_id`, `current_node_id`, `created_at`, `completed`

3. **`responses`**
   - Almacena historial de conversaciones

### Índice Vectorial (MongoDB Atlas)

```javascript
{
  "type": "vectorSearch",
  "fields": [{
    "type": "vector",
    "path": "embedding",
    "numDimensions": 768, // o la dimensión de text-embedding-004
    "similarity": "dotProduct"
  }]
}
```

## 🧪 Sección de Pruebas de Preguntas

### Objetivo

Esta sección está diseñada para documentar y evaluar el rendimiento del sistema RAG con diferentes tipos de preguntas.

---

### Categorías de Preguntas a Evaluar

#### 1. Preguntas Directas (Factual Retrieval)

**Objetivo**: Verificar recuperación de información específica

- [ ] ¿Qué establece el artículo X sobre...?
- [ ] ¿Cuál es la definición de [término jurídico]?
- [ ] ¿Qué requisitos se necesitan para...?

**Criterios de éxito**:

- Respuesta precisa y concisa
- Citas directas del documento
- Score de similitud > 0.7

---

#### 2. Preguntas Comparativas

**Objetivo**: Evaluar capacidad de relacionar múltiples fragmentos

- [ ] ¿Cuál es la diferencia entre A y B?
- [ ] ¿Cómo se relaciona el artículo X con el artículo Y?
- [ ] ¿Qué cambió entre la versión anterior y actual?




**Criterios de éxito**:

- Solicita aclaración cuando es necesario
- Proporciona respuesta general si el contexto lo permite
- No inventa información

---

#### 5. Preguntas Fuera de Contexto

**Objetivo**: Verificar que el sistema no alucine respuestas

- [ ] ¿Cuál es el precio de...? (información no en documentos)
- [ ] ¿Quién ganó las elecciones de 2024?
- [ ] ¿Cuál es la receta de...?

**Criterios de éxito**:

- Responde: "No se encontró información en los documentos"
- No genera respuestas inventadas
- Score de similitud < 0.4

---

### Métricas de Evaluación

#### Precisión de Búsqueda

```
Precisión = Documentos Relevantes Recuperados / Total Documentos Recuperados
```

#### Recall (Cobertura)

```
Recall = Documentos Relevantes Recuperados / Total Documentos Relevantes en DB
```

#### Score Promedio de Similitud

- **Excelente**: > 0.8
- **Bueno**: 0.6 - 0.8
- **Aceptable**: 0.4 - 0.6
- **Pobre**: < 0.4

---

### 📊 Resultados de Pruebas

#### Sesión de Pruebas #1

**Fecha**: ******\_******  
**Documentos indexados**: **\_\_\_**  
**Total de preguntas probadas**: **\_\_\_**

##### Resultados Generales

- ✅ Preguntas exitosas: \_**\_ / \_\_**
- ❌ Preguntas fallidas: \_**\_ / \_\_**
- 🟡 Respuestas parciales: \_**\_ / \_\_**

##### Preguntas que Funcionaron Bien

| #   | Pregunta | Score | Fuentes | Comentario |
| --- | -------- | ----- | ------- | ---------- |
| 1   |          |       |         |            |
| 2   |          |       |         |            |
| 3   |          |       |         |            |

##### Preguntas que NO Funcionaron

| #   | Pregunta | Problema Identificado | Mejora Propuesta |
| --- | -------- | --------------------- | ---------------- |
| 1   |          |                       |                  |
| 2   |          |                       |                  |
| 3   |          |                       |                  |

---

### 🔧 Casos Límite a Probar

1. **Documentos muy largos**: ¿Se fragmentan correctamente?
2. **Términos técnicos**: ¿Reconoce terminología jurídica específica?
3. **Sinónimos**: ¿Encuentra "contrato" cuando se busca "convenio"?
4. **Negaciones**: ¿Entiende "que no esté permitido"?
5. **Fechas y números**: ¿Recupera información temporal precisa?
6. **Múltiples idiomas**: ¿Maneja texto en español correctamente?

---

### 📈 Mejoras Iterativas

#### Basado en Resultados de Pruebas

**Si las preguntas directas fallan**:

- [ ] Ajustar `chunk_size` y `chunk_overlap`
- [ ] Revisar `min_score` threshold
- [ ] Aumentar `k` (número de documentos recuperados)

**Si las preguntas complejas fallan**:

- [ ] Implementar re-ranking de resultados
- [ ] Aumentar tamaño del contexto al LLM
- [ ] Considerar multiple queries para una pregunta

**Si hay muchos falsos positivos**:

- [ ] Aumentar `min_score` threshold
- [ ] Implementar filtros de metadatos más específicos
- [ ] Revisar calidad de los documentos indexados

---

### 🎯 Objetivos de Mejora Continua

- [ ] Alcanzar >90% de precisión en preguntas directas
- [ ] Alcanzar >75% de precisión en preguntas complejas
- [ ] Reducir falsos positivos a <5%
- [ ] Responder "no sé" cuando corresponda (>95% de casos)
- [ ] Mantener latencia promedio <2 segundos

---

## 🤝 Contribuciones

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agrega nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto fue desarrollado por **EnigmaCoders** para el hackathon.

---

## 👥 Equipo

**EnigmaCoders** - Desarrolladores del sistema RAG jurídico

## 🔗 Enlaces Útiles

- [LangChain Documentation](https://python.langchain.com/)
- [MongoDB Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/)
- [Google Generative AI](https://ai.google.dev/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**Última actualización**: Noviembre 2025
