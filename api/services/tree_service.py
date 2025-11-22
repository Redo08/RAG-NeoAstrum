"""
Servicio para navegar el árbol de productos financieros con LangChain
"""
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import requests # <--- ¡NUEVO!
from bson.objectid import ObjectId # <--- ¡NUEVO! Para usar ObjectId más fácil
# 5. Guardar interacción en MongoDB
from ..database import Database

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "chatbot_financiero"

# Cliente MongoDB
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]

# Colecciones
sessions_collection = db["sessions"]
responses_collection = db["responses"]

# LLM
LLM = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# Árbol de productos financieros
with open("api/config.json", "r", encoding="utf-8") as f:
    FINANCIAL_TREE = json.load(f)
    
# Configuración microservicio Notificaciones
NOTIFICATIONS_SERVICE_URL = os.environ.get("NOTIFICATIONS_SERVICE_URL", "https://ms-notifications.onrender.com/send-email") # <--- ¡NUEVO!

def get_node_by_id(node_id: str, tree: Dict = None) -> Optional[Dict]:
    """Encuentra un nodo en el árbol por su ID"""
    if tree is None:
        tree = FINANCIAL_TREE
    
    if tree.get("id") == node_id:
        return tree
    
    # Buscar en categorías
    for categoria in tree.get("categorias", []):
        if categoria.get("id") == node_id:
            return categoria
        
        # Buscar en subcategorías
        for subcategoria in categoria.get("subcategorias", []):
            if subcategoria.get("id") == node_id:
                return subcategoria
            
            # Buscar en tipos
            for tipo in subcategoria.get("tipos", []):
                if tipo.get("id") == node_id:
                    return tipo
    
    return None


def get_children(node: Dict) -> List[Dict]:
    """Obtiene los nodos hijos de un nodo"""
    children = []
    
    if "categorias" in node:
        children = node["categorias"]
    elif "subcategorias" in node:
        children = node["subcategorias"]
    elif "tipos" in node:
        children = node["tipos"]
    
    return children


def create_session(user_id: str = "anonymous") -> str:
    """Crea una nueva sesión"""
    session = {
        "user_id": user_id,
        "current_node_id": "root",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "completed": False
    }
    result = sessions_collection.insert_one(session)
    return str(result.inserted_id)


def get_session(session_id: str) -> Optional[Dict]:
    """Obtiene una sesión por ID"""
    from bson import ObjectId
    try:
        return sessions_collection.find_one({"_id": ObjectId(session_id)})
    except:
        return None


def update_session_node(session_id: str, node_id: str):
    """Actualiza el nodo actual de la sesión"""
    from bson import ObjectId
    sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {
            "$set": {
                "current_node_id": node_id,
                "updated_at": datetime.utcnow()
            },
            "$unset": {
                "pending_restart": ""  # Limpiar flag de reinicio pendiente
            }
        }
    )


def set_pending_restart(session_id: str):
    """Marca la sesión como pendiente de confirmación de reinicio"""
    from bson import ObjectId
    sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {
            "$set": {
                "pending_restart": True,
                "updated_at": datetime.utcnow()
            }
        }
    )


def check_restart_confirmation(user_message: str) -> bool:
    """Verifica si el usuario confirma el reinicio"""
    confirmations = ["sí", "si", "yes", "ok", "dale", "claro", "confirmar", "reiniciar", "empezar de nuevo"]
    negations = ["no", "nope", "continuar", "seguir"]
    
    message_lower = user_message.lower().strip()
    
    # Verificar confirmaciones
    if any(conf in message_lower for conf in confirmations):
        return True
    
    # Verificar negaciones
    if any(neg in message_lower for neg in negations):
        return False
    
    # Usar LLM para casos ambiguos
    prompt = f"""El usuario respondió: "{user_message}"

Se le preguntó si quiere reiniciar la conversación y volver al inicio.

¿El usuario está confirmando que SÍ quiere reiniciar?

Responde SOLO con "SI" o "NO"."""
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_message)
    ]
    
    try:
        result = LLM.invoke(messages)
        response = result.content.strip().upper()
        return "SI" in response or "YES" in response
    except:
        return False


def save_response(session_id: str, question_node_id: str, user_response: str, selected_child_id: str):
    """Guarda la respuesta del usuario"""
    response = {
        "session_id": session_id,
        "question_node_id": question_node_id,
        "user_response": user_response,
        "selected_child_id": selected_child_id,
        "created_at": datetime.utcnow()
    }
    responses_collection.insert_one(response)


def detect_child_match(user_message: str, children: List[Dict]) -> Optional[Dict]:
    """Usa LLM para detectar qué hijo coincide con la respuesta del usuario"""
    if not children:
        return None
    
    # Crear descripción de las opciones
    options_text = "\n".join([
        f"- Opción {i+1}: {child['nombre']} - {child['descripcion']}"
        + (f"\n  Palabras clave: {', '.join(child.get('preguntas_relacionadas', []))}" 
           if child.get('preguntas_relacionadas') else "")
        for i, child in enumerate(children)
    ])
    
    system_prompt = f"""Eres un asistente que ayuda a clasificar las respuestas de usuarios en un chatbot financiero.

Dado el mensaje del usuario, debes identificar cuál de las siguientes opciones coincide mejor con su intención:

{options_text}

Responde ÚNICAMENTE con una de estas opciones:
- El número de la opción (1, 2, 3, etc.) que mejor coincida
- "OFFTOPIC" si es una pregunta casual no relacionada (como "¿qué día es hoy?", "¿cómo estás?", etc.)
- "CHANGE_TOPIC" si el usuario menciona un producto financiero diferente que NO está en las opciones actuales
- "UNCLEAR" si no estás seguro o ninguna opción coincide claramente"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    try:
        result = LLM.invoke(messages)
        response = result.content.strip()
        
        if response == "OFFTOPIC":
            return {"type": "offtopic"}
        elif response == "CHANGE_TOPIC":
            return {"type": "change_topic"}
        elif response == "UNCLEAR":
            return {"type": "unclear"}
        else:
            # Extraer número
            option_num = int(response) - 1
            if 0 <= option_num < len(children):
                return {"type": "match", "child": children[option_num]}
    except:
        pass
    
    return {"type": "unclear"}


def generate_question(node: Dict, children: List[Dict]) -> str:
    """Genera una pregunta natural basada en el nodo actual y sus hijos"""
    if not children:
        return f"Has llegado al producto: {node['nombre']}. {node['descripcion']}. ¿Necesitas información adicional?"
    
    children_names = ", ".join([child['nombre'] for child in children])
    
    prompt = f"""Genera una pregunta natural y amigable para un chatbot financiero.

Contexto actual: {node['nombre']} - {node['descripcion']}

Opciones disponibles para el usuario: {children_names}

La pregunta debe:
1. Ser breve y directa
2. Invitar al usuario a especificar su interés
3. Mencionar las opciones de manera natural

Responde SOLO con la pregunta, sin explicaciones adicionales."""
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Genera la pregunta")
    ]
    
    result = LLM.invoke(messages)
    return result.content.strip()


def handle_offtopic(user_message: str, current_node: Dict) -> str:
    """Maneja preguntas fuera de contexto"""
    prompt = f"""El usuario está navegando en un chatbot financiero y está en: {current_node['nombre']}.

El usuario ha hecho una pregunta fuera de contexto: "{user_message}"

Responde brevemente a su pregunta y luego redirige amablemente la conversación de vuelta al tema financiero.

Tu respuesta debe:
1. Ser muy breve (máximo 2 líneas)
2. Responder su pregunta básicamente
3. Volver a preguntar sobre el tema financiero actual"""
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_message)
    ]
    
    result = LLM.invoke(messages)
    return result.content.strip()

def process_message(session_id: str, message: str, user_id: str = "anonymous") -> Dict[str, Any]:
    """
    Procesa un mensaje del usuario.
    Integra búsqueda RAG + árbol de decisión + LLM.
    """
    try:
        # 1. Validar o crear sesión
        if not session_id:
            # Crear nueva sesión automáticamente
            new_session = start_conversation(user_id)
            session_id = new_session["session_id"]
            print(f"🆕 Nueva sesión creada: {session_id}")
        
        # 2. Obtener sesión existente
        session = get_session(session_id)
        if not session:
            return {"error": "Sesión no encontrada"}
        
        # 3. **AQUÍ INTEGRAMOS RAG** - Buscar respuesta usando documentos
        # Import RAG answer function using package-relative path; supports both module and direct script executions
        try:
            from .rag_service import answer_question  # when imported as part of 'api.services'
        except ImportError:
            try:
                from api.services.rag_service import answer_question  # explicit package path fallback
            except ImportError:
                from rag_service import answer_question  # last resort if running in a flat script context
        
        print(f"🔍 Buscando respuesta RAG para: '{message}'")
        rag_result = answer_question(
            question=message,
            k=3,  # Top 3 documentos más relevantes
            min_score=0.5  # Score mínimo de similitud
        )
        
        if "error" in rag_result:
            print(f"⚠️ Error en RAG: {rag_result['error']}")
            # Continuar con árbol de decisión tradicional
            bot_response = "No pude procesar tu pregunta correctamente."
            sources = []
        else:
            bot_response = rag_result["answer"]
            sources = rag_result.get("sources", [])
            print(f"✅ Respuesta RAG generada con {len(sources)} fuentes")
        
        # 4. **OPCIONAL**: También puedes usar tu árbol de decisión
        # Si quieres combinar RAG + árbol, descomenta esto:
        """
        current_node_id = session.get("current_node_id", "root")
        tree_response = get_tree_response(current_node_id, message)
        
        # Combinar respuestas o elegir una según lógica
        if tree_response.get("is_final"):
            bot_response = tree_response["response"]
        """
        

        responses_collection = Database.get_collection("responses")
        
        response_doc = {
            "session_id": session_id,
            "user_id": user_id,
            "user_message": message,
            "bot_response": bot_response,
            "sources_used": sources,
            "timestamp": datetime.utcnow(),
            "question_node_id": session.get("current_node_id"),  # Para analytics
        }
        
        responses_collection.insert_one(response_doc)
        
        # 6. Actualizar sesión
        sessions_collection = Database.get_collection("sessions")
        sessions_collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "updated_at": datetime.utcnow(),
                    "last_message": message
                },
                "$push": {
                    "messages": {
                        "role": "user",
                        "content": message,
                        "timestamp": datetime.utcnow()
                    }
                }
            }
        )
        
        # 7. Retornar respuesta completa
        return {
            "session_id": session_id,
            "response": bot_response,
            "sources": sources,  # Fuentes usadas para la respuesta
            "message_id": str(response_doc.get("_id")),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error en process_message: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def start_conversation(user_id: str = "anonymous") -> Dict[str, Any]:
    """Inicia una nueva conversación"""
    session_id = create_session(user_id)
    root_node = FINANCIAL_TREE
    children = get_children(root_node)
    
    question = generate_question(root_node, children)
    
    return {
        "session_id": session_id,
        "response": f"¡Hola! Soy tu asistente financiero. {question}",
        "current_node": root_node,
        "options": [{"id": c["id"], "nombre": c["nombre"]} for c in children]
    }
    
def set_data_capture_step(session_id: str, step: int):
    """Marca la sesión para la captura de datos (1: preguntado, 2: datos enviados)"""
    from bson import ObjectId
    sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {
            "$set": {
                "data_capture_step": step,
                "updated_at": datetime.utcnow()
            }
        }
    )
    
def get_session_summary(session_id: str) -> str:
    """Consulta MongoDB, crea una lista del camino tomado y usa el LLM para resumir."""
    
    # 1. Fase de Recolección de Datos (Igual que antes)
    try:
        session_obj_id = ObjectId(session_id)
    except:
        return "Resumen de sesión no disponible."

    pipeline = [
        {"$match": {"session_id": session_id}},
        {"$sort": {"created_at": 1}}
    ]
    responses = list(responses_collection.aggregate(pipeline))
    
    if not responses:
        return "El usuario no navegó por el árbol."

    # Crear una lista simple del camino para el LLM
    path_list = []
    for i, resp in enumerate(responses):
        node_id = resp["selected_child_id"]
        node = get_node_by_id(node_id)
        if node:
            path_list.append(f"Paso {i+1}: {node['nombre']} ({node['descripcion']}).")

    raw_path_text = "\n".join(path_list)

    # 2. Fase de Generación de Resumen con LLM (¡NUEVO!)
    
    prompt = f"""
    Eres un asistente de ventas financiero. Recibiste el siguiente camino de navegación de un cliente en nuestro chatbot:

    {raw_path_text}

    Tu tarea es generar un resumen narrativo breve (máximo 3-4 líneas) que sirva para que un asesor comercial entienda:
    1. El principal producto de interés del cliente.
    2. El camino general que tomó para llegar a esa decisión.

    Ejemplo de respuesta: "El cliente comenzó explorando Préstamos de Consumo y terminó especificando su interés en un Crédito de Vehículo. Está buscando financiación para la compra de un carro."
    
    Responde ÚNICAMENTE con el resumen narrativo, sin títulos ni explicaciones adicionales.
    """
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Genera el resumen")
    ]
    
    try:
        # Usamos el LLM que ya tienes definido (Gemini-2.5-flash)
        result = LLM.invoke(messages)
        return result.content.strip()
    except Exception as e:
        print(f"Error al generar resumen con LLM: {e}")
        # Retorna el listado simple si el LLM falla
        return "Resumen (Fallo LLM):\n" + raw_path_text

def send_lead_notification(session_id: str, lead_data: Dict[str, str], product_node: Dict):
    """Llama al microservicio de notificaciones con los datos del lead y el resumen."""
    
    summary = get_session_summary(session_id) # Esta función ahora usa el LLM
    
    # ... (rest of the function remains the same) ...
    subject = f"NUEVO LEAD CHATBOT: {product_node['nombre']}"
    
    body = f"""
    ¡Has capturado un nuevo Lead del chatbot financiero!

    **Producto de Interés Final:** {product_node['nombre']}
    **Descripción:** {product_node['descripcion']}

    **--- Datos del Cliente (Lead) ---**
    - **Nombre:** {lead_data['name']}
    - **Teléfono:** {lead_data['phone']}
    - **Correo Electrónico:** {lead_data['email']}
    - **ID de Sesión:** {session_id}

    **--- RESUMEN NARRATIVO DE NAVEGACIÓN (Generado por IA) ---**
    {summary}
    """
    
    # 3. Datos de la solicitud a ms-notifications
    notification_data = {
        # ¡IMPORTANTE! Reemplaza 'correo_interno_recepcion@ejemplo.com' por tu correo real de recepción de leads.
        "to": "juan.reyes54587@ucaldas.edu.co", 
        "subject": subject,
        "body": body,
        "is_html": False
    }

    try:
        response = requests.post(NOTIFICATIONS_SERVICE_URL, json=notification_data, timeout=10)
        response.raise_for_status() # Lanza error para códigos 4xx/5xx
        print(f"Notificación de lead enviada con éxito. Status: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error al enviar la notificación de lead a ms-notifications: {e}")
        return False