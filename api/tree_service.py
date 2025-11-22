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

# Configuración

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCvYqtwqUNuvzqGhEdhC4f7DRJ3qBccncQ"

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


def process_message(session_id: Optional[str], user_message: str, user_id: str = "anonymous") -> Dict[str, Any]:
    """Procesa el mensaje del usuario y navega el árbol"""
    # Si no hay session_id, crear una nueva sesión
    if not session_id:
        session_id = create_session(user_id)
        is_new_session = True
    else:
        is_new_session = False
    
    # Obtener sesión
    session = get_session(session_id)
    if not session:
        return {"error": "Sesión no encontrada"}
    
    # Verificar si hay un reinicio pendiente
    if session.get("pending_restart"):
        if check_restart_confirmation(user_message):
            # Usuario confirma reinicio
            update_session_node(session_id, "root")
            root_node = FINANCIAL_TREE
            children = get_children(root_node)
            question = generate_question(root_node, children)
            
            return {
                "session_id": session_id,
                "response": f"¡Perfecto! Empecemos de nuevo. {question}",
                "current_node": root_node,
                "options": [{"id": c["id"], "nombre": c["nombre"]} for c in children],
                "restarted": True
            }
        else:
            # Usuario no quiere reiniciar, continuar con el flujo normal
            from bson import ObjectId
            sessions_collection.update_one(
                {"_id": ObjectId(session_id)},
                {"$unset": {"pending_restart": ""}}
            )
            # Continuar procesando el mensaje actual
    
    current_node_id = session["current_node_id"]
    current_node = get_node_by_id(current_node_id)
    
    if not current_node:
        return {"error": "Nodo actual no encontrado"}
    
    # Si es una nueva sesión, dar bienvenida
    if is_new_session:
        children = get_children(current_node)
        question = generate_question(current_node, children)
        
        # También procesar el mensaje inicial del usuario
        match_result = detect_child_match(user_message, children)
        
        if match_result["type"] == "match":
            # Si el primer mensaje ya tiene una intención clara, avanzar
            selected_child = match_result["child"]
            save_response(session_id, current_node_id, user_message, selected_child["id"])
            update_session_node(session_id, selected_child["id"])
            
            new_children = get_children(selected_child)
            if new_children:
                next_question = generate_question(selected_child, new_children)
                return {
                    "session_id": session_id,
                    "response": f"¡Hola! Entiendo que te interesa {selected_child['nombre']}. {next_question}",
                    "current_node": selected_child,
                    "options": [{"id": c["id"], "nombre": c["nombre"]} for c in new_children],
                    "is_new": True
                }
            else:
                return {
                    "session_id": session_id,
                    "response": f"¡Hola! Te interesa: {selected_child['nombre']}. {selected_child['descripcion']}. ¿Necesitas más información sobre este producto?",
                    "current_node": selected_child,
                    "is_leaf": True,
                    "is_new": True
                }
        else:
            # Primer mensaje no claro, hacer pregunta de bienvenida
            return {
                "session_id": session_id,
                "response": f"¡Hola! Soy tu asistente financiero. {question}",
                "current_node": current_node,
                "options": [{"id": c["id"], "nombre": c["nombre"]} for c in children],
                "is_new": True
            }
    
    # Obtener hijos
    children = get_children(current_node)
    
    # --- Lógica de NODO HOJA (Captura de Leads) ---
    if not children:
        # 1. Si el paso de captura NO está iniciado, preguntar si quiere más info
        if session.get("data_capture_step") is None:
            set_data_capture_step(session_id, 1)
            return {
                "session_id": session_id,
                "response": f"¡Has encontrado el producto! Te interesa: {current_node['nombre']}. {current_node['descripcion']}. ¿Quieres que un asesor se ponga en contacto contigo para darte más detalles?",
                "current_node": current_node,
                "is_leaf": True,
                "options": [
                    {"id": "confirm_data", "nombre": "Sí, quiero contacto"},
                    {"id": "cancel_data", "nombre": "No, gracias"}
                ]
            }
        
        # 2. Si el usuario ya está en el paso 1 (preguntado)
        elif session.get("data_capture_step") == 1:
            message_lower = user_message.lower().strip()
            
            # --- Manejo de la Confirmación/Rechazo ---
            if any(keyword in message_lower for keyword in ["sí", "si", "quiero contacto", "aceptar", "dale", "ok"]):
                # El usuario acepta, pedir los datos formalmente
                set_data_capture_step(session_id, 1) # Lo mantenemos en el paso 1 esperando los datos
                return {
                    "session_id": session_id,
                    "response": "¡Excelente! Por favor, envíame tu **Nombre**, **Teléfono** y **Correo Electrónico** separados por comas. Ejemplo: *Juan Pérez, 3001234567, juan.perez@email.com*",
                    "current_node": current_node
                }
            
            elif any(keyword in message_lower for keyword in ["no", "no gracias", "cancelar", "seguir aquí"]):
                # El usuario rechaza, limpiar el estado y seguir normal
                sessions_collection.update_one(
                    {"_id": ObjectId(session_id)},
                    {"$unset": {"data_capture_step": ""}} # Limpiar el paso
                )
                return {
                    "session_id": session_id,
                    "response": "¿De acuerdo. ¿Necesitas ayuda con algún otro producto o pregunta general?",
                    "current_node": current_node
                }
                
            # --- Manejo de la Recepción de Datos (Si no es una confirmación ni un rechazo) ---
            # Asumimos que si no es un sí/no, el usuario está enviando los datos
            
            # Intentar parsear el mensaje del usuario para obtener Name, Phone, Email
            parts = [p.strip() for p in user_message.split(',') if p.strip()]
            
            if len(parts) >= 3:
                name, phone, email = parts[0], parts[1], parts[2]
                
                # Intentar enviar la notificación
                lead_data = {"name": name, "phone": phone, "email": email}
                success = send_lead_notification(session_id, lead_data, current_node)
                
                # Marcar sesión como completada (o limpiar el step)
                set_data_capture_step(session_id, 2)
                
                if success:
                    return {
                        "session_id": session_id,
                        "response": f"¡Gracias, {name}! Tus datos y el resumen del producto ({current_node['nombre']}) han sido enviados. Un asesor te contactará pronto. ¿En qué más puedo ayudarte?",
                        "current_node": current_node,
                        "is_complete": True
                    }
                else:
                    return {
                        "session_id": session_id,
                        "response": "Hubo un error al enviar tus datos. Por favor, inténtalo más tarde. Disculpa las molestias.",
                        "current_node": current_node
                    }
            else:
                # El usuario aceptó pero no envió los datos correctamente
                return {
                    "session_id": session_id,
                    "response": "No pude extraer los 3 datos requeridos (Nombre, Teléfono, Correo). Por favor, envíalos separados por comas. Ejemplo: *Juan Pérez, 3001234567, juan.perez@email.com*",
                    "current_node": current_node
                }
        
        # 3. Si el usuario ya completó o no quiere enviar más
        else:
            return {
                "session_id": session_id,
                "response": "¿Necesitas ayuda con algún otro producto financiero? Si quieres empezar de nuevo, solo di 'reiniciar'.",
                "current_node": current_node
            }
            
    # Detectar coincidencia
    match_result = detect_child_match(user_message, children)
    
    if match_result["type"] == "change_topic":
        # Usuario quiere cambiar a otro tema financiero
        set_pending_restart(session_id)
        return {
            "session_id": session_id,
            "response": "Veo que te interesa otro producto financiero. ¿Quieres que empecemos de nuevo desde el inicio para explorar otras opciones?",
            "current_node": current_node,
            "pending_restart": True,
            "options": [
                {"id": "confirm", "nombre": "Sí, empezar de nuevo"},
                {"id": "cancel", "nombre": "No, continuar aquí"}
            ]
        }
    
    if match_result["type"] == "offtopic":
        # Pregunta fuera de contexto
        response = handle_offtopic(user_message, current_node)
        return {
            "session_id": session_id,
            "response": response,
            "current_node": current_node,
            "is_offtopic": True
        }
    
    elif match_result["type"] == "unclear":
        # No está claro, volver a preguntar
        question = generate_question(current_node, children)
        return {
            "session_id": session_id,
            "response": f"No estoy seguro de entender. {question}",
            "current_node": current_node,
            "options": [{"id": c["id"], "nombre": c["nombre"]} for c in children]
        }
    
    elif match_result["type"] == "match":
        # Coincidencia encontrada
        selected_child = match_result["child"]
        
        # Guardar respuesta
        save_response(session_id, current_node_id, user_message, selected_child["id"])
        
        # Actualizar sesión
        update_session_node(session_id, selected_child["id"])
        
        # Obtener nuevos hijos
        new_children = get_children(selected_child)
        
        if new_children:
            question = generate_question(selected_child, new_children)
            return {
                "session_id": session_id,
                "response": question,
                "current_node": selected_child,
                "options": [{"id": c["id"], "nombre": c["nombre"]} for c in new_children]
            }
        else:
            # Nodo hoja alcanzado
            return {
                "session_id": session_id,
                "response": f"Perfecto! Te interesa: {selected_child['nombre']}. {selected_child['descripcion']}. ¿Necesitas más información sobre este producto?",
                "current_node": selected_child,
                "is_leaf": True
            }
    
    return {"error": "Error al procesar mensaje"}


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