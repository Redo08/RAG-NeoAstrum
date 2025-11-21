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

# Configuración
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDQxZCOVDtlt0srL1xyLg4ToFficIlJnhU"

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
FINANCIAL_TREE = {
    "id": "root",
    "nombre": "Finanzas en Colombia",
    "descripcion": "Árbol de productos financieros para chatbot",
    "categorias": [
        {
            "id": "prestamo",
            "nombre": "Préstamo",
            "descripcion": "Productos de financiación",
            "subcategorias": [
                {
                    "id": "prestamo_consumo",
                    "nombre": "Préstamo de Consumo",
                    "descripcion": "Créditos para necesidades personales",
                    "tipos": [
                        {
                            "id": "consumo_libre_inversion",
                            "nombre": "Préstamo de Libre Inversión",
                            "descripcion": "Préstamo sin destino específico",
                            "preguntas_relacionadas": [
                                "préstamo de libre inversión",
                                "crédito personal",
                                "crédito para cualquier cosa"
                            ]
                        },
                        {
                            "id": "consumo_vehiculo",
                            "nombre": "Crédito de Vehículo",
                            "descripcion": "Crédito para compra de carro",
                            "preguntas_relacionadas": [
                                "crédito para carro",
                                "financiar vehículo",
                                "préstamo para comprar carro"
                            ]
                        }
                    ]
                },
                {
                    "id": "prestamo_hipotecario",
                    "nombre": "Préstamo Hipotecario",
                    "descripcion": "Créditos para vivienda",
                    "tipos": [
                        {
                            "id": "hipotecario_vis_no_vis",
                            "nombre": "Hipoteca VIS / No VIS",
                            "descripcion": "Créditos de vivienda VIS y No VIS",
                            "preguntas_relacionadas": [
                                "crédito de vivienda",
                                "hipoteca para apartamento",
                                "préstamo para comprar casa"
                            ]
                        },
                        {
                            "id": "hipotecario_leasing_habitacional",
                            "nombre": "Leasing Habitacional",
                            "descripcion": "Arriendo financiero de vivienda con opción de compra",
                            "preguntas_relacionadas": [
                                "qué es leasing habitacional",
                                "leasing para comprar vivienda",
                                "diferencia leasing e hipoteca"
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "id": "inversion",
            "nombre": "Inversión",
            "descripcion": "Productos para generar rentabilidad",
            "subcategorias": [
                {
                    "id": "inversion_cdt",
                    "nombre": "CDT",
                    "descripcion": "Certificados de Depósito a Término",
                    "tipos": [
                        {
                            "id": "cdt_tasa_fija",
                            "nombre": "CDT a Tasa Fija",
                            "descripcion": "Tasa fija durante todo el período",
                            "preguntas_relacionadas": [
                                "cdt tasa fija",
                                "rendimiento cdt fijo",
                                "invertir en cdt fijo"
                            ]
                        },
                        {
                            "id": "cdt_tasa_variable",
                            "nombre": "CDT a Tasa Variable",
                            "descripcion": "Tasa indexada a DTF o IBR",
                            "preguntas_relacionadas": [
                                "cdt tasa variable",
                                "cdt ligado a DTF",
                                "cdt indexado a IBR"
                            ]
                        }
                    ]
                },
                {
                    "id": "inversion_acciones",
                    "nombre": "Acciones",
                    "descripcion": "Participación en empresas listadas",
                    "tipos": [
                        {
                            "id": "acciones_ordinarias",
                            "nombre": "Acciones Ordinarias",
                            "descripcion": "Acciones con derecho a voto",
                            "preguntas_relacionadas": [
                                "acciones ordinarias",
                                "comprar acciones comunes",
                                "invertir en bolsa"
                            ]
                        },
                        {
                            "id": "acciones_preferenciales",
                            "nombre": "Acciones Preferenciales",
                            "descripcion": "Acciones sin voto pero con dividendos prioritarios",
                            "preguntas_relacionadas": [
                                "acciones preferenciales",
                                "acciones que pagan más dividendos",
                                "preferenciales vs ordinarias"
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "id": "renta",
            "nombre": "Renta",
            "descripcion": "Productos de alquiler o arriendo",
            "subcategorias": [
                {
                    "id": "renta_arriendo",
                    "nombre": "Arriendo",
                    "descripcion": "Alquiler de bienes inmuebles",
                    "tipos": [
                        {
                            "id": "arriendo_residencial",
                            "nombre": "Arriendo de Vivienda",
                            "descripcion": "Alquiler de casas y apartamentos",
                            "preguntas_relacionadas": [
                                "arriendo de vivienda",
                                "rentar apartamento",
                                "buscar casa en arriendo"
                            ]
                        },
                        {
                            "id": "arriendo_comercial",
                            "nombre": "Arriendo de Local Comercial",
                            "descripcion": "Local para negocios",
                            "preguntas_relacionadas": [
                                "arriendo de local",
                                "local comercial",
                                "cuánto vale arrendar un local"
                            ]
                        }
                    ]
                },
                {
                    "id": "renta_leasing",
                    "nombre": "Leasing",
                    "descripcion": "Arriendo financiero",
                    "tipos": [
                        {
                            "id": "leasing_vehicular",
                            "nombre": "Leasing Vehicular",
                            "descripcion": "Arrendamiento financiero para carro",
                            "preguntas_relacionadas": [
                                "leasing para carro",
                                "arrendar carro con opción de compra",
                                "leasing vehicular"
                            ]
                        },
                        {
                            "id": "leasing_operativo",
                            "nombre": "Leasing Operativo",
                            "descripcion": "Arrendamiento sin opción de compra",
                            "preguntas_relacionadas": [
                                "leasing operativo",
                                "arrendar maquinaria",
                                "leasing sin opción de compra"
                            ]
                        }
                    ]
                }
            ]
        }
    ]
}


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
    
    # Si no hay hijos, es un nodo hoja
    if not children:
        return {
            "session_id": session_id,
            "response": f"Has llegado al producto final: {current_node['nombre']}. {current_node['descripcion']}. ¿En qué más puedo ayudarte?",
            "current_node": current_node,
            "is_leaf": True
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