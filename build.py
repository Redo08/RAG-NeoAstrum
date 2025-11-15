import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Configura tu API key (obtén una gratis en: https://aistudio.google.com/app/apikey)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDQxZCOVDtlt0srL1xyLg4ToFficIlJnhU"

# Crea el modelo
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Funciones de herramientas
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def get_user_location(user_id: str) -> str:
    """Get user location based on user ID."""
    return "Florida" if user_id == "1" else "SF"

# Sistema de instrucciones
SYSTEM_PROMPT = """You are an expert weather forecaster who speaks in puns.

When a user asks for weather:
1. First determine the location
2. If they don't specify, assume they mean their current location (use the user_location provided)
3. Get the weather using the available weather data
4. Return a punny weather forecast

Remember to be creative with your weather puns!"""

# Historial de conversación
conversation_history = []

# Función para chatear
def chat(user_message: str, user_id: str = "1"):
    global conversation_history
    
    # Obtener datos del usuario
    user_location = get_user_location(user_id)
    
    # Analizar si el usuario pregunta por el clima
    if any(word in user_message.lower() for word in ["weather", "climate", "forecast", "temperatura", "clima"]):
        # Determinar la ciudad
        if "san francisco" in user_message.lower() or "sf" in user_message.lower():
            city = "San Francisco"
        elif any(word in user_message.lower() for word in ["outside", "here", "my location", "where i am"]):
            city = user_location
        else:
            # Usar la ubicación del usuario por defecto
            city = user_location
        
        # Obtener el clima
        weather_info = get_weather_for_location(city)
        context = f"\nUser's location: {user_location}\nWeather data: {weather_info}"
    else:
        context = f"\nUser's location: {user_location}"
    
    # Crear los mensajes
    messages = [SystemMessage(content=SYSTEM_PROMPT + context)]
    
    # Agregar historial previo
    messages.extend(conversation_history)
    
    # Agregar mensaje actual
    messages.append(HumanMessage(content=user_message))
    
    # Obtener respuesta
    response = model.invoke(messages)
    
    # Guardar en historial
    conversation_history.append(HumanMessage(content=user_message))
    conversation_history.append(AIMessage(content=response.content))

    print("---- DEBUG INFO ----")
    print("Messages sent to LLM:")
    for msg in messages:
        print(f"{msg.type}: {msg.content}")
    
    print("---- DEBUG INFO ----")
    print("Response from LLM:")
    print(response.content)
    
    return response.content

# Función para reiniciar conversación
def reset_conversation():
    global conversation_history
    conversation_history = []

# Pruebas
if __name__ == "__main__":
    print("=" * 70)
    print("🌤️  WEATHER FORECAST BOT (Punny Edition)")
    print("=" * 70)
    
    # Conversación 1
    print("\n📍 Question 1: What's the weather outside?")
    response1 = chat("what is the weather outside?", user_id="1")
    print(f"\n🤖 Response:\n{response1}")
    
    print("\n" + "=" * 70)
    
    # Conversación 2 (mantiene contexto)
    print("\n📍 Question 2: Thank you!")
    response2 = chat("thank you!", user_id="1")
    print(f"\n🤖 Response:\n{response2}")
    
    print("\n" + "=" * 70)
    
    # Reiniciar conversación
    reset_conversation()
    
    # Nueva conversación
    print("\n📍 Question 3: What about in San Francisco?")
    response3 = chat("what about in San Francisco?", user_id="1")
    print(f"\n🤖 Response:\n{response3}")
    
    print("\n" + "=" * 70)
    print("\n✅ Done! The bot maintains conversation context automatically.")
    print("💡 Tip: Get your free API key at https://aistudio.google.com/app/apikey")