# example_save.py
import os
from dotenv import load_dotenv
from app.services.firestore_service import FirestoreService

# Cargar variables de entorno
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Inicializar FirestoreService
fs = FirestoreService(PROJECT_ID)

# Crear un usuario
fs.create_user("user_1", {
    "nombre": "Rafael Medina",
    "cedula": "12345678",
    "telefono": "3001234567",
    "correo": "rafael@example.com",
    "barrio": "El Poblado"
})

# Agregar un vehículo al usuario
fs.add_vehicle("user_1", placa="ABC123", nombre="Toyota Corolla")

# Crear una entidad
fs.create_entity("entidad_1", {
    "nombre": "Alcaldía de Medellín",
    "telefono": "6041234567",
    "correo": "contacto@medellin.gov.co",
    "horario_atencion": "8:00-17:00"
})

# Crear un trámite (el radicado se genera automáticamente)
fs.create_tramite("tramite_1", {
    "nombre": "Licencia de Funcionamiento",
    "contenido": "Requiere RUT, cédula, certificado de uso de suelo",
    "vencimiento": "2025-12-31",
    "user_id": "user_1",
    "entity_id": "entidad_1"
})

# Leer el trámite creado
tramite = fs.get_tramite("tramite_1")
print("Trámite creado con radicado:", tramite["numero_radicado"])

# Listar vehículos del usuario
vehicles = fs.list_user_vehicles("user_1")
print("Vehículos del usuario:", vehicles)
