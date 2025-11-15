import os
from google.cloud import firestore

# Ruta a tu archivo JSON de la cuenta de servicio
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"deploy\service_account.json"

# ID de tu proyecto de Google Cloud
PROJECT_ID = "enigmacoders"
# ID de tu base de datos de Firestore (Según tu imagen)
DATABASE_ID = "enigmacodersdb" 

# Inicializa el cliente de Firestore, especificando el database_id
# Si la base de datos que creaste en la consola se llama 'default', 
# entonces no necesitarías este parámetro, pero como creaste una 
# con un nombre explícito 'enigmacodersdb', es mejor especificarlo.
# ¡IMPORTANTE! Si usaste el ID `enigmacodersdb` en la consola, úsalo aquí. 
# Si usaste la base de datos que *ya existía* y la renombraste/usaste, 
# verifica su ID real. A menudo el ID sigue siendo `(default)`.

# Prueba primero con database="enigmacodersdb":
client = firestore.Client(project=PROJECT_ID, database=DATABASE_ID) 

# Si lo anterior falla y la base de datos que creaste es la principal:
# Prueba con database="(default)" si la API de Firestore no puede encontrar 'enigmacodersdb'
# client = firestore.Client(project=PROJECT_ID, database="(default)")


# Crea (o accede) a la colección "test" y al documento "doc1"
doc_ref = client.collection("test").document("doc1")

# Guarda datos en el documento
doc_ref.set({"hola": "mundo"})

print(f"¡Funciona! Documento creado en Firestore correctamente en la base de datos: {DATABASE_ID}.")