from pymongo import MongoClient
import os

# Configuración por defecto (puedes usar variables de entorno)
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
DB_NAME = os.getenv('DB_NAME', 'chatbot_financiero') # ¡CAMBIA ESTO POR EL NOMBRE REAL DE TU BD!

class Database:
    client: MongoClient = None
    db = None

    @staticmethod
    def initialize():
        try:
            Database.client = MongoClient(MONGO_URI)
            Database.db = Database.client[DB_NAME]
            print(f"✅ Conectado exitosamente a MongoDB: {DB_NAME}")
        except Exception as e:
            print(f"❌ Error conectando a MongoDB: {e}")

    @staticmethod
    def get_collection(collection_name):
        if Database.db is None:
            Database.initialize()
        return Database.db[collection_name]

# Inicializar al importar (opcional, o puedes llamar a initialize() en app.py)
Database.initialize()