from google.cloud import firestore
from typing import Optional, Dict, Any, List
from app.utils.radicado_generator import RadicadoGenerator

class FirestoreService:
    """
    Servicio de acceso a Firestore para usuarios, vehículos, trámites y entidades.
    """

    def __init__(self, project_id: str):
        self.client = firestore.Client(project=project_id)
        self.radicado_generator = RadicadoGenerator()

    # ---------------------------
    # USUARIOS
    # ---------------------------
    def create_user(self, user_id: str, data: Dict[str, Any]) -> None:
        """
        Crea o actualiza un usuario
        """
        doc_ref = self.client.collection("users").document(user_id)
        doc_ref.set(data, merge=True)

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        doc_ref = self.client.collection("users").document(user_id)
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else None

    # ---------------------------
    # VEHICULOS
    # ---------------------------
    def add_vehicle(self, user_id: str, placa: str, nombre: str) -> str:
        """
        Agrega un vehículo a un usuario
        """
        vehicles_ref = self.client.collection("users").document(user_id).collection("vehicles")
        doc_ref = vehicles_ref.document(placa)
        doc_ref.set({"placa": placa, "nombre": nombre})
        return doc_ref.id

    def list_user_vehicles(self, user_id: str) -> List[Dict[str, Any]]:
        vehicles_ref = self.client.collection("users").document(user_id).collection("vehicles")
        docs = vehicles_ref.stream()
        return [doc.to_dict() for doc in docs]

    # ---------------------------
    # ENTIDADES
    # ---------------------------
    def create_entity(self, entity_id: str, data: Dict[str, Any]) -> None:
        """
        Crea o actualiza una entidad
        """
        doc_ref = self.client.collection("entities").document(entity_id)
        doc_ref.set(data, merge=True)

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        doc_ref = self.client.collection("entities").document(entity_id)
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else None

    # ---------------------------
    # TRÁMITES
    # ---------------------------
    def create_tramite(self, tramite_id: str, data: Dict[str, Any]) -> None:
        """
        Crea o actualiza un trámite
        Se genera automáticamente 'numero_radicado' si no viene en data.
        data debe incluir:
          - nombre
          - contenido
          - vencimiento
          - user_id
          - entity_id
        """
        # Generar radicado automáticamente si no viene
        if "numero_radicado" not in data or not data["numero_radicado"]:
            data["numero_radicado"] = self.radicado_generator.generate()

        doc_ref = self.client.collection("tramites").document(tramite_id)
        doc_ref.set(data, merge=True)

    def get_tramite(self, tramite_id: str) -> Optional[Dict[str, Any]]:
        doc_ref = self.client.collection("tramites").document(tramite_id)
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else None

    def list_tramites_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        query_ref = self.client.collection("tramites").where("user_id", "==", user_id)
        docs = query_ref.stream()
        return [doc.to_dict() for doc in docs]

    def list_tramites_by_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        query_ref = self.client.collection("tramites").where("entity_id", "==", entity_id)
        docs = query_ref.stream()
        return [doc.to_dict() for doc in docs]
