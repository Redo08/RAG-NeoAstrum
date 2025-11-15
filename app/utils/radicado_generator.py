import random
import string
from datetime import datetime

class RadicadoGenerator:
    """
    Generador de números de radicado alfanuméricos.
    Formato sugerido: RAD-YYYYMMDD-XXXX
      - RAD: prefijo fijo
      - YYYYMMDD: fecha
      - XXXX: 4 caracteres aleatorios (letras + números)
    """

    def __init__(self, prefix: str = "RAD"):
        self.prefix = prefix

    def generate(self) -> str:
        """
        Genera un número de radicado único
        """
        date_str = datetime.now().strftime("%Y%m%d")
        random_str = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
        radicado = f"{self.prefix}-{date_str}-{random_str}"
        return radicado