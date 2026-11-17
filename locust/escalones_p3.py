"""Perfil de carga escalonado para locustfile_p3.py (se pasa como segundo archivo con -f).

Solo se carga cuando se quiere la prueba completa; sin este archivo, -u/-r/-t mandan (útil para
humo). Locust ignora -u y -t en cuanto encuentra una LoadTestShape.
"""

from locust import LoadTestShape


class EscalonesP3(LoadTestShape):
    """Escalones de usuarios: se ve en qué escalón empieza a degradarse la latencia.

    (segundo en que termina el escalón, usuarios, usuarios que entran por segundo)
    """

    escalones = [(60, 5, 5), (120, 10, 5), (180, 20, 5), (240, 40, 10), (300, 80, 20)]

    def tick(self):
        t = self.get_run_time()
        for fin, usuarios, tasa in self.escalones:
            if t < fin:
                return usuarios, tasa
        return None  # termina la prueba
