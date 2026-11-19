"""Perfil de carga escalonado para locustfile_p1.py (se pasa como segundo archivo con -f).

Mismos escalones que P3 (5 a 80 usuarios, 60 s cada uno) para comparar las preguntas en el
reporte. Sin este archivo, -u/-r/-t mandan (útil para humo); Locust los ignora en cuanto
encuentra una LoadTestShape.
"""

from locust import LoadTestShape


class EscalonesP1(LoadTestShape):
    """(segundo en que termina el escalón, usuarios, usuarios que entran por segundo)."""

    escalones = [(60, 5, 5), (120, 10, 5), (180, 20, 5), (240, 40, 10), (300, 80, 20)]

    def tick(self):
        t = self.get_run_time()
        for fin, usuarios, tasa in self.escalones:
            if t < fin:
                return usuarios, tasa
        return None  # termina la prueba
