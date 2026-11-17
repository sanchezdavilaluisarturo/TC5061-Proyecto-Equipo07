"""Prueba de carga de POST /p3/segmentar (segmentos de estaciones) con Locust.

Qué mide: latencia (p50/p95/p99), peticiones por segundo y tasa de error del endpoint de P3
mientras sube el número de usuarios concurrentes.

Por qué tres tamaños de petición: el costo de /p3/segmentar crece con el número de estaciones
(se arma un DataFrame de 44 columnas y se reasignan grupos), así que se prueban el caso mínimo
(1 estación), el ejemplo del repo (3) y la red completa en una sola llamada (470, el máximo
realista del batch nocturno). El payload sale de references/api_p3_peticion.json: la API valida
con Pydantic estricto (extra="forbid", perfiles que suman 1, origen_id sin repetir), así que se
clona ese ejemplo con ids nuevos en vez de inventar datos.

Uso (con la API levantada, ver README):
    locust -f locust/locustfile_p3.py --headless --host http://127.0.0.1:8000 \
        --csv reports/carga/p3 --html reports/carga/p3_reporte.html
"""

import copy
import json
from pathlib import Path

from locust import HttpUser, LoadTestShape, between, task

EJEMPLO = Path(__file__).resolve().parents[1] / "references" / "api_p3_peticion.json"
RED_COMPLETA = 470  # ~472 estaciones en la red (ver P3_MAX_ESTACIONES en api/p3.py)
MEDIANO = 50


def construir_peticion(n: int, base: list[dict]) -> dict:
    """n estaciones válidas: se reciclan las del ejemplo con origen_id consecutivos distintos."""
    estaciones = []
    for i in range(n):
        estacion = copy.deepcopy(base[i % len(base)])
        estacion["origen_id"] = i + 1
        estaciones.append(estacion)
    return {"estaciones": estaciones}


_base = json.loads(EJEMPLO.read_text())["estaciones"]
# Se serializan una sola vez: así el generador de carga no gasta CPU armando JSON por petición
PETICIONES = {
    "1 estacion": construir_peticion(1, _base),
    f"{MEDIANO} estaciones": construir_peticion(MEDIANO, _base),
    f"{RED_COMPLETA} estaciones (red completa)": construir_peticion(RED_COMPLETA, _base),
}
PESOS = {"1 estacion": 6, f"{MEDIANO} estaciones": 3, f"{RED_COMPLETA} estaciones (red completa)": 1}


class UsuarioP3(HttpUser):
    # Pausa de 0.5 a 2 s entre peticiones: simula consumidores reales, no un bucle sin pausa
    wait_time = between(0.5, 2)

    def _segmentar(self, nombre: str) -> None:
        peticion = PETICIONES[nombre]
        esperadas = len(peticion["estaciones"])
        # El nombre agrupa las estadísticas por tamaño de petición, no por URL
        with self.client.post(
            "/p3/segmentar", json=peticion, name=f"POST /p3/segmentar [{nombre}]", catch_response=True
        ) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}: {r.text[:200]}")
                return
            predicciones = r.json().get("predicciones", [])
            # 200 con menos predicciones de las enviadas también es un fallo funcional
            if len(predicciones) != esperadas:
                r.failure(f"se esperaban {esperadas} predicciones y llegaron {len(predicciones)}")

    @task(PESOS["1 estacion"])
    def una_estacion(self) -> None:
        self._segmentar("1 estacion")

    @task(PESOS[f"{MEDIANO} estaciones"])
    def cincuenta_estaciones(self) -> None:
        self._segmentar(f"{MEDIANO} estaciones")

    @task(PESOS[f"{RED_COMPLETA} estaciones (red completa)"])
    def red_completa(self) -> None:
        self._segmentar(f"{RED_COMPLETA} estaciones (red completa)")


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
