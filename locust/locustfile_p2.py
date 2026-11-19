"""Prueba de carga de POST /p2/predecir (riesgo de estación) con Locust.

Dos consumidores, como se usaría P2 en operación:
- UsuarioConsulta: alguien que revisa una estación antes de ir por una bici (1 registro).
- BarridoBin: el proceso que al cierre de cada bin de 30 min califica la red completa (472).
  En operación es una llamada cada 30 min; aquí se comprime para estresar el servidor.

Humo:     locust -f locust/locustfile_p2.py --headless --host URL -u 3 -r 1 -t 25s
Completa: locust -f locust/locustfile_p2.py,locust/escalones_p2.py --master --headless --expect-workers N_WORKERS \
              --host URL --csv reports/carga/p2/p2 --csv reports/carga/p2/p2 --csv-full-history \
              --html reports/carga/p2/p2_reporte.html
"""

import copy
import json
from pathlib import Path

from locust import HttpUser, between, task

EJEMPLO = Path(__file__).resolve().parents[1] / "references" / "api_p2_peticion.json"
RED_COMPLETA = 472  # estaciones de la red en un bin (ver P2_MAX_REGISTROS en api/p2.py)


def construir_peticion(n: int, base: list[dict]) -> dict:
    """n registros válidos: se reciclan los del ejemplo cambiando solo el id de estación."""
    registros = []
    for i in range(n):
        registro = copy.deepcopy(base[i % len(base)])
        registro["estacion"] = i + 1
        registros.append(registro)
    return {"registros": registros}


_base = json.loads(EJEMPLO.read_text(encoding="utf-8"))["registros"]
UNA = construir_peticion(1, _base)
RED = construir_peticion(RED_COMPLETA, _base)
RED_JSON = json.dumps(RED)


def validar(r, esperadas: int) -> None:
    """Un 200 con predicciones de menos, o con conteo por clase incoherente, también es fallo."""
    if r.status_code != 200:
        r.failure(f"HTTP {r.status_code}: {r.text[:200]}")
        return
    cuerpo = r.json()
    if len(cuerpo.get("predicciones", [])) != esperadas:
        r.failure(f"se esperaban {esperadas} predicciones")
    elif sum(cuerpo.get("por_clase", {}).values()) != esperadas:
        r.failure("por_clase no suma el número de registros")


class UsuarioConsulta(HttpUser):
    weight = 4
    wait_time = between(0.5, 2)

    @task
    def una_estacion(self) -> None:
        with self.client.post(
            "/p2/predecir", json=UNA, name="POST /p2/predecir [1 estacion]", catch_response=True
        ) as r:
            validar(r, 1)


class BarridoBin(HttpUser):
    weight = 1
    wait_time = between(0.5, 2)

    @task
    def red_completa(self) -> None:
        with self.client.post(
            "/p2/predecir",
            data=RED_JSON,
            headers={"Content-Type": "application/json"},
            name=f"POST /p2/predecir [{RED_COMPLETA} estaciones]",
            catch_response=True,
        ) as r:
            validar(r, RED_COMPLETA)