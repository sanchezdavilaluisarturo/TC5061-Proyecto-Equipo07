"""Compara tres formas de segmentar la red completa (470 estaciones) con /p3/segmentar.

La unidad de trabajo es un "ciclo": dejar segmentadas las 470 estaciones. Cada clase la resuelve
distinto y registra el tiempo del ciclo completo como la petición "CICLO red completa [modo]":

    UsuarioIndividual  470 llamadas de 1 estación (cada consumidor pide la suya)
    UsuarioLotes       10 llamadas de hasta 50 estaciones
    UsuarioRedCompleta 1 llamada de 470 estaciones

Todos usan el mismo número de usuarios, así que se comparan las estaciones por segundo y el
tiempo por ciclo. Se corre una clase a la vez (ver README):
    locust -f locust/locustfile_p3_agrupado.py UsuarioLotes --headless --host URL -u 5 -r 1 -t 180s
"""

import sys
import time
from pathlib import Path

from locust import HttpUser, between, events, task

sys.path.insert(0, str(Path(__file__).resolve().parent))
from locustfile_p3 import _base, construir_peticion  # noqa: E402  (solo funciones, no la clase)

RED = construir_peticion(470, _base)["estaciones"]
LOTE = 50


def en_lotes(estaciones: list[dict], n: int) -> list[dict]:
    return [{"estaciones": estaciones[i : i + n]} for i in range(0, len(estaciones), n)]


# Se arman una sola vez: el generador no gasta CPU en JSON durante la prueba
INDIVIDUAL = en_lotes(RED, 1)
LOTES = en_lotes(RED, LOTE)
COMPLETA = en_lotes(RED, len(RED))


class _Base(HttpUser):
    abstract = True
    wait_time = between(0.5, 1)  # pausa entre ciclos, no entre las llamadas de un ciclo
    modo = ""
    peticiones: list[dict] = []

    def on_start(self) -> None:
        self.nombre = f"POST /p3/segmentar [{self.modo}]"

    def _ciclo(self) -> None:
        inicio = time.perf_counter()
        error = None
        for peticion in self.peticiones:
            with self.client.post(
                "/p3/segmentar", json=peticion, name=self.nombre, catch_response=True
            ) as r:
                if r.status_code != 200:
                    r.failure(f"HTTP {r.status_code}")
                    error = Exception(f"HTTP {r.status_code}")
                    break
        # Una fila aparte con el tiempo del ciclo completo (las 470 estaciones)
        events.request.fire(
            request_type="CICLO",
            name=f"red completa [{self.modo}]",
            response_time=(time.perf_counter() - inicio) * 1000,
            response_length=0,
            exception=error,
            context={},
        )


class UsuarioIndividual(_Base):
    modo, peticiones = "470 x 1 estacion", INDIVIDUAL

    @task
    def ciclo(self) -> None:
        self._ciclo()


class UsuarioLotes(_Base):
    modo, peticiones = f"10 x {LOTE} estaciones", LOTES

    @task
    def ciclo(self) -> None:
        self._ciclo()


class UsuarioRedCompleta(_Base):
    modo, peticiones = "1 x 470 estaciones", COMPLETA

    @task
    def ciclo(self) -> None:
        self._ciclo()
