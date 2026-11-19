"""Prueba de carga de POST /p1/predecir (viajes por estación-hora del día siguiente) con Locust.

Dos consumidores, como se usaría P1 en operación:
- BatchNocturno: al cierre del día se piden los viajes de mañana de toda la red en una sola
  llamada (472 estaciones x 20 horas de servicio = 9,440 registros, bajo el tope de 10,000 de la
  API). En operación es una llamada al día; aquí se repite para estresar el servidor.
- ConsultaEstacion: alguien que revisa una estación para mañana (1 estación x 20 horas).

El payload sale de references/api_p1_peticion.json: la API valida con Pydantic estricto (horas sin
servicio, fin de semana coherente con el día), así que se reciclan esas estaciones con origen_id
nuevos y las 20 horas de servicio de un mismo día, en vez de inventar datos.

Además de los CSV de Locust, cada petición se guarda en --archivo-peticiones: los percentiles de
stats_history son acumulados desde el inicio, y con este archivo se calculan por escalón.

Humo (3 usuarios, 25 s):
    locust -f locust/locustfile_p1.py --headless --host URL -u 3 -r 1 -t 25s
Prueba completa por escalones (5 -> 80 usuarios, 5 min):
    locust -f locust/locustfile_p1.py,locust/escalones_p1.py --headless --host URL \
        --csv reports/carga/p1/p1 --html reports/carga/p1/p1_reporte.html
"""

import copy
import csv
import json
from pathlib import Path
import time

from locust import HttpUser, between, events, task

EJEMPLO = Path(__file__).resolve().parents[1] / "references" / "api_p1_peticion.json"
RED_COMPLETA = 472
# Horas de servicio: 0 y de 5 a 23 (HORAS_SIN_SERVICIO en config.py). No se importa el paquete
# del proyecto porque la imagen de Locust de Compose no lo trae
HORAS_SERVICIO = [0, *range(5, 24)]
# El día que se predice: jueves 19 de noviembre de 2026, día hábil sin festivo
DIA = {"dia_semana": 3, "es_fin_de_semana": 0, "mes": 11, "es_festivo": 0}


def construir_peticion(n_estaciones: int, base: list[dict]) -> dict:
    """n estaciones x 20 horas: se reciclan las del ejemplo con origen_id distintos."""
    registros = []
    for i in range(n_estaciones):
        for j, hora in enumerate(HORAS_SERVICIO):
            registro = copy.deepcopy(base[(i + j) % len(base)])
            registro.update(DIA, origen_id=i + 1, hora=hora)
            registros.append(registro)
    return {"registros": registros}


_base = json.loads(EJEMPLO.read_text(encoding="utf-8"))["registros"]
# Se serializan una sola vez: el generador de carga no gasta CPU armando JSON en cada petición
CONSULTA = json.dumps(construir_peticion(1, _base))
BATCH = json.dumps(construir_peticion(RED_COMPLETA, _base))
NOMBRE_CONSULTA = "POST /p1/predecir [1 estacion x 20 h]"
NOMBRE_BATCH = f"POST /p1/predecir [{RED_COMPLETA} estaciones x 20 h]"
# Registros (estación-hora) por petición: graficas_p1.py los usa para contar predicciones
REGISTROS = {
    NOMBRE_CONSULTA: len(HORAS_SERVICIO),
    NOMBRE_BATCH: RED_COMPLETA * len(HORAS_SERVICIO),
}


@events.init_command_line_parser.add_listener
def _argumentos(parser) -> None:
    parser.add_argument(
        "--archivo-peticiones",
        default="reports/carga/p1/p1_peticiones.csv",
        help="CSV con una fila por petición (tiempo, nombre, latencia, éxito, usuarios)",
    )


_salida = {}


@events.test_start.add_listener
def _abrir_archivo(environment, **kwargs) -> None:
    ruta = Path(environment.parsed_options.archivo_peticiones)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    _salida["archivo"] = ruta.open("w", newline="", encoding="utf-8")
    _salida["csv"] = csv.writer(_salida["archivo"], lineterminator="\n")
    _salida["csv"].writerow(["timestamp", "nombre", "latencia_ms", "exito", "usuarios"])
    _salida["entorno"] = environment


@events.request.add_listener
def _guardar_peticion(name, response_time, exception, **kwargs) -> None:
    if "csv" in _salida:
        usuarios = _salida["entorno"].runner.user_count
        _salida["csv"].writerow(
            [f"{time.time():.3f}", name, f"{response_time:.1f}", int(exception is None), usuarios]
        )


@events.test_stop.add_listener
def _cerrar_archivo(**kwargs) -> None:
    if "archivo" in _salida:
        _salida.pop("csv")
        _salida.pop("archivo").close()


def validar(r, esperadas: int) -> None:
    """Un 200 con predicciones de menos, o sin la suma por zona, también es un fallo."""
    if r.status_code != 200:
        r.failure(f"HTTP {r.status_code}: {r.text[:200]}")
        return
    cuerpo = r.json()
    if len(cuerpo.get("predicciones", [])) != esperadas:
        r.failure(f"se esperaban {esperadas} predicciones")
    elif not cuerpo.get("por_zona"):
        r.failure("la respuesta no trae la suma por zona")


class ConsultaEstacion(HttpUser):
    weight = 4
    # Pausa de 0.5 a 2 s entre peticiones: simula consumidores reales, no un bucle sin pausa
    wait_time = between(0.5, 2)

    @task
    def una_estacion(self) -> None:
        with self.client.post(
            "/p1/predecir",
            data=CONSULTA,
            headers={"Content-Type": "application/json"},
            name=NOMBRE_CONSULTA,
            catch_response=True,
        ) as r:
            validar(r, REGISTROS[NOMBRE_CONSULTA])


class BatchNocturno(HttpUser):
    weight = 1
    wait_time = between(0.5, 2)

    @task
    def red_completa(self) -> None:
        with self.client.post(
            "/p1/predecir",
            data=BATCH,
            headers={"Content-Type": "application/json"},
            name=NOMBRE_BATCH,
            catch_response=True,
        ) as r:
            validar(r, REGISTROS[NOMBRE_BATCH])
