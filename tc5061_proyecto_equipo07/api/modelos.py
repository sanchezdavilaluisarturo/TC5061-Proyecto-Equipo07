"""Carga de los modelos de P1, P2 y P3 desde el Model Registry de MLflow para la API."""

from dataclasses import dataclass, field
import os
from typing import Any

from fastapi import HTTPException, Request
from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from tc5061_proyecto_equipo07 import config as cfg

MODELOS_REGISTRADOS = {
    "p1": cfg.P1_MODELO_REGISTRADO,
    "p2": cfg.P2_MODELO_REGISTRADO,
    "p3": cfg.P3_MODELO_REGISTRADO,
}


@dataclass
class ModeloServido:
    """Un modelo del Registry tal como lo sirve la API: versión resuelta, etiquetas y estado."""

    pregunta: str
    nombre: str
    alias: str
    version: int | None = None
    # Etiquetas de la versión (source_run_id, commit, data_version...), para leer el linaje
    etiquetas: dict[str, str] = field(default_factory=dict)
    # Parámetros del run de origen (umbrales de decisión de P2, hiperparámetros...)
    parametros: dict[str, str] = field(default_factory=dict)
    modelo: Any = None
    error: str | None = None

    @property
    def listo(self) -> bool:
        return self.modelo is not None


def alias_configurado() -> str:
    """Alias que se sirve; se cambia sin reconstruir la imagen (por ejemplo, a challenger)."""
    return os.getenv("MODELO_ALIAS", "champion")


def cargar_parametros(pregunta: str, id_run: str | None) -> dict[str, str]:
    """Parámetros del run de origen (por ejemplo, los umbrales de P2); vacío si no se pueden leer."""
    if not id_run:
        return {}
    try:
        return MlflowClient().get_run(id_run).data.params
    except MlflowException as error:
        # No impide servir el modelo
        logger.warning(f"{pregunta}: no se pudieron leer los parámetros del run {id_run}: {error}")
        return {}


def cargar_modelo(pregunta: str, nombre: str, alias: str) -> ModeloServido:
    """Resuelve el alias a una versión y carga esa versión; si falla, guarda el error y sigue."""
    servido = ModeloServido(pregunta=pregunta, nombre=nombre, alias=alias)
    try:
        version = MlflowClient().get_model_version_by_alias(nombre, alias)
        # Se carga por número de versión: si el alias se mueve mientras arranca la API, lo que se
        # reporta en /salud sigue siendo exactamente lo que se cargó
        servido.modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{version.version}")
    except (MlflowException, OSError, ImportError) as error:
        # Recuperable: la API levanta con los demás modelos y este endpoint responde 503
        logger.error(f"{pregunta}: no se pudo cargar models:/{nombre}@{alias}: {error}")
        servido.error = str(error)
        return servido
    servido.version = int(version.version)
    servido.etiquetas = dict(version.tags)
    servido.parametros = cargar_parametros(pregunta, servido.etiquetas.get("source_run_id"))
    logger.success(f"{pregunta}: models:/{nombre}@{alias} -> versión {version.version}")
    return servido


def cargar_modelos() -> dict[str, ModeloServido]:
    """Carga los modelos de las tres preguntas con el alias configurado."""
    if not cfg.MLFLOW_TRACKING_URI:
        raise ValueError("MLFLOW_TRACKING_URI no está definida: copiar .env.example a .env")
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    alias = alias_configurado()
    return {
        pregunta: cargar_modelo(pregunta, nombre, alias)
        for pregunta, nombre in MODELOS_REGISTRADOS.items()
    }


def obtener_modelo(request: Request, pregunta: str) -> ModeloServido:
    """Modelo listo de la pregunta, o 503 si no cargó (para usar dentro de cada endpoint)."""
    servido = request.app.state.modelos[pregunta]
    if not servido.listo:
        raise HTTPException(
            status_code=503,
            detail=f"El modelo de {pregunta} no está disponible: {servido.error}",
        )
    return servido
