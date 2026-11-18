"""Siembra el MLflow local con los modelos del servidor del equipo (una corrida y termina).

Uso (lo ejecuta el servicio `sembrador` de docker-compose):
    python -m tc5061_proyecto_equipo07.sembrador

Por cada modelo de P1, P2 y P3 copia la versión que tiene el alias (champion por defecto) del
servidor de origen al MLflow local: baja los artefactos, crea un run local con los mismos
parámetros (la API lee de ahí los umbrales de P2), registra el modelo con el mismo nombre y deja
el mismo alias. Las etiquetas originales se conservan y se agrega `origen_version` y `origen_uri`
para no perder la trazabilidad. Es idempotente: si el alias local ya apunta a una copia de esa
versión de origen, no la vuelve a copiar. Si el origen no responde pero el MLflow local ya tiene
el modelo, conserva esa copia y no falla: la siembra solo necesita red la primera vez.

Variables de entorno:
    MLFLOW_ORIGEN_URI   servidor del equipo (por defecto https://mlflow.equipo7mlops.com)
    MLFLOW_TRACKING_URI MLflow local, destino de la copia
    MODELO_ALIAS        alias a copiar (por defecto champion)
"""

import os
from pathlib import Path
import sys
import tempfile

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from tc5061_proyecto_equipo07.api.modelos import MODELOS_REGISTRADOS, alias_configurado

ORIGEN_POR_DEFECTO = "https://mlflow.equipo7mlops.com"
EXPERIMENTO_SEMBRADO = "sembrado"


def _version_con_alias(cliente: MlflowClient, nombre: str, alias: str):
    """Versión del Registry que tiene el alias, o None si no existe."""
    try:
        return cliente.get_model_version_by_alias(nombre, alias)
    except MlflowException:
        return None


def sembrar_modelo(origen_uri: str, destino_uri: str, nombre: str, alias: str) -> str:
    """Copia `nombre@alias` de origen a destino. Regresa 'copiado', 'omitido' o 'conservado'.

    Si la copia falla (por ejemplo, el servidor del equipo no responde) pero el MLflow local ya
    tiene una versión con ese alias, la conserva y regresa 'conservado': la API puede arrancar
    con lo ya sembrado. Si no hay nada local que servir, propaga el error.
    """
    destino = MlflowClient(tracking_uri=destino_uri, registry_uri=destino_uri)
    existente = _version_con_alias(destino, nombre, alias)
    try:
        return _copiar_modelo(origen_uri, destino_uri, nombre, alias, existente)
    except Exception as error:  # cualquier falla del origen se trata igual
        if existente is None:
            raise
        logger.warning(
            f"{nombre}@{alias}: no se pudo copiar del origen ({error}); se conserva la copia "
            f"local v{existente.version} (origen v{existente.tags.get('origen_version', '?')})"
        )
        return "conservado"


def _copiar_modelo(origen_uri: str, destino_uri: str, nombre: str, alias: str, existente) -> str:
    """Copia la versión de origen salvo que el alias local ya apunte a una copia de ella."""
    origen = MlflowClient(tracking_uri=origen_uri, registry_uri=origen_uri)
    destino = MlflowClient(tracking_uri=destino_uri, registry_uri=destino_uri)

    version_origen = origen.get_model_version_by_alias(nombre, alias)
    if existente and existente.tags.get("origen_version") == str(version_origen.version):
        logger.info(f"{nombre}@{alias}: ya sembrado (origen v{version_origen.version}), se omite")
        return "omitido"

    parametros = {}
    id_run_origen = version_origen.tags.get("source_run_id")
    if id_run_origen:
        try:
            parametros = origen.get_run(id_run_origen).data.params
        except MlflowException as error:
            logger.warning(f"{nombre}: no se pudieron leer los parámetros del run origen: {error}")

    with tempfile.TemporaryDirectory() as tmp:
        # `models:/` se resuelve contra el registry global, no contra un cliente: se apunta al
        # origen para bajar los artefactos y luego al destino para subirlos como un modelo más
        mlflow.set_tracking_uri(origen_uri)
        mlflow.set_registry_uri(origen_uri)
        ruta = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{nombre}/{version_origen.version}",
            dst_path=tmp,
        )
        mlflow.set_tracking_uri(destino_uri)
        mlflow.set_registry_uri(destino_uri)
        mlflow.set_experiment(EXPERIMENTO_SEMBRADO)
        with mlflow.start_run(run_name=f"{nombre}-v{version_origen.version}") as run:
            mlflow.log_params(parametros)
            mlflow.log_artifacts(str(Path(ruta)), artifact_path="model")
        nueva = mlflow.register_model(f"runs:/{run.info.run_id}/model", nombre)

    etiquetas = dict(version_origen.tags)
    etiquetas["origen_run_id"] = etiquetas.get("source_run_id", "")
    # La API lee los parámetros con source_run_id: debe apuntar al run local que sí los tiene
    etiquetas["source_run_id"] = run.info.run_id
    etiquetas["origen_version"] = str(version_origen.version)
    etiquetas["origen_uri"] = origen_uri
    for clave, valor in etiquetas.items():
        destino.set_model_version_tag(nombre, nueva.version, clave, valor)
    destino.set_registered_model_alias(nombre, alias, nueva.version)
    logger.success(f"{nombre}@{alias}: origen v{version_origen.version} -> local v{nueva.version}")
    return "copiado"


def main() -> int:
    origen_uri = os.getenv("MLFLOW_ORIGEN_URI", ORIGEN_POR_DEFECTO)
    destino_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not destino_uri:
        logger.error("MLFLOW_TRACKING_URI no está definida (MLflow local de destino)")
        return 1
    if origen_uri.rstrip("/") == destino_uri.rstrip("/"):
        logger.error("El origen y el destino son el mismo servidor: no hay nada que sembrar")
        return 1

    alias = alias_configurado()
    logger.info(f"Sembrando {alias} desde {origen_uri} hacia {destino_uri}")
    fallos = 0
    for pregunta, nombre in MODELOS_REGISTRADOS.items():
        try:
            sembrar_modelo(origen_uri, destino_uri, nombre, alias)
        except Exception as error:  # noqa: BLE001 - se reporta y se sigue con los demás
            fallos += 1
            logger.error(f"{pregunta}: no se pudo sembrar {nombre}@{alias}: {error}")
    # Solo falla si a algún modelo no le queda copia local: con salida distinta de cero,
    # depends_on (service_completed_successfully) no arranca la API
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
