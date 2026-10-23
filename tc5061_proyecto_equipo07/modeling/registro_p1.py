"""Registro de P1 en el Model Registry: toma un run candidato, lo registra y le pone un alias.

Uso (lo invoca MLproject; el run_id se copia del candidato elegido en la UI):
    mlflow run . -e register -P question=p1 -P run_id=<id del run candidato>

El comando no decide el campeón: registra el run que se le indica, con su linaje en las
etiquetas de la versión, y comprueba que `models:/<nombre>@<alias>` carga y predice.
"""

import os

from loguru import logger
import mlflow
from mlflow import MlflowClient
import typer

from tc5061_proyecto_equipo07 import config as cfg

app = typer.Typer(add_completion=False)

METRICAS = ["valid_mae", "valid_rmse", "test_mae", "test_rmse", "segundos_entrenamiento"]


def p1_descripcion_version(run) -> str:
    """Texto de la versión: qué modelo es, con qué métricas y de qué run sale."""
    metricas = ", ".join(
        f"{n}={run.data.metrics[n]:.4f}" for n in METRICAS if n in run.data.metrics
    )
    return (
        f"{run.info.run_name}: {run.data.tags.get('mlflow.note.content', '')} "
        f"Métricas: {metricas}. Run de origen: {run.info.run_id}."
    ).strip()


def p1_etiquetas_version(run) -> dict[str, str]:
    """Linaje de la versión: commit, datos, run de origen y, si hay, la búsqueda que la produjo."""
    etiquetas = {
        "question": run.data.tags.get("question", "p1"),
        "fase": run.data.tags.get("fase", cfg.FASE_ENTREGA),
        "commit": run.data.tags.get("commit", "desconocido"),
        "data_version": run.data.params.get("data_version", "desconocido"),
        "source_run_id": run.info.run_id,
        "valid_mae": f"{run.data.metrics['valid_mae']:.4f}",
    }
    if "busqueda_run_id" in run.data.tags:
        etiquetas["busqueda_run_id"] = run.data.tags["busqueda_run_id"]
    return etiquetas


def p1_registrar(run_id: str, alias: str, nombre: str = cfg.P1_MODELO_REGISTRADO) -> int:
    """Registra el modelo del run como una versión nueva, le pone el alias y regresa la versión."""
    cliente = MlflowClient()
    run = cliente.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"El run {run_id} no terminó (estado {run.info.status})")
    if not run.outputs.model_outputs:
        raise ValueError(f"El run {run_id} no tiene un modelo registrado en MLflow")
    modelo_id = run.outputs.model_outputs[0].model_id

    version = mlflow.register_model(f"models:/{modelo_id}", nombre)
    cliente.update_registered_model(
        nombre,
        description="Predice los viajes del día siguiente por estación-hora (P1). "
        "Cada versión lleva en sus etiquetas el commit, la versión de los datos y el run de origen.",
    )
    cliente.update_model_version(nombre, version.version, description=p1_descripcion_version(run))
    for etiqueta, valor in p1_etiquetas_version(run).items():
        cliente.set_model_version_tag(nombre, version.version, etiqueta, valor)
    cliente.set_registered_model_alias(nombre, alias, version.version)
    logger.success(f"{nombre} v{version.version} <- {run.info.run_name} (alias {alias})")
    return int(version.version)


def p1_verificar(alias: str, nombre: str = cfg.P1_MODELO_REGISTRADO) -> int:
    """Carga `models:/<nombre>@<alias>` y predice con su input_example; regresa cuántas filas."""
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}@{alias}")
    ejemplo = modelo.input_example
    if ejemplo is None:
        raise ValueError("El modelo no tiene input_example para comprobar la predicción")
    predicciones = modelo.predict(ejemplo)
    logger.success(f"models:/{nombre}@{alias} carga y predice {len(predicciones)} filas")
    return len(predicciones)


@app.command()
def registrar(
    run_id: str = typer.Option(..., help="Run del candidato que se va a registrar"),
    alias: str = typer.Option("champion", help="Alias de la versión (champion o challenger)"),
):
    """Registra el candidato indicado en el Registry, le pone el alias y comprueba que carga."""
    if not cfg.MLFLOW_TRACKING_URI:
        raise ValueError("MLFLOW_TRACKING_URI no está definida: exportarla o copiar .env.example")
    # `mlflow run` exporta MLFLOW_RUN_ID; aquí no se abre ningún run propio
    os.environ.pop("MLFLOW_RUN_ID", None)
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    p1_registrar(run_id, alias)
    p1_verificar(alias)


if __name__ == "__main__":
    app()
