"""Registro de P3 en el Model Registry: toma un run de modelo, lo registra y le pone un alias.

Uso (lo invoca MLproject; el run_id se copia del run elegido en la UI):
    mlflow run . -e register -P question=p3 -P run_id=<id del run del modelo>

El comando no decide el campeón: registra el run que se le indica, con su linaje en las
etiquetas de la versión, y comprueba que `models:/<nombre>@<alias>` carga y asigna grupos.
"""

import os

from loguru import logger
import mlflow
from mlflow import MlflowClient
import typer

from tc5061_proyecto_equipo07 import config as cfg

app = typer.Typer(add_completion=False)

# Cada una dice en qué conjunto se midió (ver P3_NOMBRES_METRICAS en train.py)
METRICAS_OBLIGATORIAS = ["train_silueta", "valid_ari"]
METRICAS = [
    "train_silueta",
    "train_davies_bouldin",
    "train_calinski_harabasz",
    "valid_ari",
    "train_tam_grupo_min",
]


def p3_descripcion_version(run) -> str:
    """Texto de la versión: qué modelo es, con qué métricas y de qué run sale."""
    metricas = ", ".join(
        f"{n}={run.data.metrics[n]:.4f}" for n in METRICAS if n in run.data.metrics
    )
    return (
        f"{run.info.run_name}: {run.data.tags.get('mlflow.note.content', '')} "
        f"Métricas: {metricas}. Run de origen: {run.info.run_id}."
    ).strip()


def p3_etiquetas_version(run) -> dict[str, str]:
    """Linaje de la versión: commit, datos, run de origen y, si hay, la búsqueda que la produjo."""
    etiquetas = {
        "question": run.data.tags.get("question", "p3"),
        "fase": run.data.tags.get("fase", cfg.FASE_ENTREGA),
        "commit": run.data.tags.get("commit", "desconocido"),
        "data_version": run.data.params.get("data_version", "desconocido"),
        "source_run_id": run.info.run_id,
        "train_silueta": f"{run.data.metrics['train_silueta']:.4f}",
        "valid_ari": f"{run.data.metrics['valid_ari']:.4f}",
    }
    for nombre in ("aceptado", "referencia", "busqueda_run_id"):
        if nombre in run.data.tags:
            etiquetas[nombre] = run.data.tags[nombre]
    return etiquetas


def p3_registrar(run_id: str, alias: str, nombre: str = cfg.P3_MODELO_REGISTRADO) -> int:
    """Registra el modelo del run como una versión nueva, le pone el alias y regresa la versión."""
    cliente = MlflowClient()
    run = cliente.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"El run {run_id} no terminó (estado {run.info.status})")
    if not run.outputs.model_outputs:
        raise ValueError(f"El run {run_id} no tiene un modelo registrado en MLflow")
    faltan = [n for n in METRICAS_OBLIGATORIAS if n not in run.data.metrics]
    if faltan:
        raise ValueError(
            f"El run {run_id} no tiene {', '.join(faltan)}: las métricas de P3 llevan el conjunto "
            "en el nombre; un run anterior a ese cambio debe volver a registrarse"
        )
    if run.data.tags.get("aceptado") == "false":
        logger.warning(
            f"{run.info.run_name} no cumple los criterios de aceptación (aceptado=false): "
            "el baseline sigue siendo el piso, no un candidato a campeón"
        )
    modelo_id = run.outputs.model_outputs[0].model_id

    version = mlflow.register_model(f"models:/{modelo_id}", nombre)
    cliente.update_registered_model(
        nombre,
        description="Agrupa las estaciones de MiBici por su perfil de uso (P3). "
        "Cada versión lleva en sus etiquetas el commit, la versión de los datos y el run de origen.",
    )
    cliente.update_model_version(nombre, version.version, description=p3_descripcion_version(run))
    for etiqueta, valor in p3_etiquetas_version(run).items():
        cliente.set_model_version_tag(nombre, version.version, etiqueta, valor)
    cliente.set_registered_model_alias(nombre, alias, version.version)
    logger.success(f"{nombre} v{version.version} <- {run.info.run_name} (alias {alias})")
    return int(version.version)


def p3_verificar(alias: str, nombre: str = cfg.P3_MODELO_REGISTRADO) -> int:
    """Carga `models:/<nombre>@<alias>` y asigna grupo a su input_example; regresa cuántas filas."""
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}@{alias}")
    ejemplo = modelo.input_example
    if ejemplo is None:
        raise ValueError("El modelo no tiene input_example para comprobar la asignación")
    grupos = modelo.predict(ejemplo)
    logger.success(f"models:/{nombre}@{alias} carga y asigna grupo a {len(grupos)} estaciones")
    return len(grupos)


@app.command()
def registrar(
    run_id: str = typer.Option(..., help="Run del modelo que se va a registrar"),
    alias: str = typer.Option("champion", help="Alias de la versión (champion o challenger)"),
):
    """Registra el run indicado en el Registry, le pone el alias y comprueba que carga."""
    if not cfg.MLFLOW_TRACKING_URI:
        raise ValueError("MLFLOW_TRACKING_URI no está definida: exportarla o copiar .env.example")
    # `mlflow run` exporta MLFLOW_RUN_ID; aquí no se abre ningún run propio
    os.environ.pop("MLFLOW_RUN_ID", None)
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    p3_registrar(run_id, alias)
    p3_verificar(alias)


if __name__ == "__main__":
    app()
