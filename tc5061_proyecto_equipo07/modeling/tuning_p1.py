"""Búsqueda de hiperparámetros de P1 con Optuna: un run padre por búsqueda y un run hijo por trial.

Uso (lo invoca MLproject; los parámetros vacíos conservan el valor de config.py):
    mlflow run . -e tune -P question=p1 -P model=xgb -P n_trials=20

Cada trial ajusta con (una muestra de) train y se evalúa en valid; test no interviene. El mejor
trial se vuelve a entrenar con todo train y queda como candidato `p1-<modelo>-optuna`.
"""

import os
import time

from loguru import logger
import mlflow
import numpy as np
import optuna
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset
from tc5061_proyecto_equipo07.modeling.train import (
    p1_ajustar_y_evaluar,
    p1_cargar_conjuntos,
    p1_commit_actual,
    p1_construir_modelo,
    p1_evaluar,
)

app = typer.Typer(add_completion=False)


def p1_espacio_xgb(trial: optuna.Trial) -> dict:
    """Hiperparámetros de XGBoost que explora la búsqueda."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }


def p1_espacio_hgb(trial: optuna.Trial) -> dict:
    """Hiperparámetros de HGB que explora la búsqueda."""
    # l2 y hoja mínima tienen piso: sin ellos la pérdida Poisson diverge (ver config.py)
    return {
        "max_iter": trial.suggest_int("max_iter", 100, 300, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 50, 1000, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.5, 20.0, log=True),
    }


ESPACIOS = {"xgb": p1_espacio_xgb, "hgb": p1_espacio_hgb}


def p1_etiquetas_busqueda(clave: str, descripcion: str | None = None) -> dict:
    """Etiquetas de los runs de la búsqueda (padre e hijos): stage_of_work=search."""
    etiquetas = {
        "equipo": "07",
        "question": "p1",
        "commit": p1_commit_actual(),
        "autor": "Daniel Sánchez Hernández",
        "stage_of_work": "search",
        "model_family": cfg.P1_MODELOS[clave][2],
        "fase": cfg.FASE_ENTREGA,
    }
    if descripcion:
        etiquetas["mlflow.note.content"] = descripcion
    return etiquetas


def p1_leer_opcion(valor: str, tipo, por_defecto):
    """Convierte el parámetro recibido como texto; vacío = el valor de config.py."""
    return tipo(valor) if valor.strip() else por_defecto


def p1_buscar(clave: str, train, valid, n_trials: int, timeout_min: int) -> optuna.Study:
    """Corre la búsqueda dentro del run padre activo; cada trial es un run hijo."""

    def objetivo(trial: optuna.Trial) -> float:
        hiperparametros = ESPACIOS[clave](trial)
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) as hijo:
            mlflow.set_tags(p1_etiquetas_busqueda(clave))
            mlflow.log_params(hiperparametros)
            trial.set_user_attr("run_id", hijo.info.run_id)
            modelo = p1_construir_modelo(clave, hiperparametros=hiperparametros)
            inicio = time.perf_counter()
            modelo.fit(train[cfg.P1_VARIABLES], train["viajes"])
            segundos = round(time.perf_counter() - inicio, 1)
            metricas, _ = p1_evaluar(modelo, valid, "valid")
            mlflow.log_metrics({**metricas, "segundos_entrenamiento": segundos})
        if not np.isfinite(metricas["valid_mae"]):
            raise optuna.TrialPruned()
        return metricas["valid_mae"]

    estudio = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.SEMILLA)
    )
    estudio.optimize(objetivo, n_trials=n_trials, timeout=timeout_min * 60 or None)
    return estudio


@app.command()
def buscar(
    model: str = typer.Option(..., help="xgb o hgb"),
    n_trials: str = "",
    fraccion_train: str = "",
    timeout_min: str = "",
    sin_candidato: bool = typer.Option(False, help="Solo la búsqueda, sin re-entrenar el mejor"),
):
    """Busca hiperparámetros del modelo y deja el mejor como candidato en MLflow."""
    clave = model.strip().lower()
    if clave not in ESPACIOS:
        raise typer.BadParameter(f"model debe ser xgb o hgb, no '{model}'")
    valores = cfg.P1_BUSQUEDA[clave]
    n_trials = p1_leer_opcion(n_trials, int, valores["n_trials"])
    fraccion = p1_leer_opcion(fraccion_train, float, valores["fraccion_train"])
    timeout = p1_leer_opcion(timeout_min, int, valores["timeout_min"])
    if not 0 < fraccion <= 1:
        raise typer.BadParameter(f"fraccion_train debe estar en (0, 1], no {fraccion}")
    if not cfg.MLFLOW_TRACKING_URI:
        raise ValueError("MLFLOW_TRACKING_URI no está definida: exportarla o copiar .env.example")

    # `mlflow run` exporta MLFLOW_RUN_ID; el run padre de la búsqueda debe ser uno propio
    os.environ.pop("MLFLOW_RUN_ID", None)
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.EXPERIMENTO_MLFLOW)

    conjuntos = p1_cargar_conjuntos()
    # Solo se muestrea train: valid queda completo y en orden temporal para elegir
    train = conjuntos["train"]
    if fraccion < 1:
        train = train.sample(frac=fraccion, random_state=cfg.SEMILLA).reset_index(drop=True)
    logger.info(f"Búsqueda {clave}: {n_trials} trials, {timeout} min, {len(train)} filas de train")

    nombre_padre = f"{cfg.P1_MODELOS[clave + '_optuna'][0]}-busqueda"
    with mlflow.start_run(run_name=nombre_padre) as padre:
        mlflow.set_tags(
            p1_etiquetas_busqueda(
                clave,
                "Búsqueda con Optuna (TPE); cada trial es un run hijo. Objetivo: valid_mae. "
                "Test no se usa para decidir.",
            )
        )
        mlflow.log_params(
            {
                "rango_datos": f"{cfg.MES_INICIO} a {cfg.MES_FIN}",
                "data_version": dataset.version_datos(),
                "semilla": cfg.SEMILLA,
                "objetivo": "valid_mae",
                "n_trials_max": n_trials,
                "timeout_min": timeout,
                "fraccion_train": fraccion,
                "n_train_busqueda": len(train),
            }
        )
        estudio = p1_buscar(clave, train, conjuntos["valid"], n_trials, timeout)
        terminados = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not terminados:
            raise RuntimeError("Ningún trial terminó: revisar los runs hijos en MLflow")
        mejor = estudio.best_trial
        mlflow.log_params({f"mejor_{nombre}": valor for nombre, valor in mejor.params.items()})
        mlflow.log_param("best_child_run_id", mejor.user_attrs["run_id"])
        mlflow.log_metrics(
            {"n_trials_terminados": len(terminados), "mejor_valid_mae": mejor.value}
        )
    logger.success(f"Mejor trial {mejor.number}: valid_mae {mejor.value:.4f} | {mejor.params}")

    if not sin_candidato:
        p1_ajustar_y_evaluar(
            f"{clave}_optuna",
            conjuntos,
            registrar=True,
            hiperparametros=mejor.params,
            etiquetas_extra={"busqueda_run_id": padre.info.run_id},
        )


if __name__ == "__main__":
    app()
